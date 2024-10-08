import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import einops
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models.unets.unet_motion_model import MotionAdapter
from diffusers.schedulers import (
    DDIMScheduler,
    DDIMInverseScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.free_init_utils import FreeInitMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.animatediff.pipeline_output import AnimateDiffPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AnimateDiffPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    LoraLoaderMixin,
    FreeInitMixin,
):

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["feature_extractor", "image_encoder", "motion_adapter"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: Union[UNet2DConditionModel, UNetMotionModel],
        motion_adapter: MotionAdapter,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        super().__init__()
        if isinstance(unet, UNet2DConditionModel):
            unet = UNetMotionModel.from_unet2d(unet, motion_adapter)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            motion_adapter=motion_adapter,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=False, vae_scale_factor=self.vae_scale_factor)
        self.image_unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet").cuda()
        self.image_vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae").cuda()
        
        self.i_sigma_begin = 4
        self.i_sigma_end = 4
        self.v_sigma_begin = 4
        self.v_sigma_end = 4
        self.rho = 7

        self.free_noise_enabled = False
        self.pre_inference_timesteps = 50
        self.i_guidance_scale_sigmas = self.get_sigmas_karras(self.pre_inference_timesteps, self.i_sigma_begin, self.i_sigma_end, self.rho)
        self.v_guidance_scale_sigmas = self.get_sigmas_karras(self.pre_inference_timesteps, self.v_sigma_begin, self.v_sigma_end, self.rho)
        
        print("Successfully initialized IV-Mixed Sampler Pipeline")
        
    def set_sigma_rho(self, i_sigma_begin=4, i_sigma_end=4, v_sigma_begin=4, v_sigma_end=4, rho=7.0):
        self.i_sigma_begin = i_sigma_begin
        self.i_sigma_end = i_sigma_end
        self.v_sigma_begin = v_sigma_begin
        self.v_sigma_end = v_sigma_end
        self.rho = rho
        self.i_guidance_scale_sigmas = self.get_sigmas_karras(self.pre_inference_timesteps, self.i_sigma_begin, self.i_sigma_end, self.rho)
        self.v_guidance_scale_sigmas = self.get_sigmas_karras(self.pre_inference_timesteps, self.v_sigma_begin, self.v_sigma_end, self.rho)
        
    def get_sigmas_karras(self, n, sigma_begin, sigma_end, rho=7.0, device="cpu"): # TODO: can set sigma_begin=[6,4,2], sigma_end=[2,4,6], rho=[1/7,7]
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        begin_inv_rho = sigma_begin ** (1 / rho)
        end_inv_rho = sigma_end ** (1 / rho)
        sigmas = (begin_inv_rho + ramp * (end_inv_rho - begin_inv_rho)) ** rho

        return sigmas

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def decode_latents(self, latents):
        latents = latents.float()
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
        video = video.half()
        return video

    def image_decode_latents(self, latents):
        latents = 1 / self.image_vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.image_vae.decode(latents).sample
        video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
        video = video.half()
        return video
    
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        decode_chunk_size: int = 16,
        **kwargs,
    ):


        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, (str, dict)):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        if self.free_noise_enabled:
            prompt_embeds, negative_prompt_embeds = self._encode_prompt_free_noise(
                prompt=prompt,
                num_frames=num_frames,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # prompt_embeds = prompt_embeds.repeat_interleave(repeats=num_frames, dim=0)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_videos_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # 7. Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        num_free_init_iters = self._free_init_num_iters if self.free_init_enabled else 1
        for free_init_iter in range(num_free_init_iters):
            if self.free_init_enabled:
                latents, timesteps = self._apply_free_init(
                    latents, free_init_iter, num_inference_steps, device, latents.dtype, generator
                )

            self._num_timesteps = len(timesteps)
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order


            optim_steps = 40
            self.inv_scheduler = DDIMInverseScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                                                  subfolder='scheduler')
            self.inv_scheduler.set_timesteps(num_inference_steps, device=device)
            # 8. Denoising loop
            with self.progress_bar(total=self._num_timesteps) as progress_bar:
                for i, t in enumerate(timesteps):
                    
                    if i < optim_steps:
                        
                        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                        # TODO Zigzag First Order Optim
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        noise_pred = self.unet(
                                            latent_model_input,
                                            t,
                                            encoder_hidden_states=prompt_embeds,
                                            cross_attention_kwargs=cross_attention_kwargs,
                                            added_cond_kwargs=added_cond_kwargs,
                                        ).sample
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + 4. * (noise_pred_text - noise_pred_uncond)
                        self.scheduler._step_index = None
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                        
                        # TODO Zigzag Second Order Optim
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, prev_t)
                        noise_pred = self.unet(
                                            latent_model_input,
                                            prev_t,
                                            encoder_hidden_states=prompt_embeds,
                                            cross_attention_kwargs=cross_attention_kwargs,
                                            added_cond_kwargs=added_cond_kwargs,
                                        ).sample
                        
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond - 4. * (noise_pred_text - noise_pred_uncond)
                        self.inv_scheduler._step_index = None
                        latents = self.inv_scheduler.step(noise_pred, t, latents).prev_sample
                        
                        
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    self.scheduler._step_index = None
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        # 9. Post processing
        if output_type == "latent":
            video = latents
        else:
            video_tensor = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

        # 10. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return AnimateDiffPipelineOutput(frames=video)

    @torch.no_grad()
    def forward_IVIV(
        self,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        decode_chunk_size: int = 16,
        imagestone_interval: List[int] = [0, 49],
        zz: int = 1000,
        **kwargs,
    ):


        self.image_unet = self.image_unet.cuda().half()
        self.image_vae = self.vae.cuda().float()
        self.vae = self.vae.cuda().float()
        self.unet = self.unet.cuda().half()
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, (str, dict)):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        if self.free_noise_enabled:
            prompt_embeds, negative_prompt_embeds = self._encode_prompt_free_noise(
                prompt=prompt,
                num_frames=num_frames,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            image_prompt_embeds = prompt_embeds
            image_prompt_embeds = image_prompt_embeds.repeat_interleave(repeats=num_frames, dim=0)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_videos_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # 7. Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        num_free_init_iters = self._free_init_num_iters if self.free_init_enabled else 1
        for free_init_iter in range(num_free_init_iters):
            if self.free_init_enabled:
                latents, timesteps = self._apply_free_init(
                    latents, free_init_iter, num_inference_steps, device, latents.dtype, generator
                )

            self._num_timesteps = len(timesteps)
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order


            self.inv_scheduler = DDIMInverseScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                                                  subfolder='scheduler')
            self.inv_scheduler.set_timesteps(num_inference_steps, device=device)
            # 8. Denoising loop
            optim_steps_0 = imagestone_interval[0]
            optim_steps_1 = imagestone_interval[1]
            with self.progress_bar(total=self._num_timesteps) as progress_bar:
                for i, t in enumerate(timesteps):
                    
                    if optim_steps_0 <= i <= optim_steps_1:
                        self.scheduler.set_timesteps(num_inference_steps, device=device)
                        print(f"Optimizing at step {i}")
                        prev_t = max(0, t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)
                        prev_prev_t = max(0, t - self.scheduler.config.num_train_timesteps * 2 // self.scheduler.num_inference_steps)
                        print(f"sigmas: {self.i_guidance_scale_sigmas[i]}, {self.v_guidance_scale_sigmas[i]}")

                        # ---------------------------------
                        if i % zz < zz - 1:
                            bs , frame_num = latents.shape[0], latents.shape[2]
                            new_image_latent = einops.rearrange(latents, 'b c f h w -> (b f) c h w')
                            # gaussian_noise = (image_latent - clean_image_latent * (self.scheduler.alphas_cumprod[prev_t] ** (0.5))) / ((1 - self.scheduler.alphas_cumprod[prev_t]) ** (0.5))                            
                            image_latent_model_input = torch.cat([new_image_latent] * 2) if self.do_classifier_free_guidance else new_image_latent
                            image_latent_model_input = self.scheduler.scale_model_input(image_latent_model_input, prev_t)
                            image_noise_pred = self.image_unet(
                                                image_latent_model_input,
                                                prev_t,
                                                encoder_hidden_states=image_prompt_embeds,
                                                timestep_cond=None,
                                                cross_attention_kwargs=cross_attention_kwargs,
                                                added_cond_kwargs=added_cond_kwargs,
                                            ).sample
                            if self.do_classifier_free_guidance:
                                image_noise_pred_uncond, image_noise_pred_text = image_noise_pred.chunk(2)
                                image_noise_pred = image_noise_pred_uncond + self.i_guidance_scale_sigmas[i] * (image_noise_pred_text - image_noise_pred_uncond)
                            self.scheduler._step_index = None
                            new_image_latent = self.scheduler.step(image_noise_pred, prev_t, new_image_latent, **extra_step_kwargs).prev_sample
                            new_image_latent = einops.rearrange(new_image_latent, '(b f) c h w -> b c f h w', f=frame_num, b=bs)
                            latents = new_image_latent
                        else:
                            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, prev_t)
                            noise_pred = self.unet(
                                                latent_model_input,
                                                prev_t,
                                                encoder_hidden_states=prompt_embeds,
                                                cross_attention_kwargs=cross_attention_kwargs,
                                                added_cond_kwargs=added_cond_kwargs,
                                            ).sample
                            if self.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + self.v_guidance_scale_sigmas[i] * (noise_pred_text - noise_pred_uncond)
                            self.scheduler._step_index = None
                            latents = self.scheduler.step(noise_pred, prev_t, latents, **extra_step_kwargs)
                            latents = latents.prev_sample
                                                                         
                        # TODO Zigzag First Order Optim
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
 
                        noise_pred = self.unet(
                                            latent_model_input,
                                            t,
                                            encoder_hidden_states=prompt_embeds,
                                            cross_attention_kwargs=cross_attention_kwargs,
                                            added_cond_kwargs=added_cond_kwargs,
                                        ).sample
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.v_guidance_scale_sigmas[i] * (noise_pred_text - noise_pred_uncond)
                        self.scheduler._step_index = None
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                        pred_original_samples = latents.pred_original_sample
                        latents = latents.prev_sample


                        # -----------
                        if i % zz < zz - 1:
                            bs , frame_num = latents.shape[0], latents.shape[2]
                            new_image_latent = einops.rearrange(latents, 'b c f h w -> (b f) c h w')
                            image_latent_model_input = torch.cat([new_image_latent] * 2) if self.do_classifier_free_guidance else new_image_latent
                            image_latent_model_input = self.scheduler.scale_model_input(image_latent_model_input, prev_prev_t)
                            image_noise_pred = self.image_unet(
                                                image_latent_model_input,
                                                prev_prev_t,
                                                encoder_hidden_states=image_prompt_embeds,
                                                timestep_cond=None,
                                                cross_attention_kwargs=cross_attention_kwargs,
                                                added_cond_kwargs=added_cond_kwargs,
                                            ).sample
                            if self.do_classifier_free_guidance:
                                image_noise_pred_uncond, image_noise_pred_text = image_noise_pred.chunk(2)
                                image_noise_pred = image_noise_pred_uncond - self.i_guidance_scale_sigmas[i] * (image_noise_pred_text - image_noise_pred_uncond)
                            self.inv_scheduler._step_index = None
                            new_image_latent = self.inv_scheduler.step(image_noise_pred, prev_t, new_image_latent).prev_sample
                            new_image_latent = einops.rearrange(new_image_latent, '(b f) c h w -> b c f h w', f=frame_num, b=bs)
                            latents = new_image_latent
                        else:
                            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, prev_prev_t)
                            noise_pred = self.unet(
                                                latent_model_input,
                                                prev_prev_t,
                                                encoder_hidden_states=prompt_embeds,
                                                cross_attention_kwargs=cross_attention_kwargs,
                                                added_cond_kwargs=added_cond_kwargs,
                                            ).sample
                            
                            if self.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond - self.v_guidance_scale_sigmas[i] * (noise_pred_text - noise_pred_uncond)
                            self.inv_scheduler._step_index = None
                            latents = self.inv_scheduler.step(noise_pred, prev_t, latents).prev_sample                 
                        # TODO Zigzag Second Order Optim
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, prev_t)
                        noise_pred = self.unet(
                                            latent_model_input,
                                            prev_t,
                                            encoder_hidden_states=prompt_embeds,
                                            cross_attention_kwargs=cross_attention_kwargs,
                                            added_cond_kwargs=added_cond_kwargs,
                                        ).sample
                        
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond - self.v_guidance_scale_sigmas[i] * (noise_pred_text - noise_pred_uncond)
                        self.inv_scheduler._step_index = None
                        latents = self.inv_scheduler.step(noise_pred, t, latents).prev_sample
                        self.scheduler.set_timesteps(num_inference_steps, device=device)
                        
                    # TODO: original pipeline
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    self.scheduler._step_index = None
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        # 9. Post processing
        if output_type == "latent":
            video = latents
        else:
            video_tensor = self.decode_latents(latents.float())
            video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

        # 10. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return AnimateDiffPipelineOutput(frames=video)