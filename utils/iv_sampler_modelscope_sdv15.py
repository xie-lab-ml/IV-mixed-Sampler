import inspect, copy
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel, UNetMotionModel
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

import torch
import einops
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    DDIMInverseScheduler,
    PNDMScheduler,
)
from diffusers import TextToVideoSDPipeline, StableDiffusionPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_output import TextToVideoSDPipelineOutput
import torch.nn.functional as F

class ModelScopeT2V(TextToVideoSDPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
    ):
        super(ModelScopeT2V, self).__init__(
                vae = vae,
                text_encoder = text_encoder,
                tokenizer = tokenizer,
                unet = unet,
                scheduler = scheduler 
        )
        self.recall_timesteps = 1
        self.ensemble = 1
        self.ensemble_rate = 0.1
        self.pre_num_inference_steps = 50
        self.fast_ensemble = False
        self.momentum = 0.
        self.traj_momentum = 0.05
        self.ensemble_guidance_scale = False
        self.noise_type = "uniform"


        # --------- Image Stone -------------
        self.img_pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(dtype=torch.float16,device=torch.device("cuda"))
        self.image_unet = self.img_pipeline.unet
        self.image_vae = self.img_pipeline.vae
        self.free_noise_enabled = False
        self.pre_inference_timesteps = 50
        self.i_sigma_begin = 4
        self.i_sigma_end = 4
        self.v_sigma_begin = 4
        self.v_sigma_end = 4
        self.rho = 7
        
        self.i_guidance_scale_sigmas = self.get_sigmas_karras(self.pre_inference_timesteps, self.i_sigma_begin, self.i_sigma_end, self.rho)
        self.v_guidance_scale_sigmas = self.get_sigmas_karras(self.pre_inference_timesteps, self.v_sigma_begin, self.v_sigma_end, self.rho)
        print("Successfully initialized IV-Mixed Sampler Pipeline")

    def set_sigma_rho(self, i_sigma_begin=4, i_sigma_end=4, v_sigma_begin=4, v_sigma_end=4, rho=7.0):
        self.i_sigma_begin = i_sigma_begin
        self.i_sigma_end = i_sigma_end
        self.v_sigma_begin = v_sigma_begin
        self.v_sigma_end = v_sigma_end
        self.rho = rho
        
    def get_sigmas_karras(self, n, sigma_begin, sigma_end, rho=7.0, device="cpu"): # TODO: can set sigma_begin=[6,4,2], sigma_end=[2,4,6], rho=[1/7,7]
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        begin_inv_rho = sigma_begin ** (1 / rho)
        end_inv_rho = sigma_end ** (1 / rho)
        sigmas = (begin_inv_rho + ramp * (end_inv_rho - begin_inv_rho)) ** rho

        return sigmas
    
    @torch.no_grad()
    def forward_IVIV(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        decode_chunk_size: int = 16,
        imagestone_interval: List[int] = [0, 49],
        *args,
        **kwargs
    ):
        self.image_unet = self.image_unet.cuda().half()
        self.image_vae = self.vae.cuda().float()
        self.vae = self.vae.cuda().float()
        self.unet = self.unet.cuda().half()

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            kwargs.get("cross_attention_kwargs", None).get("scale", None) if kwargs.get("cross_attention_kwargs", None) is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=kwargs.get("prompt_embeds", None),
            negative_prompt_embeds=kwargs.get("negative_prompt_embeds", None),
            lora_scale=text_encoder_lora_scale,
            clip_skip=kwargs.get("clip_skip", None),
        )

        img_prompt_embeds, img_negative_prompt_embeds = self.img_pipeline.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=kwargs.get("prompt_embeds", None),
                negative_prompt_embeds=kwargs.get("negative_prompt_embeds", None),
                lora_scale=text_encoder_lora_scale,
                clip_skip=kwargs.get("clip_skip", None),
            )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            img_prompt_embeds = torch.cat([img_negative_prompt_embeds, img_prompt_embeds])
        image_prompt_embeds = img_prompt_embeds.repeat_interleave(repeats=num_frames, dim=0)


        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            kwargs.get("generator", None),
            kwargs.get("latents", None)
        )
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        added_cond_kwargs = None
        extra_step_kwargs = self.prepare_extra_step_kwargs(kwargs.get("generator", None), kwargs.get("eta", 0.))

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        self.inv_scheduler = DDIMInverseScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",
                                                                  subfolder='scheduler')
        self.inv_scheduler.set_timesteps(num_inference_steps, device=device)
        # 8. Denoising loop
        optim_steps_0 = imagestone_interval[0]
        optim_steps_1 = imagestone_interval[1]

        width, height = latents.shape[3], latents.shape[4]
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if optim_steps_0 <= i <= optim_steps_1:
                    self.scheduler.set_timesteps(num_inference_steps, device)
                    print(f"Optimizing at step {i}")

                    prev_t = max(0, t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)
                    prev_prev_t = max(0, t - self.scheduler.config.num_train_timesteps * 2 // self.scheduler.num_inference_steps)
                    print(f"sigmas: {self.i_guidance_scale_sigmas[i]}, {self.v_guidance_scale_sigmas[i]}")

                    bs , frame_num = latents.shape[0], latents.shape[2]
                    new_image_latent = einops.rearrange(latents, 'b c f h w -> (b f) c h w')
                    new_image_latent = F.interpolate(new_image_latent, size=(64, 64), mode='nearest')
                    image_latent_model_input = torch.cat([new_image_latent] * 2) if do_classifier_free_guidance else new_image_latent
                    image_latent_model_input = self.scheduler.scale_model_input(image_latent_model_input, t)
                    image_noise_pred = self.image_unet(
                        image_latent_model_input,
                        t,
                        encoder_hidden_states=image_prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                        added_cond_kwargs=None,
                    ).sample
                    if do_classifier_free_guidance:
                        image_noise_pred_uncond, image_noise_pred_text = image_noise_pred.chunk(2)
                        image_noise_pred = image_noise_pred_uncond + self.i_guidance_scale_sigmas[i] * (image_noise_pred_text - image_noise_pred_uncond)
                    self.scheduler._step_index = None
                    new_image_latent = self.scheduler.step(image_noise_pred, t, new_image_latent, **extra_step_kwargs).prev_sample
                    new_image_latent = F.interpolate(new_image_latent, size=(width, height), mode='nearest')                    
                    new_image_latent = einops.rearrange(new_image_latent, '(b f) c h w -> b c f h w', f=frame_num, b=bs)
                    latents = new_image_latent

                    #TODO: Zigzag Denosing for Video Latents
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, prev_t)

                    noise_pred = self.unet(
                        latent_model_input,
                        prev_t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.v_guidance_scale_sigmas[i] * (noise_pred_text - noise_pred_uncond)

                    # reshape latents
                    bsz, channel, frames, width, height = latents.shape
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                    noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                    self.scheduler._step_index = None
                    latents = self.scheduler.step(noise_pred, prev_t, latents, **extra_step_kwargs)
                    pred_original_samples = latents.pred_original_sample
                    latents = latents.prev_sample

                    # reshape latents back
                    latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                    pred_original_samples = pred_original_samples[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)


                    bs , frame_num = latents.shape[0], latents.shape[2]
                    new_image_latent = einops.rearrange(latents, 'b c f h w -> (b f) c h w')                    
                    new_image_latent = F.interpolate(new_image_latent, size=(64, 64), mode='nearest')
                    image_latent_model_input = torch.cat([new_image_latent] * 2) if do_classifier_free_guidance else new_image_latent
                    image_latent_model_input = self.scheduler.scale_model_input(image_latent_model_input, prev_prev_t)
                    image_noise_pred = self.image_unet(
                        image_latent_model_input,
                        prev_prev_t,
                        encoder_hidden_states=image_prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                        added_cond_kwargs=None,
                    ).sample
                    if do_classifier_free_guidance:
                        image_noise_pred_uncond, image_noise_pred_text = image_noise_pred.chunk(2)
                        image_noise_pred = image_noise_pred_uncond - self.i_guidance_scale_sigmas[i] * (image_noise_pred_text - image_noise_pred_uncond)

                    self.inv_scheduler._step_index = None
                    new_image_latent = self.inv_scheduler.step(image_noise_pred, prev_t, new_image_latent).prev_sample
                    new_image_latent = F.interpolate(new_image_latent, size=(width, height), mode='nearest')
                    new_image_latent = einops.rearrange(new_image_latent, '(b f) c h w -> b c f h w', f=frame_num, b=bs)
                    latents = new_image_latent
                    latents = latents.squeeze(3)


                    # TODO: Zigzag Inverse for Video Latents
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, prev_t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        prev_t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond - self.v_guidance_scale_sigmas[i] * (noise_pred_text - noise_pred_uncond)

                    # reshape latents
                    bsz, channel, frames, width, height = latents.shape
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                    noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                    self.inv_scheduler._step_index = None
                    latents = self.inv_scheduler.step(noise_pred, t, latents).prev_sample
                    latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                    self.scheduler.set_timesteps(num_inference_steps, device=device)


                # TODO: Original Pipeline
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=kwargs.get("cross_attention_kwargs", None),
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                self.scheduler._step_index = None
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post processing
        if output_type == "latent":
            video = latents
        else:
            video_tensor = self.decode_latents(latents.float())
            video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

        self.image_vae = self.vae.cuda().half()
        self.vae = self.vae.cuda().half()
        # 9. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return TextToVideoSDPipelineOutput(frames=video)
