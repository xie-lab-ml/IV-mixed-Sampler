import torch, os, sys
from tqdm import tqdm 
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models import MotionAdapter
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from accelerate.utils import ProjectConfiguration, set_seed
import argparse
import pandas as pd


from vico.video_crafter_diffusers.unet_3d_videocrafter import UNet3DVideoCrafterConditionModel
from vico.video_crafter_diffusers.pipeline_text_to_video_videocrafter_iviv_sampler import TextToVideoVideoCrafterPipeline


def get_args():
      parser = argparse.ArgumentParser()

      # get the VDM pipeline
      parser.add_argument(
            "--pipeline",
            type=str,
            choices=['Animatediff', 'ModelScope', 'VideoCrafter'],
            default='Animatediff',
      )
      parser.add_argument(
            "--interval_begin",
            type=int,
            default=0,
      )
      parser.add_argument(
            "--interval_end",
            type=int,
            default=49,
      )
      parser.add_argument(
            "--zz",
            type=int,
            default=1000,
      )
      # dynamic CFG scale
      parser.add_argument(
            "--i_sigma_begin",
            type=int,
            choices=[2, 4, 6],
            default=4,
      )
      parser.add_argument(
            "--i_sigma_end",
            type=int,
            choices=[2, 4, 6],
            default=4,
      )
      parser.add_argument(
            "--v_sigma_begin",
            type=int,
            choices=[2, 4, 6],
            default=4,
      )
      parser.add_argument(
            "--v_sigma_end",
            type=int,
            choices=[2, 4, 6],
            default=4,
      )
      parser.add_argument(
            "--rho",
            type=float,
            default=7.,
      )
      parser.add_argument(
            "--lora",
            type=str,
            default='None',
            choices=['None', 'amechass', 'beauty'],
      )

      # prompt
      parser.add_argument('--prompt', type=str, default='Two horses race across a grassy field at sunset.')
      parser.add_argument('--seed', type=int, default=23523)
      parser.add_argument("--inference-step", type=int, default=50, help="Inference Steps")

      args = parser.parse_args()

      return args


def main(args):
      set_seed(args.seed)
      
      print("Model: ", args.pipeline, "\n",
            "i_sigma_begin: ", args.i_sigma_begin, "\n",
            "i_sigma_end: ", args.i_sigma_end, "\n",
            "v_sigma_begin: ", args.v_sigma_begin, "\n",
            "v_sigma_end: ", args.v_sigma_end, "\n",
            "rho: ", args.rho, "\n",
            "prompt: ", args.prompt, "\n",
            "seed: ", args.seed, "\n",
            "inference_step: ", args.inference_step)
      
      if args.pipeline == 'Animatediff':
            from utils.iv_sampler_animatediff import AnimateDiffPipeline

            adapter = MotionAdapter.from_pretrained(
                              "guoyww/animatediff-motion-adapter-v1-5-3", 
                              torch_dtype=torch.float16)

            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

            pipe = AnimateDiffPipeline.from_pretrained(model_id, 
                              motion_adapter=adapter).to(dtype=torch.float16,device=torch.device("cuda"))
            
            scheduler = DDIMScheduler.from_pretrained(
                              model_id, subfolder="scheduler", 
                              clip_sample=False, timestep_spacing="linspace", 
                              steps_offset=1)
            
            pipe.scheduler = scheduler
            if args.lora != 'None':
                  pipe.image_unet = UNet2DConditionModel.from_pretrained(f"./lora/{args.lora}", subfolder="unet").cuda()
                  pipe.image_vae = AutoencoderKL.from_pretrained(f"./lora/{args.lora}", subfolder="vae").cuda()
                  print(f"Load lora weight from ./lora/{args.lora}")
            pipe.enable_vae_slicing()


      elif args.pipeline == 'ModelScope':
            from utils.iv_sampler_modelscope_sdv15 import ModelScopeT2V

            pipe = ModelScopeT2V.from_pretrained(
                              "ali-vilab/text-to-video-ms-1.7b", 
                              torch_dtype=torch.float16, 
                              variant="fp16")
            
            scheduler = DDIMScheduler.from_pretrained(
                              "ali-vilab/text-to-video-ms-1.7b", 
                              subfolder="scheduler", 
                              clip_sample=False, 
                              timestep_spacing="linspace", 
                              steps_offset=1)

            pipe.scheduler = scheduler
            pipe.enable_vae_slicing()

      elif args.pipeline == 'VideoCrafter':
            from vico.video_crafter_diffusers.unet_3d_videocrafter import UNet3DVideoCrafterConditionModel
            from vico.video_crafter_diffusers.pipeline_text_to_video_videocrafter_iviv_sampler import TextToVideoVideoCrafterPipeline

            pipe = TextToVideoVideoCrafterPipeline.from_pretrained(
                  "cerspense/zeroscope_v2_576w", 
                  torch_dtype=torch.float16
            )

            pipe.unet = UNet3DVideoCrafterConditionModel.from_pretrained(
                  "/home/shaoshitong/.cache/huggingface/hub/adamdad/videocrafterv2_diffusers",
                  torch_dtype=torch.float16,
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                  pipe.scheduler.config, 
                  algorithm_type="sde-dpmsolver++"
            )

      pipe.set_sigma_rho(args.i_sigma_begin, args.i_sigma_end, args.v_sigma_begin, args.v_sigma_end, args.rho)
      pipe.enable_model_cpu_offload()
      pipe = pipe.to("cuda")



      if args.pipeline == 'Animatediff':
            from diffusers.utils import export_to_gif, export_to_video
            
            iv_mix_output = pipe.forward_IVIV(
                prompt=args.prompt,
                num_inference_steps=args.inference_step,
                num_frames=16,
                width=512,
                height=512,
                imagestone_interval = [args.interval_begin, args.interval_end],
                zz=args.zz,
                
            ) # 1.32
            
            iv_mix_frames = iv_mix_output.frames[0]
            export_to_video(iv_mix_frames, f"./{args.prompt}_{args.pipeline}_iv_mix.mp4")

            from diffusers import AnimateDiffPipeline

            adapter = MotionAdapter.from_pretrained(
                              "guoyww/animatediff-motion-adapter-v1-5-3", 
                              torch_dtype=torch.float16)

            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

            pipe = AnimateDiffPipeline.from_pretrained(model_id, 
                              motion_adapter=adapter).to(dtype=torch.float16,device=torch.device("cuda"))
            
            scheduler = DDIMScheduler.from_pretrained(
                              model_id, subfolder="scheduler", 
                              clip_sample=False, timestep_spacing="linspace", 
                              steps_offset=1)
            
            pipe.scheduler = scheduler
            pipe.enable_vae_slicing()
            
            standard_output = pipe(
                        prompt=args.prompt,
                        num_inference_steps=args.inference_step,
                        num_frames=16,
                        width=512,
                        height=512,
                  ) # 00:21

            standard_frames = standard_output.frames[0]
            export_to_video(standard_frames, f"./{args.prompt}_{args.pipeline}_standard.mp4")

      elif args.pipeline == 'ModelScope':
            from diffusers.utils import export_to_gif, export_to_video

            iv_mix_output = pipe.forward_IVIV(
                  prompt=args.prompt,
                  num_inference_steps=args.inference_step,
                  num_frames=16,
                  width=512,
                  height=512,
                  imagestone_interval = [args.interval_begin, args.interval_end],
            )

            iv_mix_frames = iv_mix_output.frames[0]
            export_to_video(iv_mix_frames, f"./{args.prompt}_{args.pipeline}_iv_mix.mp4")
      
            standard_output = pipe(
                  prompt=args.prompt,
                  num_inference_steps=args.inference_step,
                  num_frames=16,
                  width=512,
                  height=512,
            )

            standard_frames = standard_output.frames[0]
            export_to_video(standard_frames, f"./{args.prompt}_{args.pipeline}_standard.mp4")
            
      elif args.pipeline == 'VideoCrafter':
            from vico.video_crafter_diffusers.pipeline_text_to_video_videocrafter import export_to_video

            iv_mix_output = pipe(
                prompt=args.prompt,
                num_inference_steps=args.inference_step,
                num_frames=16,
                width=512,
                height=512,
                imagestone_interval = [args.interval_begin, args.interval_end],
            )
            
            iv_mix_frames = iv_mix_output.frames[0]
            export_to_video(iv_mix_frames, f"./{args.prompt}_{args.pipeline}_iv_mix.mp4")
            
            del pipe

            from vico.video_crafter_diffusers.unet_3d_videocrafter import UNet3DVideoCrafterConditionModel
            from vico.video_crafter_diffusers.pipeline_text_to_video_videocrafter import TextToVideoVideoCrafterPipeline

            pipe = TextToVideoVideoCrafterPipeline.from_pretrained(
                  "cerspense/zeroscope_v2_576w", 
                  torch_dtype=torch.float16
            )

            # "adamdad/videocrafterv2_diffusers"
            pipe.unet = UNet3DVideoCrafterConditionModel.from_pretrained(
                  "/home/shaoshitong/.cache/huggingface/hub/adamdad/videocrafterv2_diffusers",
                  # "adamdad/videocrafterv2_diffusers",
                  torch_dtype=torch.float16,
            )

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                  pipe.scheduler.config, 
                  algorithm_type="sde-dpmsolver++"
            )

            pipe.enable_model_cpu_offload()
            pipe = pipe.to("cuda")
                  
            standard_output = pipe(
                prompt=args.prompt,
                num_inference_steps=args.inference_step,
                num_frames=16,
                width=512,
                height=512,
            )

            standard_frames = standard_output.frames[0]
            export_to_video(standard_frames, f"./{args.prompt}_{args.pipeline}_standard.mp4")



if __name__ == '__main__':
      args = get_args()
      main(args)