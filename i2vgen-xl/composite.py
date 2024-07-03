import os
import sys
from pathlib import Path
import torch
import argparse
import logging
from omegaconf import OmegaConf
from PIL import Image
import json

# HF imports
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
)
from diffusers.utils import load_image, export_to_video, export_to_gif
from functools import partial

# Project imports
from utils import (
    seed_everything,
    load_video_frames,
    convert_video_to_frames
)
from pipelines.pipeline_i2vgen_xl import I2VGenXLPipeline
from pnp_utils import (
    register_spatial_attention_pnp,
    register_temp_attention_pnp,
    register_temp_conv_injection,
    register_out_conv_injection,
    register_resnet_injection,
    modify_diffuser_attention_forward
)

from common import PRETRAINED_MODEL_PATH


def init_pnp(pipe, scheduler, config):
    conv_injection_t = int(config.n_steps * config.pnp_f_t)
    spatial_attn_qk_injection_t = int(config.n_steps * config.pnp_spatial_attn_t)
    temp_attn_qk_injection_t = int(config.n_steps * config.pnp_temp_attn_t)
    cross_attn_kv_injection_t = int(config.n_steps * config.pnp_cross_attn_t)
    conv_injection_timesteps = scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
    spatial_attn_qk_injection_timesteps = (
        scheduler.timesteps[:spatial_attn_qk_injection_t] if spatial_attn_qk_injection_t >= 0 else []
    )
    temp_attn_qk_injection_timesteps = (
        scheduler.timesteps[:temp_attn_qk_injection_t] if temp_attn_qk_injection_t >= 0 else []
    )
    cross_attn_kv_injection_timesteps = (
        scheduler.timesteps[:cross_attn_kv_injection_t] if cross_attn_kv_injection_t >= 0 else []
    )

    modify_diffuser_attention_forward(pipe.unet)

    register_temp_attention_pnp(pipe, temp_attn_qk_injection_timesteps, config.inject_background)
    register_spatial_attention_pnp(pipe, spatial_attn_qk_injection_timesteps, config.inject_background)
    register_temp_conv_injection(pipe, conv_injection_timesteps)
    register_out_conv_injection(pipe, conv_injection_timesteps)
    register_resnet_injection(pipe, conv_injection_timesteps)

    logger = logging.getLogger(__name__)
    logger.debug(f"conv_injection_t: {conv_injection_t}")
    logger.debug(f"spatial_attn_qk_injection_t: {spatial_attn_qk_injection_t}")
    logger.debug(f"temp_attn_qk_injection_t: {temp_attn_qk_injection_t}")
    logger.debug(f"conv_injection_timesteps: {conv_injection_timesteps}")
    logger.debug(f"spatial_attn_qk_injection_timesteps: {spatial_attn_qk_injection_timesteps}")
    logger.debug(f"temp_attn_qk_injection_timesteps: {temp_attn_qk_injection_timesteps}")
    logger.debug(f"cross_attn_kv_injection_timesteps: {cross_attn_kv_injection_timesteps}")


def main(template_config, configs_list):
    # Initialize the pipeline
    pipe = I2VGenXLPipeline.from_pretrained(
        PRETRAINED_MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to(device)

    # Initialize the DDIM scheduler
    ddim_scheduler = DDIMScheduler.from_pretrained(
        PRETRAINED_MODEL_PATH,
        subfolder="scheduler",
    )

    for config_entry in configs_list:
        if not config_entry["active"]:
            logger.info(f"Skipping config_entry: {config_entry}")
            continue
        logger.info(f"Processing config_entry: {config_entry}")

        # Override the config with the data_meta_entry
        config = OmegaConf.merge(template_config, OmegaConf.create(config_entry))

        # Update the related paths to absolute paths
        config.video_path = os.path.join(config.video_dir, config.video_name + ".mp4")
        config.video_frames_path = os.path.join(config.video_dir, config.video_name)
        config.edited_first_frame_path = os.path.join(config.data_dir, config.edited_first_frame_path)
        config.obj_mask_path = [os.path.join(config.data_dir, p) for p in config.obj_mask_path]
        config.obj_ddim_latents_path = [os.path.join(config.data_dir, p) for p in config.obj_ddim_latents_path]
        config.bg_ddim_latents_path = os.path.join(config.data_dir, config.bg_ddim_latents_path)
        config.edited_contorl_frame_path_main = os.path.join(config.data_dir, config.edited_contorl_frame_path_main)
        config.edited_contorl_frame_path_background = os.path.join(config.data_dir,
                                                                   config.edited_contorl_frame_path_background)
        config.edited_contorl_frame_path = [os.path.join(config.data_dir, p) for p in config.edited_contorl_frame_path]

        logger.info(f"config: {OmegaConf.to_yaml(config)}")

        # Check if there are fields contain "ReplaceMe"
        for k, v in config.items():
            if "ReplaceMe" in str(v):
                logger.error(f"Field {k} contains 'ReplaceMe'")
                continue

        # This is the same as run_pnp_edit.py
        # Load first frame and source frames
        # 主分支的image
        main_1st_frame = load_image(config.edited_first_frame_path)
        main_1st_frame = main_1st_frame.resize(config.image_size, resample=Image.Resampling.LANCZOS)

        main_video_frames_path = config.edited_contorl_frame_path_main
        main_video_frames = []
        for i in range(config.n_frames):
            main_frame_path = os.path.join(main_video_frames_path, f'{i:0>5d}.png')
            main_frame = load_image(main_frame_path)
            main_frame = main_frame.resize(config.image_size, resample=Image.Resampling.LANCZOS)
            main_video_frames.append(main_frame)
        # 目标分支的image
        edited_frame_list = []
        edited_frame_1st_list = []
        for obj_video_frames_path in config.edited_contorl_frame_path:
            obj_video_frames_list = []
            for i in range(config.n_frames):
                edited_frame_path = os.path.join(obj_video_frames_path, f'{i:0>5d}.png')
                edited_frame = load_image(edited_frame_path)
                edited_frame = edited_frame.resize(config.image_size, resample=Image.Resampling.LANCZOS)
                obj_video_frames_list.append(edited_frame)
            edited_frame_list.append(obj_video_frames_list)
            edited_frame_1st_list.append(obj_video_frames_list[0])

        # 背景分支的image
        background_video_frames_path = config.edited_contorl_frame_path_background
        background_video_frames = []
        for i in range(config.n_frames):
            background_frame_path = os.path.join(background_video_frames_path, f'{i:0>5d}.png')
            background_frame = load_image(background_frame_path)
            background_frame = background_frame.resize(config.image_size, resample=Image.Resampling.LANCZOS)
            background_video_frames.append(background_frame)
        background_frame_1st = background_video_frames[0]

        # Load the initial latents at t
        ddim_init_latents_t_idx = config.ddim_init_latents_t_idx
        ddim_scheduler.set_timesteps(config.n_steps)

        # Init Pnp
        init_pnp(pipe, ddim_scheduler, config)

        # Edit video
        pipe.register_modules(scheduler=ddim_scheduler)

        from pipelines.pipeline_i2vgen_xl import I2VGenXLUnetExtension
        pipe.unet.forward = partial(I2VGenXLUnetExtension.forward, pipe.unet)

        edited_video = pipe.sample_with_pnp_pipeline_with_edit_prompt_extraction_with_attn_injection(
            prompt=config.editing_prompt,
            main_first_image=main_1st_frame,
            main_image_list=main_video_frames,
            background_first_image=background_frame_1st,
            background_image_list=background_video_frames,
            objs_first_image=edited_frame_1st_list,
            objs_image_list=edited_frame_list,
            height=config.image_size[1],
            width=config.image_size[0],
            num_frames=config.n_frames,
            num_inference_steps=config.n_steps,
            guidance_scale=config.cfg,
            negative_prompt=config.editing_negative_prompt,
            target_fps=config.target_fps,
            generator=torch.manual_seed(config.seed),
            return_dict=True,
            ddim_init_latents_t_idx=ddim_init_latents_t_idx,
            ddim_inv_prompt=config.ddim_inv_prompt,
            obj_mask=config.obj_mask_path,
            obj_width_height=config.obj_width_height,
            random_noise_ratio=config.random_noise_ratio,
            bg_inv_latents_path=config.bg_ddim_latents_path,
            obj_ddim_latents_path=config.obj_ddim_latents_path,
            obj_ddim_latents_idx_offset=config.obj_ddim_latents_idx_offset,
            obj_random_noise_fusion=config.obj_random_noise_fusion,
            fusion_steps=config.fusion_step
        ).frames[0]
        # Save video
        # Add the config to the output_dir, TODO: make this more elegant
        config_suffix = (
                "ddim_init_latents_t_idx_"
                + str(ddim_init_latents_t_idx)
                + "_nsteps_"
                + str(config.n_steps)
                + "_cfg_"
                + str(config.cfg)
                + "_pnpf"
                + str(config.pnp_f_t)
                + "_pnps"
                + str(config.pnp_spatial_attn_t)
                + "_pnpt"
                + str(config.pnp_temp_attn_t)
                + "_ratio"
                + str(config.random_noise_ratio)
                + "noise_fusion_step"
                + str(f"{config.fusion_step[0]}-{config.fusion_step[1]}")
        )
        output_dir = os.path.join(config.output_dir, config_suffix)
        os.makedirs(output_dir, exist_ok=True)
        edited_video = [frame.resize(config.image_size, resample=Image.LANCZOS) for frame in edited_video]

        edited_video_file_name = "video"
        export_to_video(edited_video, os.path.join(output_dir, f"{edited_video_file_name}.mp4"), fps=config.target_fps)
        export_to_gif(edited_video, os.path.join(output_dir, f"{edited_video_file_name}.gif"))
        logger.info(f"Saved video to: {os.path.join(output_dir, f'{edited_video_file_name}.mp4')}")
        logger.info(f"Saved gif to: {os.path.join(output_dir, f'{edited_video_file_name}.gif')}")
        for i, frame in enumerate(edited_video):
            frame.save(os.path.join(output_dir, f"{edited_video_file_name}_{i:05d}.png"))
            logger.info(f"Saved frames to: {os.path.join(output_dir, f'{edited_video_file_name}_{i:05d}.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_config", type=str,
                        default="configs/group_composite/template.yaml")
    parser.add_argument(
        "--configs_json", type=str, default="../demo/boat_surf/group_config.json"
    )  # This is going to override the template_config

    args = parser.parse_args()
    template_config = OmegaConf.load(args.template_config)

    # Set up logging
    logging_level = logging.DEBUG if template_config.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"template_config: {OmegaConf.to_yaml(template_config)}")

    # Load data jsonl into list
    configs_json = args.configs_json
    assert Path(configs_json).exists()
    with open(configs_json, "r") as file:
        configs_list = json.load(file)
    logger.info(f"Loaded {len(configs_list)} configs from {configs_json}")

    # Set up device and seed
    device = torch.device(template_config.device)
    torch.set_grad_enabled(False)
    seed_everything(template_config.seed)
    main(template_config, configs_list)
