# Copyright 2023 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision.transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models.unets.unet_i2vgen_xl import I2VGenXLUNet, UNet3DConditionOutput
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline

# [Modified]
# Project import
from pnp_utils import register_time, register_time_all
from utils import load_ddim_latents_at_t, mask_preprocess, cvt_cv_aff2torch_aff, warp_affine_torch
from PIL import Image
from copy import deepcopy

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import I2VGenXLPipeline

        >>> pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
        >>> pipeline.enable_model_cpu_offload()

        >>> image_url = "https://github.com/ali-vilab/i2vgen-xl/blob/main/data/test_images/img_0009.png?raw=true"
        >>> image = load_image(image_url).convert("RGB")

        >>> prompt = "Papers were floating in the air on a table in the library"
        >>> negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
        >>> generator = torch.manual_seed(8888)

        >>> frames = pipeline(
        ...     prompt=prompt,
        ...     image=image,
        ...     num_inference_steps=50,
        ...     negative_prompt=negative_prompt,
        ...     guidance_scale=9.0,
        ...     generator=generator
        ... ).frames[0]
        >>> video_path = export_to_gif(frames, "i2v.gif")
        ```
"""


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")

    return outputs


class I2VGenXLUnetExtension:
    def __init__(self):
        pass

    @staticmethod
    @torch.no_grad()
    def forward(
            model: I2VGenXLUNet,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            fps: torch.Tensor,
            image_latents_first: torch.Tensor,
            image_latents: torch.Tensor,
            image_embeddings: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            multi_frame_guidance=False,
            return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple[torch.FloatTensor]]:
        r"""
        The [`I2VGenXLUNet`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            fps (`torch.Tensor`): Frames per second for the video being generated. Used as a "micro-condition".
            image_latents (`torch.FloatTensor`): Image encodings from the VAE.
            image_embeddings (`torch.FloatTensor`): Projection embeddings of the conditioning image computed with a vision encoder.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_3d_condition.UNet3DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        batch_size, channels, num_frames, height, width = sample.shape  # 2,4,16,128,64

        if not multi_frame_guidance:
            image_embeddings = image_embeddings[:, 0:1, :].repeat(1, num_frames, 1)
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** model.num_upsamplers  # 8

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass `timesteps` as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])  # 1->2
        t_emb = model.time_proj(timesteps)  # 2->([2, 320])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=model.dtype)
        t_emb = model.time_embedding(t_emb, timestep_cond)  # ([2, 320])->([2, 1280])

        # 2. FPS
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        fps = fps.expand(fps.shape[0])
        fps_emb = model.fps_embedding(model.time_proj(fps).to(dtype=model.dtype))  # 2->([2, 1280])

        # 3. time + FPS embeddings.
        emb = t_emb + fps_emb  # t和fps是相加的
        emb = emb.repeat_interleave(repeats=num_frames, dim=0)  # ([2, 1280])->([32, 1280]) 复制num_frames
        ####################################引导特征开始处理##################################################################################
        # 4. context embeddings.
        # The context embeddings consist of both text embeddings from the input prompt
        # AND the image embeddings from the input image. For images, both VAE encodings
        # and the CLIP image embeddings are incorporated.
        # So the final `context_embeddings` becomes the query for cross-attention.
        context_emb = sample.new_zeros(batch_size, 0,
                                       model.config.cross_attention_dim)  # self.config.cross_attention_dim:1024;torch.Size([2, 0, 1024])
        context_emb = torch.cat([context_emb, encoder_hidden_states],
                                dim=1)  # torch.Size([2, 77, 1024])->([2, 77, 1024]);encoder_hidden_states是提示特征
        # context_emb_list_0 = []
        # context_emb_list_1 = []
        context_emb_list = []
        for i in range(image_latents.size(dim=2)):
            image_latents_for_context_embds = image_latents[:, :, i if multi_frame_guidance else 0, :].unsqueeze(
                dim=2)  # image_latents:([2, 4, 16, 128, 64])，只取了第一帧
            image_latents_context_embs = image_latents_for_context_embds.permute(0, 2, 1, 3, 4).reshape(
                # torch.Size([2, 4, 128, 64])
                image_latents_for_context_embds.shape[0] * image_latents_for_context_embds.shape[2],
                image_latents_for_context_embds.shape[1],
                image_latents_for_context_embds.shape[3],
                image_latents_for_context_embds.shape[4],
            )
            image_latents_context_embs = model.image_latents_context_embedding(
                image_latents_context_embs)  # ([2, 4, 128, 64])->([2, 1024, 8, 8])

            _batch_size, _channels, _height, _width = image_latents_context_embs.shape
            image_latents_context_embs = image_latents_context_embs.permute(0, 2, 3, 1).reshape(
                _batch_size, _height * _width, _channels
            )
            context_emb_ = torch.cat([context_emb, image_latents_context_embs],
                                     dim=1)  # torch.Size([2, 77, 1024]);torch.Size([2, 64, 1024])->torch.Size([2, 141, 1024])

            image_emb = model.context_embedding(
                image_embeddings[:, i, :].unsqueeze(dim=1))  # torch.Size([2, 1, 1024])->torch.Size([2, 1, 4096])
            # image_emb = model.context_embedding(image_embeddings)
            image_emb = image_emb.view(-1, model.config.in_channels,
                                       model.config.cross_attention_dim)  # torch.Size([2, 4, 1024])
            one_emb = torch.cat([context_emb_, image_emb],
                                dim=1)  # ([2, 141, 1024])+([2, 4, 1024])->torch.Size([2, 145, 1024])
            # context_emb_list_0.append(one_emb[:1,:,:])
            # context_emb_list_1.append(one_emb[1:,:,:])
            context_emb_list.append(one_emb.unsqueeze(dim=1))
            # if i == 0:
            #     emb_all = one_emb
            # else:
            #     emb_all = torch.cat([emb_all,one_emb])
            # context_emb = torch.cat(context_emb_list,dim=0)
        # context_emb_list_0 = torch.cat(context_emb_list_0,dim=0)
        # context_emb_list_1 = torch.cat(context_emb_list_1,dim=0)
        # context_emb = torch.cat([context_emb_list_0,context_emb_list_1],dim=0)
        if False:
            image_emb = model.context_embedding(image_embeddings[:, 0, :].unsqueeze(dim=1))
            image_emb = image_emb.view(-1, model.config.in_channels, model.config.cross_attention_dim)
            context_emb = torch.cat([context_emb, image_emb], dim=1)
            context_emb = context_emb.repeat_interleave(repeats=num_frames, dim=0)
        else:
            context_emb_list = torch.cat(context_emb_list, dim=1)
            context_emb = context_emb_list.reshape(
                context_emb_list.shape[0] * context_emb_list.shape[1],
                context_emb_list.shape[2],
                context_emb_list.shape[3],
            )
        ## context_emb = context_emb.repeat_interleave(repeats=num_frames, dim=0)####([2, 145, 1024])->([32, 145, 1024]) 复制num_frames

        ####################################引导特征结束处理##################################################################################
        image_latents = image_latents_first.permute(0, 2, 1, 3, 4).reshape(
            image_latents.shape[0] * image_latents.shape[2],
            image_latents.shape[1],
            image_latents.shape[3],
            image_latents.shape[4],
        )
        image_latents = model.image_latents_proj_in(image_latents)
        image_latents = (
            image_latents[None, :]
                .reshape(batch_size, num_frames, channels, height, width)
                .permute(0, 3, 4, 1, 2)
                .reshape(batch_size * height * width, num_frames, channels)
        )
        image_latents = model.image_latents_temporal_encoder(image_latents)
        image_latents = image_latents.reshape(batch_size, height, width, num_frames, channels).permute(0, 4, 3, 1,
                                                                                                       2)

        # 5. pre-process
        sample = torch.cat([sample, image_latents], dim=1)
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = model.conv_in(sample)
        sample = model.transformer_in(
            sample,
            num_frames=num_frames,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # 6. down
        down_block_res_samples = (sample,)
        for downsample_block in model.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=context_emb,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        # 7. mid
        if model.mid_block is not None:
            sample = model.mid_block(
                sample,
                emb,
                encoder_hidden_states=context_emb,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        # 8. up
        for i, upsample_block in enumerate(model.up_blocks):
            is_final_block = i == len(model.up_blocks) - 1

            res_samples = down_block_res_samples[-len(
                upsample_block.resnets):]  # 0([32, 320, 128, 64]);1([32, 320, 128, 64]);2([32, 320, 128, 64]);3([32, 320, 64, 32]);4([32, 640, 64, 32]);5([32, 640, 64, 32])
            down_block_res_samples = down_block_res_samples[: -len(
                upsample_block.resnets)]  # 0([32, 320, 128, 64])1([32, 320, 128, 64]);2([32, 320, 128, 64])

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=context_emb,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )

        # 9. post-process
        sample = model.conv_norm_out(sample)
        sample = model.conv_act(sample)

        sample = model.conv_out(sample)

        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)


@dataclass
class I2VGenXLPipelineOutput(BaseOutput):
    r"""
    Output class for image-to-video pipeline.

    Args:
        frames (`List[np.ndarray]` or `torch.FloatTensor`)
            List of denoised frames (essentially images) as NumPy arrays of shape `(height, width, num_channels)` or as
            a `torch` tensor. The length of the list denotes the video length (the number of frames).
    """

    frames: Union[List[np.ndarray], torch.FloatTensor]


# Modified from DiffEditInversionPipelineOutput
@dataclass
class StableVideoDiffusionInversionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.FloatTensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `num_timesteps * batch_size` or numpy array of shape `(num_timesteps,
            batch_size, height, width, num_channels)`. PIL images or numpy array present the denoised images of the
            diffusion pipeline.
    """

    inverted_latents: torch.FloatTensor
    # images: Union[List[PIL.Image.Image], np.ndarray] # TODO: we can return the noisy video.


class I2VGenXLPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-to-video generation as proposed in [I2VGenXL](https://i2vgen-xl.github.io/).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`I2VGenXLUNet`]):
            A [`I2VGenXLUNet`] to denoise the encoded video latents.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            image_encoder: CLIPVisionModelWithProjection,
            feature_extractor: CLIPImageProcessor,
            unet: I2VGenXLUNet,
            scheduler: DDIMScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # `do_resize=False` as we do custom resizing.
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_resize=False)

    def prepare_img_embedding_and_img_latents(self,
                                              images,
                                              width, height,
                                              num_videos_per_prompt, num_frames,
                                              device):
        image_embeddings_all_list = []
        image_latents_all_list = []

        for i in range(len(images)):
            image = images[i]
            cropped_image = _center_crop_wide(image, (width, width))  # 中间crop？？？？？？？
            cropped_image = _resize_bilinear(
                image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
            )
            image_embeddings = self._encode_image(cropped_image, device,
                                                  num_videos_per_prompt)  # torch.Size([1, 1, 1024])

            # 3.2.2 Image latents.
            resized_image = _center_crop_wide(image, (width, height))
            image = self.image_processor.preprocess(resized_image).to(device=device, dtype=image_embeddings.dtype)
            image_latents = self.prepare_image_latents(
                image,
                device=device,
                num_frames=num_frames,
                num_videos_per_prompt=num_videos_per_prompt,
            )

            # [Modified]
            # 3.2.1 Edited first frame encodings.
            # ddim_inv_1st_frame = ddim_inv_frame_list[i]
            # cropped_image = _center_crop_wide(ddim_inv_1st_frame, (width, width))
            # cropped_image = _resize_bilinear(
            #     ddim_inv_1st_frame,
            #     (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
            # )
            # _embeddings = self._encode_image(cropped_image, device, num_videos_per_prompt)
            # if self.do_classifier_free_guidance:
            #     ddim_inv_1st_frame_embeddings = _embeddings.chunk(2)[1]  # Chunk 0 is negative prompt
            # else:
            #     ddim_inv_1st_frame_embeddings = _embeddings
            #
            # # 3.2.2 Edited first frame latents.
            # resized_image = _center_crop_wide(ddim_inv_1st_frame, (width, height))
            # ddim_inv_1st_frame = self.image_processor.preprocess(resized_image).to(device=device,
            #                                                                        dtype=image_embeddings.dtype)
            # _latents = self.prepare_image_latents(
            #     ddim_inv_1st_frame,
            #     device=device,
            #     num_frames=num_frames,
            #     num_videos_per_prompt=num_videos_per_prompt,
            # )
            # if self.do_classifier_free_guidance:
            #     ddim_inv_1st_frame_latents = _latents.chunk(2)[1]
            # else:
            #     ddim_inv_1st_frame_latents = _latents

            # image_embeddings_all = torch.cat([ddim_inv_1st_frame_embeddings, image_embeddings])
            # image_latents_all = torch.cat([ddim_inv_1st_frame_latents, image_latents])

            image_embeddings_all_list.append(image_embeddings[:, :, 0].unsqueeze(dim=2))
            image_latents_all_list.append(image_latents)
        return torch.cat(image_embeddings_all_list, dim=1), torch.cat(image_latents_all_list, dim=2)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def encode_prompt(
            self,
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
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
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
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
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if self.do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
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

            # Apply clip_skip to negative prompt embeds
            if clip_skip is None:
                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]
            else:
                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                negative_prompt_embeds = negative_prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                negative_prompt_embeds = self.text_encoder.text_model.final_layer_norm(negative_prompt_embeds)

        if self.do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def _encode_image(self, image, device, num_videos_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # Normalize the image with CLIP training stats.
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if self.do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)  # TODO: Why is this zero?
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def decode_latents(self, latents, decode_chunk_size=None):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        if decode_chunk_size is not None:
            frames = []
            for i in range(0, latents.shape[0], decode_chunk_size):
                frame = self.vae.decode(latents[i: i + decode_chunk_size]).sample
                frames.append(frame)
            image = torch.cat(frames, dim=0)
        else:
            image = self.vae.decode(latents).sample

        decode_shape = (batch_size, num_frames, -1) + image.shape[2:]
        video = image[None, :].reshape(decode_shape).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
            self,
            prompt,
            image,
            height,
            width,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

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

        if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

    def prepare_image_latents(
            self,
            image,
            device,
            num_frames,
            num_videos_per_prompt,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor

        # Add frames dimension to image latents
        image_latents = image_latents.unsqueeze(2)

        # Append a position mask for each subsequent frame
        # after the intial image latent frame
        frame_position_mask = []
        for frame_idx in range(num_frames - 1):
            scale = (frame_idx + 1) / (num_frames - 1)
            frame_position_mask.append(torch.ones_like(image_latents[:, :, :1]) * scale)
        if frame_position_mask:
            frame_position_mask = torch.cat(frame_position_mask, dim=2)
            image_latents = torch.cat([image_latents, frame_position_mask], dim=2)

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)

        if self.do_classifier_free_guidance:
            image_latents = torch.cat([image_latents] * 2)

        return image_latents

    # Modified from SVD/_encode_vae_image
    def encode_vae_video(
            self,
            video: List[PIL.Image.Image],
            device,
            height: int = 576,
            width: int = 1024,
    ):
        # video is a list of PIL images
        # [batch*frames] while batch is always 1  TODO: generalize to batch > 1
        dtype = next(self.vae.parameters()).dtype
        n_frames = len(video)
        video_latents = []
        for i in range(0, n_frames):
            frame = video[i]
            resized_frame = _center_crop_wide(frame, (width, height))
            frame = self.image_processor.preprocess(resized_frame)
            frame = frame.to(device=device, dtype=dtype)
            image_latents = self.vae.encode(frame).latent_dist.sample()  # [1, channels, height, width]
            image_latents = image_latents * self.vae.config.scaling_factor
            logger.debug(f"image_latents.shape: {image_latents.shape}")
            image_latents = image_latents.squeeze(0)  # [channels, height, width]
            video_latents.append(image_latents)
        video_latents = torch.stack(video_latents)  # [batch*frames, channels, height, width]
        video_latents = video_latents.reshape(1, n_frames, *video_latents.shape[1:])
        video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [batch, channels, frames, height, width]
        video_latents = video_latents.to(device=device, dtype=dtype)
        # [batch, channels, frames, height, width]
        return video_latents

    # Copied from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth.TextToVideoSDPipeline.prepare_latents
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

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        logger.debug(f"latents.shape: {latents.shape}")
        logger.debug(f"init_noise_sigma: {self.scheduler.init_noise_sigma}")
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            image: PipelineImageInput = None,
            height: Optional[int] = 704,
            width: Optional[int] = 1280,
            target_fps: Optional[int] = 16,
            num_frames: int = 16,
            num_inference_steps: int = 50,
            guidance_scale: float = 9.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: float = 0.0,
            num_videos_per_prompt: Optional[int] = 1,
            decode_chunk_size: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = 1,
            ddim_init_latents_t_idx: Optional[int] = 1,  # Modified
    ):
        r"""
        The call function to the pipeline for image-to-video generation with [`I2VGenXLPipeline`].

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            target_fps (`int`, *optional*):
                Frames per second. The rate at which the generated images shall be exported to a video after generation. This is also used as a "micro-condition" while generation.
            num_frames (`int`, *optional*):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            eta (`float`, *optional*):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            num_videos_per_prompt (`int`, *optional*):
                The number of images to generate per prompt.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        logger.info(f"height: {height}, width: {width}")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, image, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds)
        logger.info(f"Prompt: {prompt}")

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
        self._guidance_scale = guidance_scale

        # 3.1 Encode input text prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3.2 Encode image prompt
        # 3.2.1 Image encodings.
        # https://github.com/ali-vilab/i2vgen-xl/blob/2539c9262ff8a2a22fa9daecbfd13f0a2dbc32d0/tools/inferences/inference_i2vgen_entrance.py#L114
        cropped_image = _center_crop_wide(image, (width, width))
        cropped_image = _resize_bilinear(
            cropped_image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
        )
        image_embeddings = self._encode_image(cropped_image, device, num_videos_per_prompt)

        # 3.2.2 Image latents.
        resized_image = _center_crop_wide(image, (width, height))
        image = self.image_processor.preprocess(resized_image).to(device=device, dtype=image_embeddings.dtype)
        image_latents = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        # 3.3 Prepare additional conditions for the UNet.
        if self.do_classifier_free_guidance:
            fps_tensor = torch.tensor([target_fps, target_fps]).to(device)
        else:
            fps_tensor = torch.tensor([target_fps]).to(device)
        fps_tensor = fps_tensor.repeat(batch_size * num_videos_per_prompt, 1).ravel()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.scheduler.timesteps = self.scheduler.timesteps[ddim_init_latents_t_idx:]
        timesteps = self.scheduler.timesteps
        logger.info(f"self.scheduler: {self.scheduler}")
        logger.info(f"timesteps: {timesteps}")
        logger.info(f"Sampling starts from latents_at_t={self.scheduler.timesteps[0]}")

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

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps_tensor,
                    image_latents=image_latents,
                    image_embeddings=image_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    logger.debug(f"doing classifier free guidance with guidance_scale: {guidance_scale}")
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                batch_size, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(batch_size, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            return I2VGenXLPipelineOutput(frames=latents)

        video_tensor = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)
        video = tensor2vid(video_tensor, self.image_processor, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return I2VGenXLPipelineOutput(frames=video)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def sample_with_pnp_pipeline_with_edit_prompt_extraction_with_attn_injection(
            self,
            prompt: Union[str, List[str]] = None,
            main_first_image: PipelineImageInput = None,
            main_image_list: PipelineImageInput = None,
            background_first_image: PipelineImageInput = None,
            background_image_list: PipelineImageInput = None,
            objs_first_image: PipelineImageInput = None,
            objs_image_list: PipelineImageInput = None,
            height: Optional[int] = 704,
            width: Optional[int] = 1280,
            target_fps: Optional[int] = 16,
            num_frames: int = 16,
            num_inference_steps: int = 50,
            guidance_scale: float = 9.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: float = 0.0,
            num_videos_per_prompt: Optional[int] = 1,
            decode_chunk_size: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = 1,
            fusion_steps=(0, 3),
            ddim_init_latents_t_idx: Optional[int] = 1,  # [Modified]
            ddim_inv_prompt: Union[str, List[str]] = None,  # [Modified] this is the same prompt for ddim reconstruction
            obj_mask=None,
            obj_width_height=None,
            obj_ddim_latents_idx_offset=None,
            obj_random_noise_fusion=False,
            random_noise_ratio=0.0,
            bg_inv_latents_path=None,
            obj_ddim_latents_path=None
    ):
        r"""
                The call function to the pipeline for image-to-video generation with [`I2VGenXLPipeline`].

                Args:
                    prompt (`str` or `List[str]`, *optional*):
                        The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                    image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                        Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                        [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
                    height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                        The height in pixels of the generated image.
                    width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                        The width in pixels of the generated image.
                    target_fps (`int`, *optional*):
                        Frames per second. The rate at which the generated images shall be exported to a video after generation. This is also used as a "micro-condition" while generation.
                    num_frames (`int`, *optional*):
                        The number of video frames to generate.
                    num_inference_steps (`int`, *optional*):
                        The number of denoising steps.
                    guidance_scale (`float`, *optional*, defaults to 7.5):
                        A higher guidance scale value encourages the model to generate images closely linked to the text
                        `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                    negative_prompt (`str` or `List[str]`, *optional*):
                        The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                        pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                    eta (`float`, *optional*):
                        Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                        to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
                    num_videos_per_prompt (`int`, *optional*):
                        The number of images to generate per prompt.
                    decode_chunk_size (`int`, *optional*):
                        The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                        between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                        for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
                    generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                        A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                        generation deterministic.
                    latents (`torch.FloatTensor`, *optional*):
                        Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                        generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                        tensor is generated by sampling using the supplied random `generator`.
                    prompt_embeds (`torch.FloatTensor`, *optional*):
                        Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                        provided, text embeddings are generated from the `prompt` input argument.
                    negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                        Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                        not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                    output_type (`str`, *optional*, defaults to `"pil"`):
                        The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                    return_dict (`bool`, *optional*, defaults to `True`):
                        Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                        plain tuple.
                    cross_attention_kwargs (`dict`, *optional*):
                        A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                        [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                    clip_skip (`int`, *optional*):
                        Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                        the output of the pre-final layer will be used for computing the prompt embeddings.

                Examples:

                Returns:
                    [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] or `tuple`:
                        If `return_dict` is `True`, [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] is
                        returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
                """
        downscale = 8

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor  # 1024
        width = width or self.unet.config.sample_size * self.vae_scale_factor  # 512
        logger.info(f"height: {height}, width: {width}")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, main_first_image, height, width, negative_prompt, prompt_embeds,
                          negative_prompt_embeds)
        logger.info(f"Prompt: {prompt}")
        assert len(obj_mask) == len(obj_ddim_latents_path)
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            # [Modified]
            assert len(ddim_inv_prompt) == len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale

        # 3.1 Encode input text prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )

        # [Modified]
        # 3.1 Encode ddim inversion prompt
        obj_num = len(obj_ddim_latents_path) + 1  # NOTE: 各路素材反演时的prompt，默认为null prompt
        ddim_inv_prompt_embeds, _ = self.encode_prompt(
            ddim_inv_prompt,
            device,
            num_videos_per_prompt,
            negative_prompt=negative_prompt,  # None, # NOTE: 和inversion中统一
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        ddim_inv_prompt_embeds = ddim_inv_prompt_embeds.repeat(obj_num, 1, 1)
        # [Modified]
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # [ddim_inversion_prompt, editing_negative_prompt, editing_prompt]
        if self.do_classifier_free_guidance:
            prompt_embeds_all = torch.cat([ddim_inv_prompt_embeds, negative_prompt_embeds, prompt_embeds])
        else:
            prompt_embeds_all = torch.cat([ddim_inv_prompt_embeds, prompt_embeds])  # ([1, 77, 1024])->([2, 77, 1024])

        # 准备主分支的first_image_latents
        image = main_first_image
        resized_image = _center_crop_wide(image, (width, height))
        image = self.image_processor.preprocess(resized_image).to(device=device, dtype=prompt_embeds.dtype)
        first_image_latents = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt,
        )
        # 准备主分支的image_latent
        resized_image = _center_crop_wide(main_first_image, (width, height))
        image = self.image_processor.preprocess(resized_image).to(device=device, dtype=prompt_embeds.dtype)
        image = image.to(device=device)
        main_image_latents_list = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        if self.do_classifier_free_guidance:
            # main_image_latents_list = torch.cat([main_image_latents_list] * 2)
            pass

        # 准备主分支的image_embeddings
        main_image_embedding_list = []
        for i in range(len(main_image_list)):
            image = main_image_list[i]
            cropped_image = _center_crop_wide(image, (width, width))  # 中间crop？？？？？？？
            cropped_image = _resize_bilinear(
                image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
            )
            image_embeddings = self._encode_image(cropped_image, device,
                                                  num_videos_per_prompt)  # torch.Size([1, 1, 1024])
            main_image_embedding_list.append(image_embeddings)
        main_image_embedding_list = torch.cat(main_image_embedding_list, dim=1)

        # 准备背景分支的first_image_latents
        image = background_first_image
        resized_image = _center_crop_wide(image, (width, height))
        image = self.image_processor.preprocess(resized_image).to(device=device, dtype=prompt_embeds.dtype)
        background_first_image_latents = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt,
        )
        if self.do_classifier_free_guidance:
            background_first_image_latents = background_first_image_latents.chunk(2)[1]

        # 准备背景分支的image_latent
        resized_image = _center_crop_wide(background_image_list[0], (width, height))
        image = self.image_processor.preprocess(resized_image).to(device=device, dtype=prompt_embeds.dtype)
        image = image.to(device=device)
        background_image_latents_list = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt
        )

        if self.do_classifier_free_guidance:
            background_image_latents_list = background_image_latents_list.chunk(2)[1]

        # 目标分支的first_image_latents准备(多个目标)
        ddim_inv_1st_frame_latents = []
        for ddim_inv_1st_frame in objs_first_image:
            resized_image = _center_crop_wide(ddim_inv_1st_frame, (width, height))
            ddim_inv_1st_frame = self.image_processor.preprocess(resized_image).to(device=device,
                                                                                   dtype=prompt_embeds.dtype)
            _latents = self.prepare_image_latents(
                ddim_inv_1st_frame,
                device=device,
                num_frames=num_frames,
                num_videos_per_prompt=num_videos_per_prompt,
            )
            if self.do_classifier_free_guidance:
                ddim_inv_1st_frame_latent = _latents.chunk(2)[1]
            else:
                ddim_inv_1st_frame_latent = _latents

            ddim_inv_1st_frame_latents.append(ddim_inv_1st_frame_latent)

        ddim_inv_1st_frame_latents = torch.cat(ddim_inv_1st_frame_latents)
        first_image_latents_all_list = torch.cat(
            [background_first_image_latents, ddim_inv_1st_frame_latents, first_image_latents])

        # 目标分支的image_latents
        objs_image_latents_list = []
        for ddim_inv_frame in objs_image_list:
            resized_image = _center_crop_wide(ddim_inv_frame[0], (width, height))
            image = self.image_processor.preprocess(resized_image).to(device=device, dtype=prompt_embeds.dtype)
            image = image.to(device=device)
            obj_image_latents_list = self.prepare_image_latents(
                image,
                device=device,
                num_frames=num_frames,
                num_videos_per_prompt=num_videos_per_prompt,
            )

            if self.do_classifier_free_guidance:
                obj_image_latents_list = obj_image_latents_list.chunk(2)[1]

            objs_image_latents_list.append(obj_image_latents_list)
        objs_image_latents_list = torch.cat(objs_image_latents_list)

        all_image_latents_list = torch.cat(
            [background_image_latents_list, objs_image_latents_list, main_image_latents_list])

        # 准备背景分支的image_embeddings
        background_image_embedding_list = []
        for i in range(len(background_image_list)):
            image = background_image_list[i]
            cropped_image = _center_crop_wide(image, (width, width))  # 中间crop？？？？？？？
            cropped_image = _resize_bilinear(
                image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
            )
            image_embeddings = self._encode_image(cropped_image, device,
                                                  num_videos_per_prompt)  # torch.Size([1, 1, 1024])
            background_image_embedding_list.append(image_embeddings)
        background_image_embedding_list = torch.cat(background_image_embedding_list, dim=1)
        if self.do_classifier_free_guidance:
            background_image_embedding_list = background_image_embedding_list.chunk(2)[1]

        # 准备目标分支的image_embeddings
        obj_image_embeddings_all_list = []
        # NOTE: 提取每一帧作为prompt
        for ddim_inv_frame in objs_image_list:
            obj_inv_image_embeddings_list = []
            for i in range(len(ddim_inv_frame)):
                image = ddim_inv_frame[i]
                cropped_image = _center_crop_wide(image, (width, width))  # 中间crop？？？？？？？
                cropped_image = _resize_bilinear(
                    image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
                )
                obj_inv_image_embeddings = self._encode_image(cropped_image, device,
                                                              num_videos_per_prompt)  # torch.Size([1, 1, 1024])

                if self.do_classifier_free_guidance:
                    obj_inv_image_embeddings = obj_inv_image_embeddings.chunk(2)[1]  # Chunk 0 is negative prompt
                else:
                    obj_inv_image_embeddings = obj_inv_image_embeddings

                obj_inv_image_embeddings_list.append(obj_inv_image_embeddings)
            obj_inv_image_embeddings_list = torch.cat(obj_inv_image_embeddings_list, dim=1)
            obj_image_embeddings_all_list.append(obj_inv_image_embeddings_list)
        obj_image_embeddings_all_list = torch.cat(obj_image_embeddings_all_list)

        image_embeddings_all_list = torch.cat(
            [background_image_embedding_list, obj_image_embeddings_all_list, main_image_embedding_list])

        # 3.3 Prepare additional conditions for the UNet.
        if self.do_classifier_free_guidance:
            fps_tensor = torch.tensor([target_fps] * (obj_num + 2)).to(device)
            # fps_tensor = torch.tensor([target_fps,target_fps,target_fps, target_fps]).to(device)
        else:
            fps_tensor = torch.tensor([target_fps, target_fps]).to(device)
        fps_tensor = fps_tensor.repeat(batch_size * num_videos_per_prompt, 1).ravel()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler = deepcopy(self.scheduler)
        self.scheduler.timesteps = self.scheduler.timesteps[ddim_init_latents_t_idx:]
        timesteps = self.scheduler.timesteps
        logger.info(f"self.scheduler: {self.scheduler}")
        logger.info(f"timesteps: {timesteps}")
        logger.info(f"Sampling starts from latents_at_t={self.scheduler.timesteps[0]}")

        obj_fusion_timesteps = []
        for i in range(len(obj_ddim_latents_path)):
            fusion_timesteps = []
            for j in range(*fusion_steps):
                fusion_timesteps.append(scheduler.timesteps[obj_ddim_latents_idx_offset[i]:][j])
            logger.info(f"Fusion timesteps for object-{i}={fusion_timesteps}")
            obj_fusion_timesteps.append(fusion_timesteps)

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
        # ddim_inv_latents = load_ddim_latents_at_t(timesteps[0], ddim_inv_latents_path).to(self.device)
        # latents = ddim_inv_latents
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # NOTE: 遮罩处理
        # assert obj_mask is not None and edit_mask is not None
        obj_mask_tensor_list = []
        obj_mask_binary_tensor_list = []

        batch_size, channel, frames, latent_height, latent_width = latents.shape
        for om, (obj_w, obj_h) in zip(obj_mask, obj_width_height):
            batch_size, channel, frames, _, _ = latents.shape
            obj_mask_tensor, obj_mask_binary_tensor = mask_preprocess(om, device, latents.dtype, batch_size, channel,
                                                                      frames,
                                                                      downscale=downscale)

            obj_mask_tensor_list.append(obj_mask_tensor)
            obj_mask_binary_tensor_list.append(obj_mask_binary_tensor)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        process_t = 3
        process_ratios = torch.linspace(start=random_noise_ratio, end=1.0, steps=process_t)

        mask_cum_union_list = []  # binary
        mask_num = len(obj_mask_binary_tensor_list)
        for i in range(mask_num):
            start = i
            mask_union = obj_mask_binary_tensor_list[i]
            for j in range(start + 1, mask_num):
                mask_union = torch.bitwise_or(mask_union, obj_mask_binary_tensor_list[j])

            mask_cum_union_list.append(mask_union)

        obj_mask_cover_tensor_list = []  # float [0,1]
        obj_mask_cover_tensor_binary_list = []  # bool
        for i in range(mask_num):
            mask_a = mask_cum_union_list[i]
            j = i + 1
            if j == mask_num:
                mask_b = torch.zeros_like(mask_a)
            else:
                mask_b = mask_cum_union_list[j]
            # mask_a -= mask_b
            mask_a = torch.bitwise_xor(mask_a, mask_b)

            obj_mask_cover_tensor_list.append(obj_mask_tensor_list[i] * mask_a)
            obj_mask_cover_tensor_binary_list.append(torch.bitwise_and(obj_mask_binary_tensor_list[i], mask_a))

        obj_mask_tensor_list_final = obj_mask_tensor_list
        obj_mask_binary_tensor_list_final = obj_mask_binary_tensor_list

        fusion_counter = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                bg_ddim_inv_latents_at_t = load_ddim_latents_at_t(t, bg_inv_latents_path).to(
                    self.device)  # NOTE: background
                if fusion_steps[0] <= i < fusion_steps[1]:  # < len(timesteps) // 5
                    print(f'noise fusion at t: {t}')
                    mix_ratio = random_noise_ratio  # process_ratios[i]

                    # NOTE: 1. mix with background to avoid illusion
                    latents = mix_ratio * latents + (1.0 - mix_ratio) * bg_ddim_inv_latents_at_t
                    objs_inv_latents_at_t_list = []
                    for j, (obj_mask_tensor, obj_fusion_timestep) in enumerate(
                            zip(obj_mask_tensor_list_final, obj_fusion_timesteps)):
                        obj_inv_latents_at_t = load_ddim_latents_at_t(
                            obj_fusion_timestep[fusion_counter],
                            obj_ddim_latents_path[j]).to(self.device)
                        objs_inv_latents_at_t_list.append(obj_inv_latents_at_t)

                        ddim_inv_obj_latents_at_t = obj_inv_latents_at_t * obj_mask_tensor
                        background_latents = latents * (1.0 - obj_mask_tensor)
                        # NOTE: 前景部分噪声混合
                        if obj_random_noise_fusion:
                            foreground_latents = latents * obj_mask_tensor
                            foreground_fusion = foreground_latents * mix_ratio + (
                                    1 - mix_ratio) * ddim_inv_obj_latents_at_t
                        else:
                            # NOTE: 无噪声混合
                            foreground_fusion = ddim_inv_obj_latents_at_t
                        latents = background_latents + foreground_fusion

                    objs_inv_latents_at_t = torch.cat(objs_inv_latents_at_t_list)
                else:
                    objs_inv_latents_at_t_list = []
                    for j, obj_mask_tensor in enumerate(obj_mask_tensor_list):
                        # obj_mask_tensor = obj_mask_tensor[1].to(torch.float16)
                        obj_inv_latents_at_t = load_ddim_latents_at_t(t, obj_ddim_latents_path[j]).to(self.device)

                        objs_inv_latents_at_t_list.append(obj_inv_latents_at_t)
                    objs_inv_latents_at_t = torch.cat(objs_inv_latents_at_t_list)

                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([bg_ddim_inv_latents_at_t, objs_inv_latents_at_t, latents,
                                                    latents])
                else:
                    latent_model_input = torch.cat([objs_inv_latents_at_t, latents])
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # [Modified]
                # Pnp
                register_time_all(self, t.item(),
                                  list(zip(obj_mask_tensor_list_final, obj_mask_binary_tensor_list_final)))

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_all,
                    fps=fps_tensor,
                    image_latents_first=first_image_latents_all_list,
                    image_latents=all_image_latents_list,
                    image_embeddings=image_embeddings_all_list,  # image_embeddings_all_list,
                    cross_attention_kwargs=cross_attention_kwargs,
                    multi_frame_guidance=False,
                    return_dict=False,
                )[0]
                # if mask is not None:
                #     noise_pred_obj = self.unet(
                #         latent_model_input_obj,
                #         t,
                #         encoder_hidden_states=prompt_embeds_all,
                #         fps=fps_tensor,
                #         image_latents=image_latents_all,
                #         image_embeddings=image_embeddings_all,
                #         cross_attention_kwargs=cross_attention_kwargs,
                #         return_dict=False,
                #     )[0]
                # [Modified]
                # Order: ddim_inversion_prompt, editing_negative_prompt, editing_prompt
                if self.do_classifier_free_guidance:
                    noise_pred_negative, noise_pred_editing = noise_pred.chunk(obj_num + 2)[-2], \
                                                              noise_pred.chunk(obj_num + 2)[-1]
                    logger.debug(f"doing classifier free guidance with guidance_scale: {guidance_scale}")
                    noise_pred = noise_pred_negative + guidance_scale * (noise_pred_editing - noise_pred_negative)
                else:
                    _noise_pred_ddim_inv, noise_pred_editing = noise_pred.chunk(2)
                    noise_pred = noise_pred_editing

                # reshape latents
                batch_size, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(batch_size, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            return I2VGenXLPipelineOutput(frames=latents)

        video_tensor = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)
        video = tensor2vid(video_tensor, self.image_processor, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return I2VGenXLPipelineOutput(frames=video)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def invert(
            self,
            prompt: Union[str, List[str]] = None,
            image: PipelineImageInput = None,
            height: Optional[int] = 704,
            width: Optional[int] = 1280,
            target_fps: Optional[int] = 16,
            num_frames: int = 16,
            num_inference_steps: int = 50,
            guidance_scale: float = 9.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: float = 0.0,
            num_videos_per_prompt: Optional[int] = 1,
            decode_chunk_size: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = 1,
            output_dir: Optional[str] = None,
    ):
        r"""
        The call function to the pipeline for image-to-video generation with [`I2VGenXLPipeline`].

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            target_fps (`int`, *optional*):
                Frames per second. The rate at which the generated images shall be exported to a video after generation. This is also used as a "micro-condition" while generation.
            num_frames (`int`, *optional*):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            eta (`float`, *optional*):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            num_videos_per_prompt (`int`, *optional*):
                The number of images to generate per prompt.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        logger.info(f"height: {height}, width: {width}")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, image, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        logger.info(f"prompt: {prompt}")

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale
        logger.debug(f"self._guidance_scale: {self._guidance_scale}")

        # 3.1 Encode input text prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3.2 Encode image prompt
        # 3.2.1 Image encodings.
        # https://github.com/ali-vilab/i2vgen-xl/blob/2539c9262ff8a2a22fa9daecbfd13f0a2dbc32d0/tools/inferences/inference_i2vgen_entrance.py#L114
        cropped_image = _center_crop_wide(image, (width, width))
        cropped_image = _resize_bilinear(
            cropped_image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
        )
        image_embeddings = self._encode_image(cropped_image, device, num_videos_per_prompt)

        # 3.2.2 Image latents.
        resized_image = _center_crop_wide(image, (width, height))
        image = self.image_processor.preprocess(resized_image).to(device=device, dtype=image_embeddings.dtype)
        image_latents = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        # 3.3 Prepare additional conditions for the UNet.
        if self.do_classifier_free_guidance:
            fps_tensor = torch.tensor([target_fps, target_fps]).to(device)
        else:
            fps_tensor = torch.tensor([target_fps]).to(device)
        fps_tensor = fps_tensor.repeat(batch_size * num_videos_per_prompt, 1).ravel()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        logger.debug(f"self.scheduler: {self.scheduler}")
        logger.debug(f"timesteps: {timesteps}")

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
        import time
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        inverted_latents = []  # Modified
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimention
                logger.debug(f"image_latents.shape: {image_latents.shape}")
                logger.debug(f"latent_model_input.shape: {latent_model_input.shape}")

                # predict the noise residual
                # s = time.time()
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps_tensor,
                    image_latents=image_latents,
                    image_embeddings=image_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                # e = time.time()
                # print('unet_time:',e-s)

                # perform guidance

                if self.do_classifier_free_guidance:
                    logger.debug(f"do_classifier_free_guidance with guidance_scale: {guidance_scale}")
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                batch_size, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                # s = time.time()
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # e = time.time()
                # print('scheduler_time:',e-s)
                # reshape latents back

                latents = latents[None, :].reshape(batch_size, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                inverted_latents.append(latents.detach().clone())  # Modified

                os.makedirs(output_dir, exist_ok=True)
                s = time.time()
                torch.save(
                    latents.detach().clone(),
                    os.path.join(output_dir, f"ddim_latents_{t}.pt"),
                )
                e = time.time()
                print('save_time:', e - s)
                logger.info(f"saved noisy latents at t={t} to {output_dir}")

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # assert len(inverted_latents) == len(timesteps)
        inverted_latents = torch.stack(list(reversed(inverted_latents)), 1)

        if not return_dict:
            return inverted_latents

        # if output_type == "latent":
        #     return I2VGenXLPipelineOutput(frames=latents)
        #
        # video_tensor = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)
        # video = tensor2vid(video_tensor, self.image_processor, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        # TODO: we can return the noisy video.
        return StableVideoDiffusionInversionPipelineOutput(inverted_latents=inverted_latents)


# The following utilities are taken and adapted from
# https://github.com/ali-vilab/i2vgen-xl/blob/main/utils/transforms.py.


def _convert_pt_to_pil(image: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(image, list) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, 0)

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image_numpy = VaeImageProcessor.pt_to_numpy(image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_numpy)
        image = image_pil

    return image


def _resize_bilinear(
        image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]],
        resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        image = [u.resize(resolution, PIL.Image.BILINEAR) for u in image]
    else:
        image = image.resize(resolution, PIL.Image.BILINEAR)
    return image


def _center_crop_wide(
        image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]],
        resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        scale = min(image[0].size[0] / resolution[0], image[0].size[1] / resolution[1])
        image = [u.resize((round(u.width // scale), round(u.height // scale)), resample=PIL.Image.BOX) for u in image]

        # center crop
        x1 = (image[0].width - resolution[0]) // 2
        y1 = (image[0].height - resolution[1]) // 2
        image = [u.crop((x1, y1, x1 + resolution[0], y1 + resolution[1])) for u in image]
        return image
    else:
        scale = min(image.size[0] / resolution[0], image.size[1] / resolution[1])
        image = image.resize((round(image.width // scale), round(image.height // scale)), resample=PIL.Image.BOX)
        x1 = (image.width - resolution[0]) // 2
        y1 = (image.height - resolution[1]) // 2
        image = image.crop((x1, y1, x1 + resolution[0], y1 + resolution[1]))
        return image
