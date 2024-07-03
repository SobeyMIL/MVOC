import glob

import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torch
import yaml
from torchvision.utils import save_image
import torchvision.transforms as T
from torchvision.io import read_video, write_video
import os
import random
import numpy as np
from einops import rearrange
import logging

from typing import Optional
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward
from diffusers.models.transformers.transformer_2d import Transformer2DModel, Transformer2DModelOutput
from diffusers.utils import is_torch_version
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel, TransformerTemporalModelOutput, \
    TransformerSpatioTemporalModel
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.upsampling import Upsample2D
from diffusers.models.downsampling import Downsample2D
from functools import partial
import inspect

from typing import Dict, Any

logger = logging.getLogger(__name__)


# # Modified from tokenflow_utils.py
def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, "t", t)
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "t", t)
            module = model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "t", t)


def register_time_all(model, t, mask):
    # up resnet temp-convs
    up_blocks_id = [0, 1, 2, 3]
    up_resnets_id = [0, 1, 2]
    for blocks_id_t in up_blocks_id:
        for resnets_id in up_resnets_id:
            conv_module = model.unet.up_blocks[blocks_id_t].resnets[resnets_id]
            setattr(conv_module, "t", t)
            conv_module = model.unet.up_blocks[blocks_id_t].temp_convs[resnets_id]
            setattr(conv_module, "t", t)

            conv_module = model.unet.up_blocks[blocks_id_t].resnets[resnets_id]
            setattr(conv_module, "mask", mask)
            conv_module = model.unet.up_blocks[blocks_id_t].temp_convs[resnets_id]
            setattr(conv_module, "mask", mask)

    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "t", t)
            module = model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "t", t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "mask", mask)
            module = model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "mask", mask)

    # down注意力
    down_blocks_id = [0, 1, 2]
    down_attentions_id = [0, 1]
    down_transformer_blocks_id = [0]

    for blocks_id_t in down_blocks_id:
        for attentions_id_t in down_attentions_id:
            for transformer_blocks_t in down_transformer_blocks_id:
                module = model.unet.down_blocks[blocks_id_t].attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn1.processor
                setattr(module, "t", t)
                module = model.unet.down_blocks[blocks_id_t].temp_attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn1.processor
                setattr(module, "t", t)
                module = model.unet.down_blocks[blocks_id_t].attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn2.processor
                setattr(module, "t", t)
                module = model.unet.down_blocks[blocks_id_t].temp_attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn2.processor
                setattr(module, "t", t)
                module = model.unet.down_blocks[blocks_id_t].attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn1.processor
                setattr(module, "mask", mask)
                module = model.unet.down_blocks[blocks_id_t].temp_attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn1.processor
                setattr(module, "mask", mask)
                module = model.unet.down_blocks[blocks_id_t].attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn2.processor
                setattr(module, "mask", mask)
                module = model.unet.down_blocks[blocks_id_t].temp_attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn2.processor
                setattr(module, "mask", mask)
                # up注意力
    up_blocks_id = [1, 2, 3]
    up_attentions_id = [0, 1, 2]
    up_transformer_blocks_id = [0]

    for blocks_id_t in up_blocks_id:
        for attentions_id_t in up_attentions_id:
            for transformer_blocks_t in up_transformer_blocks_id:
                module = model.unet.up_blocks[blocks_id_t].attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn1.processor
                setattr(module, "t", t)
                module = model.unet.up_blocks[blocks_id_t].temp_attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn1.processor
                setattr(module, "t", t)
                module = model.unet.up_blocks[blocks_id_t].attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn2.processor
                setattr(module, "t", t)
                module = model.unet.up_blocks[blocks_id_t].temp_attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn2.processor
                setattr(module, "t", t)
                module = model.unet.up_blocks[blocks_id_t].attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn1.processor
                setattr(module, "mask", mask)
                module = model.unet.up_blocks[blocks_id_t].temp_attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn1.processor
                setattr(module, "mask", mask)
                module = model.unet.up_blocks[blocks_id_t].attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn2.processor
                setattr(module, "mask", mask)
                module = model.unet.up_blocks[blocks_id_t].temp_attentions[attentions_id_t].transformer_blocks[
                    transformer_blocks_t].attn2.processor
                setattr(module, "mask", mask)
    # mid注意力
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor
    setattr(module, "t", t)
    module = model.unet.mid_block.temp_attentions[0].transformer_blocks[0].attn1.processor
    setattr(module, "t", t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2.processor
    setattr(module, "t", t)
    module = model.unet.mid_block.temp_attentions[0].transformer_blocks[0].attn2.processor
    setattr(module, "t", t)

    setattr(module, "mask", mask)
    module = model.unet.mid_block.temp_attentions[0].transformer_blocks[0].attn1.processor
    setattr(module, "mask", mask)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2.processor
    setattr(module, "mask", mask)
    module = model.unet.mid_block.temp_attentions[0].transformer_blocks[0].attn2.processor
    setattr(module, "mask", mask)

    # out_conv
    module = model.unet.conv_out
    setattr(module, "t", t)
    setattr(module, "mask", mask)

    # in_conv
    module = model.unet.conv_in
    setattr(module, "t", t)
    setattr(module, "mask", mask)


def modify_diffuser_attention_forward(unet):
    def transformer_temporal_model_forward(model: TransformerTemporalModel,
                                           hidden_states: torch.FloatTensor,
                                           encoder_hidden_states: Optional[torch.LongTensor] = None,
                                           timestep: Optional[torch.LongTensor] = None,
                                           class_labels: torch.LongTensor = None,
                                           num_frames: int = 1,
                                           cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                                           return_dict: bool = True,
                                           ) -> TransformerTemporalModelOutput:
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        hidden_states = model.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        hidden_states = model.proj_in(hidden_states)

        # 2. Blocks
        for block in model.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                height=height,
                width=width,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        hidden_states = model.proj_out(hidden_states)  # torch.Size([1024, 16, 1280])->torch.Size([1024, 16, 1280])
        hidden_states = (
            hidden_states[None, None, :]
                .reshape(batch_size, height, width, num_frames, channel)
                .permute(0, 3, 4, 1, 2)
                .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)

    def basic_transformer_block_forward(model: BasicTransformerBlock,
                                        hidden_states: torch.FloatTensor,
                                        attention_mask: Optional[torch.FloatTensor] = None,
                                        encoder_hidden_states: Optional[torch.FloatTensor] = None,
                                        encoder_attention_mask: Optional[torch.FloatTensor] = None,
                                        timestep: Optional[torch.LongTensor] = None,
                                        cross_attention_kwargs: Dict[str, Any] = None,
                                        class_labels: Optional[torch.LongTensor] = None,
                                        height: Optional[int] = None,
                                        width: Optional[int] = None,
                                        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                                        ) -> torch.FloatTensor:

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if model.norm_type == "ada_norm":
            norm_hidden_states = model.norm1(hidden_states, timestep)
        elif model.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = model.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif model.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = model.norm1(hidden_states)
        elif model.norm_type == "ada_norm_continuous":
            norm_hidden_states = model.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif model.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    model.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = model.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if model.pos_embed is not None:
            norm_hidden_states = model.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = model.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if model.only_cross_attention else None,
            attention_mask=attention_mask,
            height=height,
            width=width,
            **cross_attention_kwargs,
        )
        if model.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif model.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = model.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if model.attn2 is not None:
            if model.norm_type == "ada_norm":
                norm_hidden_states = model.norm2(hidden_states, timestep)
            elif model.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = model.norm2(hidden_states)
            elif model.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif model.norm_type == "ada_norm_continuous":
                norm_hidden_states = model.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if model.pos_embed is not None and model.norm_type != "ada_norm_single":
                norm_hidden_states = model.pos_embed(norm_hidden_states)

            attn_output = model.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if model.norm_type == "ada_norm_continuous":
            norm_hidden_states = model.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not model.norm_type == "ada_norm_single":
            norm_hidden_states = model.norm3(hidden_states)

        if model.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if model.norm_type == "ada_norm_single":
            norm_hidden_states = model.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if model._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(model.ff, norm_hidden_states, model._chunk_dim, model._chunk_size)
        else:
            ff_output = model.ff(norm_hidden_states)

        if model.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif model.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    def attention_forward(model: Attention,
                          hidden_states: torch.FloatTensor,  # torch.Size([32, 8192, 320])
                          encoder_hidden_states: Optional[torch.FloatTensor] = None,  # torch.Size([32, 145, 1024])
                          attention_mask: Optional[torch.FloatTensor] = None,
                          height: Optional[int] = None,
                          width: Optional[int] = None,
                          **cross_attention_kwargs,
                          ) -> torch.Tensor:
        self = model
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
        argspec = inspect.getfullargspec(self.processor)
        needs_height = 'height' in argspec.args
        if needs_height:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                height=height,
                width=width,
                **cross_attention_kwargs,
            )
        else:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                # height=height,
                # width=width,
                **cross_attention_kwargs,
            )

    def transformer2dmodel_forward(model: Transformer2DModel,
                                   hidden_states: torch.Tensor,
                                   encoder_hidden_states: Optional[torch.Tensor] = None,
                                   timestep: Optional[torch.LongTensor] = None,
                                   added_cond_kwargs: Dict[str, torch.Tensor] = None,
                                   class_labels: Optional[torch.LongTensor] = None,
                                   cross_attention_kwargs: Dict[str, Any] = None,
                                   attention_mask: Optional[torch.Tensor] = None,
                                   encoder_attention_mask: Optional[torch.Tensor] = None,
                                   return_dict: bool = True,
                                   ):
        self = model
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states)

        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states = self.pos_embed(hidden_states)

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    height=height,
                    width=width,
                )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    for name, module in unet.named_modules():
        if isinstance(module, TransformerTemporalModel):
            module.forward = partial(transformer_temporal_model_forward, module)
        elif isinstance(module, BasicTransformerBlock):
            module.forward = partial(basic_transformer_block_forward, module)
        elif isinstance(module, Attention):
            module.forward = partial(attention_forward, module)
        elif isinstance(module, Transformer2DModel):
            module.forward = partial(transformer2dmodel_forward, module)
        else:
            pass


def register_spatial_attention_pnp(model, injection_schedule, inject_background=False):
    class ModifiedSpaAttnProcessor(AttnProcessor2_0):
        def __call__(
                self,
                attn,  # attn: Attention,
                hidden_states: torch.FloatTensor,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                temb: Optional[torch.FloatTensor] = None,
                height: Optional[int] = None,
                width: Optional[int] = None,
                scale: float = 1.0,
        ) -> torch.FloatTensor:
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            # Modified here
            # chunk_size = batch_size // 3  # batch_size is 3*chunk_size because concat[source, uncond, cond]
            chunk_size = batch_size // 5
            # chunk_size = batch_size // 2
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            args = () if USE_PEFT_BACKEND else (scale,)
            query = attn.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)

            # Modified here.
            # if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            #     logger.debug(f"PnP Injecting Spa-Attn at t={self.t}")
            #     # inject source into unconditional
            #     query[3*chunk_size : 4 * chunk_size] = query[:chunk_size]
            #     key[3*chunk_size : 4 * chunk_size] = key[:chunk_size]
            #     # inject source into conditional
            #     query[4 * chunk_size :] = query[:chunk_size]
            #     key[4 * chunk_size :] = key[:chunk_size]

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                logger.debug(f"PnP Injecting Spa-Attn at t={self.t}")
                # inject source into unconditional

                query = rearrange(query, "b (h w) c -> b h w c",
                                  h=height)  # torch.Size([80, 14400, 320])->torch.Size([80, 90, 160, 320])
                key = rearrange(key, "b (h w) c -> b h w c", h=height)
                value = rearrange(value, "b (h w) c -> b h w c", h=height)

                if self.inject_background:
                    v_inject = value[:chunk_size]
                    q_inject = query[:chunk_size]
                    k_inject = key[:chunk_size]  # torch.Size([16, 90, 160, 320])
                else:
                    # not use background
                    v_inject = value[4 * chunk_size:]
                    q_inject = query[4 * chunk_size:]
                    k_inject = key[4 * chunk_size:]  # torch.Size([16, 90, 160, 320])

                for j, obj_mask_tensor in enumerate(self.mask):
                    obj_q = query[chunk_size * (j + 1):chunk_size * (j + 2)]
                    obj_k = key[chunk_size * (j + 1):chunk_size * (j + 2)]
                    obj_v = value[chunk_size * (j + 1):chunk_size * (j + 2)]

                    obj_mask_tensor = obj_mask_tensor[1].to(torch.float16)
                    obj_mask_tensor = rearrange(obj_mask_tensor, "a b l h w -> (a b) l h w")
                    obj_mask_tensor = F.interpolate(obj_mask_tensor, size=(height, width), mode='nearest')
                    obj_mask_tensor = obj_mask_tensor[0]
                    # for i in range(16):
                    #     see_mask = obj_mask_tensor[i,:,:]
                    #     save_image(see_mask, 'mask/qkv_mask_new_%d.png'%i)
                    # torchvision.utils.save_image(a, save_path)
                    obj_mask_tensor = obj_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, query.shape[-1])

                    # q_inject= obj_q * obj_mask_tensor
                    # k_inject= obj_k * obj_mask_tensor

                    q_inject = q_inject * (1 - obj_mask_tensor) + obj_q * obj_mask_tensor
                    k_inject = k_inject * (1 - obj_mask_tensor) + obj_k * obj_mask_tensor

                query[3 * chunk_size: 4 * chunk_size] = q_inject
                key[3 * chunk_size: 4 * chunk_size] = k_inject

                query[4 * chunk_size:] = q_inject
                key[4 * chunk_size:] = k_inject

                query = rearrange(query, "b h w c -> b (h w) c")
                key = rearrange(key, "b h w c -> b (h w) c")
                value = rearrange(value, "b h w c -> b (h w) c")

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            # module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            modified_processor = ModifiedSpaAttnProcessor()
            setattr(modified_processor, "injection_schedule", injection_schedule)
            setattr(modified_processor, "inject_background", inject_background)
            module.processor = modified_processor


def register_temp_attention_pnp(model, injection_schedule, inject_background=False):  # ,conv_injection_timesteps
    class ModifiedTmpAttnProcessor(AttnProcessor2_0):
        def __call__(
                self,
                attn,  # attn: Attention,
                hidden_states: torch.FloatTensor,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                temb: Optional[torch.FloatTensor] = None,
                height: Optional[int] = None,
                width: Optional[int] = None,
                scale: float = 1.0,
        ) -> torch.FloatTensor:
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            # Modified here
            # chunk_size = batch_size // 3  # batch_size is 3*chunk_size because concat[source, uncond, cond]
            chunk_size = batch_size // 5
            # chunk_size = batch_size // 2
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            args = () if USE_PEFT_BACKEND else (scale,)
            query = attn.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)

            # Modified here.
            # if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
            #     logger.debug(f"PnP Injecting Tmp-Attn at t={self.t}")
            #     # inject source into unconditional
            #     query[chunk_size : 2 * chunk_size] = query[:chunk_size]
            #     key[chunk_size : 2 * chunk_size] = key[:chunk_size]
            #     # inject source into conditional
            #     query[2 * chunk_size :] = query[:chunk_size]
            #     key[2 * chunk_size :] = key[:chunk_size]
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                logger.debug(f"PnP Injecting Spa-Attn at t={self.t}")
                # inject source into unconditional

                query = rearrange(query, "(b h w) l c -> b h w l c", h=height, w=width)
                key = rearrange(key, "(b h w) l c -> b h w l c", h=height, w=width)
                chunk_size = key.shape[0] // 5

                if self.inject_background:
                    v_inject = value[:chunk_size]
                    q_inject = query[:chunk_size]
                    k_inject = key[:chunk_size]
                else:
                    # not use background
                    v_inject = value[4 * chunk_size:]
                    q_inject = query[4 * chunk_size:]
                    k_inject = key[4 * chunk_size:]  # torch.Size([16, 90, 160, 320])

                for j, obj_mask_tensor in enumerate(self.mask):
                    # mask resize
                    # obj_mask_tensor = obj_mask_tensor[1].to(torch.float16)
                    obj_q = query[chunk_size * (j + 1):chunk_size * (j + 2)]
                    obj_k = key[chunk_size * (j + 1):chunk_size * (j + 2)]

                    # obj_q = query[:chunk_size]
                    # obj_k = key[:chunk_size]

                    obj_mask_tensor = obj_mask_tensor[0]
                    obj_mask_tensor = obj_mask_tensor.squeeze(0)
                    obj_mask_tensor = F.interpolate(obj_mask_tensor, size=(height, width), mode='nearest')
                    obj_mask_tensor = rearrange(obj_mask_tensor, "b l h w -> b h w l")
                    obj_mask_tensor = obj_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 1, query.shape[-1])[:chunk_size]

                    q_inject = q_inject * (1 - obj_mask_tensor) + obj_q * obj_mask_tensor
                    k_inject = k_inject * (1 - obj_mask_tensor) + obj_k * obj_mask_tensor

                    # query[3*chunk_size : 4 * chunk_size][:,:height//2,:,:,:]  = q_inject[:,:height//2,:,:,:]
                    # key[3*chunk_size : 4 * chunk_size][:,:height//2,:,:,:] = k_inject[:,:height//2,:,:,:]
                    # query[4 * chunk_size :][:,:height//2,:,:,:] = q_inject[:,:height//2,:,:,:]
                    # key[4 * chunk_size :][:,:height//2,:,:,:] = k_inject[:,:height//2,:,:,:]

                    query[3 * chunk_size: 4 * chunk_size] = q_inject
                    key[3 * chunk_size: 4 * chunk_size] = k_inject
                    # inject source into conditional
                    query[4 * chunk_size:] = q_inject
                    key[4 * chunk_size:] = k_inject

                # obj1_mask_tensor = self.mask[0][0]
                # # obj1_mask_tensor = rearrange(obj1_mask_tensor,"a b l h w -> a b h w l")
                # obj1_mask_tensor = obj1_mask_tensor.squeeze(0)
                # obj1_mask_tensor=F.interpolate(obj1_mask_tensor,size=(height,width),mode='nearest')
                # obj1_mask_tensor = rearrange(obj1_mask_tensor,"b l h w -> b h w l")
                # obj1_mask_tensor = obj1_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 1, query.shape[-1])[:chunk_size]

                # obj2_mask_tensor = self.mask[1][0]
                # obj2_mask_tensor = obj2_mask_tensor.squeeze(0)
                # obj2_mask_tensor=F.interpolate(obj2_mask_tensor,size=(height,width),mode='nearest')
                # obj2_mask_tensor = rearrange(obj2_mask_tensor,"b l h w -> b h w l")
                # obj2_mask_tensor = obj2_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 1, query.shape[-1])[:chunk_size]

                # query[2*chunk_size : 3 * chunk_size] = query[2*chunk_size : 3 * chunk_size]*(1-obj1_mask_tensor) + obj1_mask_tensor*query[:chunk_size]
                # query[2*chunk_size : 3 * chunk_size] = query[2*chunk_size : 3 * chunk_size]*(1-obj1_mask_tensor) + obj1_mask_tensor*query[:chunk_size]

                # key[2*chunk_size : 3 * chunk_size] = key[2*chunk_size : 3 * chunk_size]*(1-obj2_mask_tensor) + obj2_mask_tensor*key[chunk_size:2*chunk_size]
                # key[2*chunk_size : 3 * chunk_size] = key[2*chunk_size : 3 * chunk_size]*(1-obj2_mask_tensor) + obj2_mask_tensor*key[chunk_size:2*chunk_size]
                # # inject source into conditional
                # query[3 * chunk_size :]= query[3 * chunk_size :]*(1-obj1_mask_tensor) + obj1_mask_tensor*query[:chunk_size]
                # query[3 * chunk_size :] = query[3 * chunk_size :]*(1-obj1_mask_tensor) + obj1_mask_tensor*query[:chunk_size]
                # key[3 * chunk_size :] = key[3 * chunk_size :]*(1-obj2_mask_tensor) + obj2_mask_tensor*key[chunk_size:2*chunk_size]
                # key[3 * chunk_size :] = key[3 * chunk_size :]*(1-obj2_mask_tensor) + obj2_mask_tensor*key[chunk_size:2*chunk_size]

                query = rearrange(query, "b h w l c -> (b h w) l c")
                key = rearrange(key, "b h w l c -> (b h w) l c")

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            # if self.conv_injection_timesteps is not None and (self.t in self.conv_injection_timesteps or self.t == 1000):
            #     logger.debug(f"PnP Injecting Tmp-Attn at t={self.t}")
            #     # inject source into unconditional
            #     hidden_states[chunk_size : 2 * chunk_size] = hidden_states[:chunk_size]
            #     # inject source into conditional
            #     hidden_states[2 * chunk_size :] = hidden_states[:chunk_size]
            return hidden_states

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1
            modified_processor = ModifiedTmpAttnProcessor()
            setattr(modified_processor, "injection_schedule", injection_schedule)
            setattr(modified_processor, "inject_background", inject_background)
            module.processor = modified_processor


def register_resnet_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(
                input_tensor: torch.FloatTensor,
                temb: torch.FloatTensor,
                scale: float = 1.0,
        ) -> torch.FloatTensor:
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = (
                    self.upsample(input_tensor, scale=scale)
                    if isinstance(self.upsample, Upsample2D)
                    else self.upsample(input_tensor)
                )
                hidden_states = (
                    self.upsample(hidden_states, scale=scale)
                    if isinstance(self.upsample, Upsample2D)
                    else self.upsample(hidden_states)
                )
            elif self.downsample is not None:
                input_tensor = (
                    self.downsample(input_tensor, scale=scale)
                    if isinstance(self.downsample, Downsample2D)
                    else self.downsample(input_tensor)
                )
                hidden_states = (
                    self.downsample(hidden_states, scale=scale)
                    if isinstance(self.downsample, Downsample2D)
                    else self.downsample(hidden_states)
                )

            hidden_states = self.conv1(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv1(hidden_states)

            if self.time_emb_proj is not None:
                if not self.skip_time_act:
                    temb = self.nonlinearity(temb)
                temb = (
                    self.time_emb_proj(temb, scale)[:, :, None, None]
                    if not USE_PEFT_BACKEND
                    else self.time_emb_proj(temb)[:, :, None, None]
                )

            if self.time_embedding_norm == "default":
                if temb is not None:
                    hidden_states = hidden_states + temb
                hidden_states = self.norm2(hidden_states)
            elif self.time_embedding_norm == "scale_shift":
                if temb is None:
                    raise ValueError(
                        f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                    )
                time_scale, time_shift = torch.chunk(temb, 2, dim=1)
                hidden_states = self.norm2(hidden_states)
                hidden_states = hidden_states * (1 + time_scale) + time_shift
            else:
                hidden_states = self.norm2(hidden_states)

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv2(hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                logger.debug(f"PnP Injecting Conv at t={self.t}")
                source_batch_size = int(hidden_states.shape[0] // 5)

                inject_hidden_states = hidden_states[:source_batch_size]

                # not use background
                # inject_hidden_states = hidden_states[4 * source_batch_size :]

                for j, obj_mask_tensor in enumerate(self.mask):
                    obj_hidden_states = hidden_states[source_batch_size * (j + 1):source_batch_size * (j + 2)]

                    # obj_q = query[:chunk_size]
                    # obj_k = key[:chunk_size]
                    # obj_v = value[:chunk_size]

                    obj_mask_tensor = obj_mask_tensor[1].to(torch.float16)
                    obj_mask_tensor = rearrange(obj_mask_tensor, "a b l h w -> (a b) l h w")
                    obj_mask_tensor = obj_mask_tensor[0]
                    obj_mask_tensor = obj_mask_tensor.squeeze(0)
                    # for i in range(16):
                    #     see_mask = obj_mask_tensor[i,:,:]
                    #     save_image(see_mask, 'mask/qkv_mask_new_%d.png'%i)
                    # torchvision.utils.save_image(a, save_path)
                    obj_mask_tensor = obj_mask_tensor.unsqueeze(1).repeat(1, inject_hidden_states.shape[1], 1, 1)

                    # q_inject= obj_q * obj_mask_tensor
                    # k_inject= obj_k * obj_mask_tensor

                    inject_hidden_states = inject_hidden_states * (
                            1 - obj_mask_tensor) + obj_hidden_states * obj_mask_tensor

                hidden_states[3 * source_batch_size: 4 * source_batch_size] = inject_hidden_states
                # inject conditional
                hidden_states[4 * source_batch_size:] = inject_hidden_states

                # inject unconditional
                # hidden_states[3*source_batch_size : 4 * source_batch_size] =hidden_states[3*source_batch_size : 4 * source_batch_size]*0.1 +  inject_hidden_states*0.9
                # # inject conditional
                # hidden_states[4 * source_batch_size :] =hidden_states[4 * source_batch_size :] + inject_hidden_states*0.9

            if self.conv_shortcut is not None:
                input_tensor = (
                    self.conv_shortcut(input_tensor, scale)
                    if not USE_PEFT_BACKEND
                    else self.conv_shortcut(input_tensor)
                )

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    # for net_name, net in model.unet.named_children():  #
    #     # print(net_name)
    #     if "up" in net_name:
    #         for i in range(len(model.unet.up_blocks)):
    #             for net_down_name, net_down in model.unet.up_blocks[i].named_children():
    #                 print(net_down_name,net_down)
    # conv_module = model.unet.up_blocks[0].resnets[0]
    up_blocks_id = [3]  # [0,1,2,3]
    up_resnets_id = [0, 1, 2]
    for blocks_id_t in up_blocks_id:
        for resnets_id in up_resnets_id:
            conv_module = model.unet.up_blocks[blocks_id_t].resnets[resnets_id]
            conv_module.forward = conv_forward(conv_module)
            setattr(conv_module, "injection_schedule", injection_schedule)


def register_temp_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
            hidden_states = (
                hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
            )

            identity = hidden_states
            hidden_states = self.conv1(hidden_states)
            hidden_states = self.conv2(hidden_states)
            hidden_states = self.conv3(hidden_states)
            hidden_states = self.conv4(hidden_states)

            hidden_states = identity + hidden_states

            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
                (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
            )

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                logger.debug(f"PnP Injecting Conv at t={self.t}")
                source_batch_size = int(hidden_states.shape[0] // 5)

                inject_hidden_states = hidden_states[:source_batch_size]

                # not use background
                # inject_hidden_states = hidden_states[4 * source_batch_size :]

                for j, obj_mask_tensor in enumerate(self.mask):
                    obj_hidden_states = hidden_states[source_batch_size * (j + 1):source_batch_size * (j + 2)]

                    obj_mask_tensor = obj_mask_tensor[1].to(torch.float16)
                    obj_mask_tensor = rearrange(obj_mask_tensor, "a b l h w -> (a b) l h w")
                    obj_mask_tensor = obj_mask_tensor[0]
                    obj_mask_tensor = obj_mask_tensor.squeeze(0)
                    obj_mask_tensor = obj_mask_tensor.unsqueeze(1).repeat(1, inject_hidden_states.shape[1], 1, 1)

                    inject_hidden_states = inject_hidden_states * (
                            1 - obj_mask_tensor) + obj_hidden_states * obj_mask_tensor

                hidden_states[3 * source_batch_size: 4 * source_batch_size] = inject_hidden_states
                # inject conditional
                hidden_states[4 * source_batch_size:] = inject_hidden_states

                # # inject unconditional
                # hidden_states[3*source_batch_size : 4 * source_batch_size] =hidden_states[3*source_batch_size : 4 * source_batch_size]*0.1 + inject_hidden_states*0.9
                # # inject conditional
                # hidden_states[4 * source_batch_size :] = hidden_states[4 * source_batch_size :]*0.1 + inject_hidden_states*0.9
            return hidden_states

        return forward

    # for net_name, net in model.unet.named_children():  #
    #     # print(net_name)
    #     if "up" in net_name:
    #         for i in range(len(model.unet.up_blocks)):
    #             for net_down_name, net_down in model.unet.up_blocks[i].named_children():
    #                 print(net_down_name,net_down)
    # conv_module = model.unet.up_blocks[0].resnets[0]
    up_blocks_id = [3]  # [0,1,2,3]
    up_temp_convs_id = [0, 1, 2]
    for blocks_id_t in up_blocks_id:
        for temp_convs_id in up_temp_convs_id:
            conv_module = model.unet.up_blocks[blocks_id_t].temp_convs[temp_convs_id]
            conv_module.forward = conv_forward(conv_module)
            setattr(conv_module, "injection_schedule", injection_schedule)


def register_out_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input: torch.Tensor) -> torch.Tensor:

            sample = self._conv_forward(input, self.weight, self.bias)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(sample.shape[0] // 5)

                inject_hidden_states = sample[:source_batch_size]
                # not use background
                # inject_hidden_states = sample[4 * source_batch_size :]

                for j, obj_mask_tensor in enumerate(self.mask):
                    obj_hidden_states = sample[source_batch_size * (j + 1):source_batch_size * (j + 2)]

                    # obj_q = query[:chunk_size]
                    # obj_k = key[:chunk_size]
                    # obj_v = value[:chunk_size]

                    obj_mask_tensor = obj_mask_tensor[1].to(torch.float16)
                    obj_mask_tensor = rearrange(obj_mask_tensor, "a b l h w -> (a b) l h w")
                    obj_mask_tensor = obj_mask_tensor[0]
                    obj_mask_tensor = obj_mask_tensor.squeeze(0)
                    obj_mask_tensor = obj_mask_tensor.unsqueeze(1).repeat(1, inject_hidden_states.shape[1], 1, 1)

                    # q_inject= obj_q * obj_mask_tensor
                    # k_inject= obj_k * obj_mask_tensor

                    inject_hidden_states = inject_hidden_states * (
                            1 - obj_mask_tensor) + obj_hidden_states * obj_mask_tensor

                    # sample[3*source_batch_size : 4 * source_batch_size] = sample[3*source_batch_size : 4 * source_batch_size]*(1-obj_mask_tensor)+ obj_hidden_states * obj_mask_tensor
                    # # inject conditional
                    # sample[4 * source_batch_size :] = sample[4 * source_batch_size :]*(1-obj_mask_tensor)+ obj_hidden_states * obj_mask_tensor
                # inject unconditional
                sample[3 * source_batch_size: 4 * source_batch_size] = inject_hidden_states
                # inject conditional
                sample[4 * source_batch_size:] = inject_hidden_states

                # sample[3*source_batch_size : 4 * source_batch_size] = sample[3*source_batch_size : 4 * source_batch_size]*0.1 +inject_hidden_states*0.9
                # # inject conditional
                # sample[4 * source_batch_size :] = sample[4 * source_batch_size :]*0.1 + 0.9*inject_hidden_states

            return sample

        return forward

    # register
    conv_module = model.unet.conv_out
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, "injection_schedule", injection_schedule)
