# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# Adapted from nvidia/LocateAnything-3B c32291ca5e996f5a7a485845b4f57a233936bba0.
# NVIDIA License; see LICENSE in this directory.
"""MoonViT and the block-difference Qwen2 decoder on native operations."""
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy.ops
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device
from comfy.text_encoders.llama import Llama2_, Qwen25_3BConfig


def inference_mask(qlen, past, mtp, device, dtype, block_size=6):
    """MTP queries see their window, but never its duplicate prefix token."""
    if qlen == 1:
        return None
    mask = torch.full((qlen, past+qlen), torch.finfo(dtype).min, device=device, dtype=dtype).triu_(past+1)
    if mtp:
        mask[-block_size:, -block_size:] = 0
        mask[-block_size:, -block_size-1] = -torch.inf
    return mask


def vision_rotary(height, width, dim, device):
    indices = torch.arange(height*width, device=device)
    frequencies = 1./(10000.**(torch.arange(0, dim, 4, device=device, dtype=torch.float32)/dim))
    angles = torch.stack(((indices % width)[:, None]*frequencies,
                          (indices // width)[:, None]*frequencies), dim=-1).flatten(1)
    return torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), -1).reshape(1, 1, height*width, dim//2, 2, 2)


class PositionEmbedding(nn.Module):
    def __init__(self, height, width, dim, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(height, width, dim, device=device, dtype=dtype))

    def forward(self, x, grid):
        weight = comfy.ops.cast_to_input(self.weight, x)
        if tuple(grid) != weight.shape[:2]:
            weight = F.interpolate(weight.permute(2, 0, 1)[None], size=grid, mode='bicubic')[0].permute(1, 2, 0)
        return x+weight.flatten(0, 1)


class PatchEmbedding(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        dim, patch = config['hidden_size'], config['patch_size']
        self.proj = operations.Conv2d(3, dim, patch, stride=patch, dtype=dtype, device=device)
        self.pos_emb = PositionEmbedding(config['init_pos_emb_height'], config['init_pos_emb_width'], dim, dtype, device)

    def forward(self, pixels, grid):
        return self.pos_emb(self.proj(pixels).flatten(1), grid)


class VisionMLP(nn.Module):
    def __init__(self, dim, inner, dtype, device, operations):
        super().__init__()
        self.fc0 = operations.Linear(dim, inner, dtype=dtype, device=device)
        self.fc1 = operations.Linear(inner, dim, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc1(F.gelu(self.fc0(x), approximate='tanh'))


class VisionBlock(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        dim = config['hidden_size']
        self.heads = config['num_attention_heads']
        self.norm0 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = VisionMLP(dim, config['intermediate_size'], dtype, device, operations)
        self.wqkv = operations.Linear(dim, 3*dim, dtype=dtype, device=device)
        self.wo = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, rotary):
        q, k, v = self.wqkv(self.norm0(x)).reshape(x.shape[0], 3, self.heads, -1).permute(1, 2, 0, 3).unbind(0)
        q, k = apply_rope(q[None], k[None], rotary)
        attention = optimized_attention_for_device(x.device, small_input=True)
        value = attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v[None]),
            self.heads, skip_reshape=True, preferred_attention=self.comfy_attention)[0]
        x = x+self.wo(value)
        return x+self.mlp(self.norm1(x))


class VisionEncoder(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.blocks = nn.ModuleList([VisionBlock(config, dtype, device, operations) for _ in range(config['num_hidden_layers'])])
        self.final_layernorm = operations.LayerNorm(config['hidden_size'], dtype=dtype, device=device)
        self.head_dim = config['hidden_size']//config['num_attention_heads']

    def forward(self, x, grid, options):
        rotary = vision_rotary(*grid, self.head_dim, x.device)
        queue = prefetch.make_prefetch_queue(list(self.blocks), x.device, options)
        for block in self.blocks:
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = block(x, rotary)
            prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, malloc_scope='block')
        prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        return self.final_layernorm(x)


class MoonVision(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.patch_embed = PatchEmbedding(config, dtype, device, operations)
        self.encoder = VisionEncoder(config, dtype, device, operations)
        self.merge = tuple(config['merge_kernel_size'])

    def forward(self, pixels, grid, options):
        x = self.encoder(self.patch_embed(pixels, grid), grid, options)
        h, w = grid
        mh, mw = self.merge
        return x.reshape(h//mh, mh, w//mw, mw, -1).permute(0, 2, 1, 3, 4).reshape(h*w//(mh*mw), -1)


class Vision(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        vision = config['vision_config']
        dim, hidden = vision['hidden_size']*4, config['text_config']['hidden_size']
        self.vision_model = MoonVision(vision, dtype, device, operations)
        self.mlp1 = nn.Sequential(operations.LayerNorm(dim, dtype=dtype, device=device),
            operations.Linear(dim, hidden, dtype=dtype, device=device), nn.GELU(),
            operations.Linear(hidden, hidden, dtype=dtype, device=device))

    def forward(self, pixels, grid, options):
        return self.mlp1(self.vision_model(pixels, grid, options))


class Language(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        values = config['text_config']
        native = replace(Qwen25_3BConfig(), **{key: values[key] for key in (
            'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads', 'rms_norm_eps', 'rope_theta', 'max_position_embeddings')})
        native.head_dim = native.hidden_size//native.num_attention_heads
        self.model = Llama2_(native, device=device, dtype=dtype, ops=operations)
        self.lm_head = operations.Linear(native.hidden_size, native.vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self, ids, positions, cache, mtp, options, visual=None, image_token=151665):
        x = self.model.embed_tokens(ids, out_dtype=self.lm_head.weight.dtype)
        if visual is not None:
            selected = ids == image_token
            if int(selected.sum()) != visual.shape[0]:
                raise ValueError('LocateAnything image placeholders do not match the prepared visual features')
            x[selected] = visual
        past = cache[0][2] if cache[0] else 0
        mask = inference_mask(ids.shape[1], past, mtp, x.device, x.dtype)
        rotary = self.model.compute_freqs_cis(positions, x.device)
        attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
        queue = prefetch.make_prefetch_queue(list(self.model.layers), x.device, options)
        for i, block in enumerate(self.model.layers):
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x, cache[i] = block(x, attention_mask=mask, freqs_cis=rotary,
                    optimized_attention=attention, past_key_value=cache[i])
            prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, malloc_scope='block')
        prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        # Only the prediction window needs the 152681-way output projection.
        return self.lm_head(self.model.norm(x[:, -6:] if mtp else x[:, -1:])).float()
