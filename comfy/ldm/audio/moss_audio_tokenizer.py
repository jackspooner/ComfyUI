# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""Native MOSS v1 codec math, with streaming state scoped to encode/decode calls.

Adapted from OpenMOSS-Team/MOSS-Audio-Tokenizer at
3cd226ba2947efa357ef453bcad111b6eafba782. Weight-normalized 1x1 convolutions
are consumed as folded FP32 inference weights; model lifecycle belongs to the
discrete codec owner, not this module.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device


def rotary_matrix(length, head_dim, offset, device):
    frequencies = torch.exp(torch.arange(head_dim // 2, device=device, dtype=torch.float32) * (-math.log(10000.0) * 2 / head_dim))
    positions = torch.arange(length, device=device, dtype=torch.float32) + offset
    angles = positions[:, None] * frequencies[None, :]
    cos, sin = angles.cos(), angles.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, length, head_dim // 2, 2, 2)


class LayerScale(nn.Module):
    def __init__(self, channels, device=None, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(channels, device=device, dtype=dtype))

    def forward(self, x):
        return x * comfy.ops.cast_to_input(self.scale, x)


class CodecAttention(nn.Module):
    def __init__(self, dim, heads, context, device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.context = context
        self.in_projs = nn.ModuleList([operations.Linear(dim, 3 * dim, bias=False, device=device, dtype=dtype)])
        self.out_projs = nn.ModuleList([operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)])
        self.comfy_attention = ComfyAttention()

    def forward(self, x, rotary, offset=0, cache=None):
        batch, length, dim = x.shape
        q, k, v = self.in_projs[0](x).reshape(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, k, rotary)
        if cache is not None:
            if not cache:
                shape = (batch, self.heads, self.context, dim // self.heads)
                cache.extend([torch.zeros(shape, dtype=k.dtype, device=k.device), torch.zeros(shape, dtype=v.dtype, device=v.device)])
            indices = (torch.arange(length, device=k.device) + offset) % self.context
            cache[0].index_copy_(2, indices, k)
            cache[1].index_copy_(2, indices, v)
            k, v = cache
            slots = torch.arange(self.context, device=x.device)
            last = offset + length - 1
            delta = slots - last % self.context
            key_positions = torch.where(delta <= 0, last + delta, last + delta - self.context)
            key_positions = torch.where(slots < offset + length, key_positions, -1)
        else:
            key_positions = torch.arange(length, device=x.device) + offset
        query_positions = torch.arange(length, device=x.device) + offset
        delta = query_positions[:, None] - key_positions[None, :]
        valid = (key_positions[None, :] >= 0) & (delta >= 0) & (delta < self.context)
        mask = torch.zeros(delta.shape, dtype=x.dtype, device=x.device).masked_fill_(~valid, -torch.inf)
        attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
        x = attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                      self.heads, mask=mask, skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.out_projs[0](x)


class CodecTransformerLayer(nn.Module):
    def __init__(self, dim, heads, intermediate, context, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = CodecAttention(dim, heads, context, device=device, dtype=dtype, operations=operations)
        self.norm1 = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.linear1 = operations.Linear(dim, intermediate, bias=False, device=device, dtype=dtype)
        self.linear2 = operations.Linear(intermediate, dim, bias=False, device=device, dtype=dtype)
        self.layer_scale_1 = LayerScale(dim, device=device, dtype=dtype)
        self.layer_scale_2 = LayerScale(dim, device=device, dtype=dtype)

    def forward(self, x, rotary, offset=0, cache=None):
        x = x + self.layer_scale_1(self.self_attn(self.norm1(x), rotary, offset, cache))
        return x + self.layer_scale_2(self.linear2(F.gelu(self.linear1(self.norm2(x)))))


class CodecTransformer(nn.Module):
    def __init__(self, dim, heads, intermediate, depth, context, device=None, dtype=None, operations=None):
        super().__init__()
        self.head_dim = dim // heads
        self.layers = nn.ModuleList([CodecTransformerLayer(dim, heads, intermediate, context, device=device, dtype=dtype, operations=operations) for _ in range(depth)])

    def forward(self, x, state=None):
        offset = state['offset'] if state is not None else 0
        rotary = rotary_matrix(x.shape[1], self.head_dim, offset, x.device)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {'prefetch_dynamic_vbars': True})
        for i, layer in enumerate(self.layers):
            comfy.model_management.throw_exception_if_processing_interrupted()
            cache = state['layers'][i] if state is not None else None
            def core():
                nonlocal x
                x = layer(x, rotary, offset, cache)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        if state is not None:
            state['offset'] += x.shape[1]
        return x


class ProjectedTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dim, heads, intermediate, depth, context, device=None, dtype=None, operations=None):
        super().__init__()
        self.input_proj = operations.Linear(input_dim, dim, bias=False, device=device, dtype=dtype) if input_dim != dim else nn.Identity()
        self.transformer = CodecTransformer(dim, heads, intermediate, depth, context, device=device, dtype=dtype, operations=operations)
        self.output_proj = operations.Linear(dim, output_dim, bias=False, device=device, dtype=dtype) if output_dim != dim else nn.Identity()

    def forward(self, x, state=None):
        return self.output_proj(self.transformer(self.input_proj(x.transpose(1, 2)), state)).transpose(1, 2)


class Patch(nn.Module):
    def __init__(self, size, downsample):
        super().__init__()
        self.size = size
        self.downsample = downsample

    def forward(self, x):
        batch, channels, length = x.shape
        if self.downsample:
            return x.reshape(batch, channels, length // self.size, self.size).permute(0, 1, 3, 2).reshape(batch, channels * self.size, length // self.size)
        return x.reshape(batch, channels // self.size, self.size, length).permute(0, 1, 3, 2).reshape(batch, channels // self.size, length * self.size)


class Codebook(nn.Module):
    def __init__(self, dim, code_dim, size, device=None, dtype=None, operations=None):
        super().__init__()
        self.size = size
        self.in_proj = operations.Conv1d(dim, code_dim, 1, device=device, dtype=dtype)
        self.out_proj = operations.Conv1d(code_dim, dim, 1, device=device, dtype=dtype)
        self.codebook = operations.Embedding(size, code_dim, device=device, dtype=dtype)

    def decode(self, codes):
        return self.out_proj(self.codebook(codes).transpose(1, 2).float()).float()

    def encode(self, x):
        z = self.in_proj(x.float()).float()
        encodings = F.normalize(z.transpose(1, 2).reshape(-1, z.shape[1]))
        codebook = F.normalize(self.codebook(torch.arange(self.size, device=x.device)).float())
        distance = encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t()
        codes = (-distance).max(1)[1].reshape(x.shape[0], -1)
        # Preserve the reference's FP32 straight-through arithmetic, including rounding.
        quantized = self.codebook(codes).transpose(1, 2).float()
        quantized = z + (quantized - z)
        return self.out_proj(quantized).float(), codes


class ResidualLFQ(nn.Module):
    def __init__(self, input_dim=768, dim=512, output_dim=768, count=32, code_dim=8, size=1024, device=None, dtype=None, operations=None):
        super().__init__()
        self.dim = dim
        self.input_proj = operations.Conv1d(input_dim, dim, 1, device=device, dtype=dtype)
        self.output_proj = operations.Conv1d(dim, output_dim, 1, device=device, dtype=dtype)
        self.quantizers = nn.ModuleList([Codebook(dim, code_dim, size, device=device, dtype=dtype, operations=operations) for _ in range(count)])

    def decode(self, codes):
        output = torch.zeros(codes.shape[0], self.dim, codes.shape[2], device=codes.device, dtype=torch.float32)
        for i in range(codes.shape[1]):
            comfy.model_management.throw_exception_if_processing_interrupted()
            output += self.quantizers[i].decode(codes[:, i])
        return self.output_proj(output)

    def encode(self, x, lengths, count):
        residual = self.input_proj(x).float()
        mask = torch.arange(x.shape[-1], device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        codes = []
        for quantizer in self.quantizers[:count or len(self.quantizers)]:
            comfy.model_management.throw_exception_if_processing_interrupted()
            quantized, indices = quantizer.encode(residual * mask[:, None])
            residual = residual - quantized * mask[:, None]
            codes.append(indices)
        return torch.stack(codes, dim=1)


class MossAudioTokenizer(nn.Module):
    sample_rate = 24000
    hop_length = 1920

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        settings = dict(device=device, dtype=dtype, operations=operations)
        self.encoder = nn.ModuleList([
            Patch(240, True), ProjectedTransformer(240, 384, 768, 12, 3072, 12, 1000, **settings),
            Patch(2, True), ProjectedTransformer(768, 384, 768, 12, 3072, 12, 500, **settings),
            Patch(2, True), ProjectedTransformer(768, 640, 768, 12, 3072, 12, 250, **settings),
            Patch(2, True), ProjectedTransformer(1280, 768, 1280, 20, 5120, 32, 125, **settings),
        ])
        self.quantizer = ResidualLFQ(**settings)
        self.decoder = nn.ModuleList([
            ProjectedTransformer(768, 1280, 1280, 20, 5120, 32, 125, **settings), Patch(2, False),
            ProjectedTransformer(640, 768, 768, 12, 3072, 12, 250, **settings), Patch(2, False),
            ProjectedTransformer(384, 768, 768, 12, 3072, 12, 500, **settings), Patch(2, False),
            ProjectedTransformer(384, 240, 768, 12, 3072, 12, 1000, **settings), Patch(240, False),
        ])

    def encode(self, waveform, lengths, count=32):
        x = F.pad(waveform, (0, -waveform.shape[-1] % self.hop_length))
        for module in self.encoder:
            x = module(x)
        code_lengths = [length // self.hop_length for length in lengths]
        return self.quantizer.encode(x, code_lengths, count), code_lengths

    def decode(self, codes, chunk_frames=100):
        # Match the reference: short inputs use full attention, longer inputs use
        # the bounded ring which inserts each entire chunk before evaluating it.
        states = [({'offset': 0, 'layers': [[] for _ in module.transformer.layers]} if isinstance(module, ProjectedTransformer) else None) for module in self.decoder]
        streaming = codes.shape[-1] > chunk_frames
        output = []
        for start in range(0, codes.shape[-1], chunk_frames):
            comfy.model_management.throw_exception_if_processing_interrupted()
            x = self.quantizer.decode(codes[:, :, start:start + chunk_frames])
            for module, state in zip(self.decoder, states):
                x = module(x, state) if streaming and state is not None else module(x)
            output.append(x)
        return torch.cat(output, dim=-1)
