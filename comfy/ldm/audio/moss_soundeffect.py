"""MOSS SoundEffect v2 continuous audio flow model. See moss_soundeffect.LICENSE."""
import torch
from torch import nn

import comfy.model_prefetch
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.ldm.flux.math import rope
from comfy.ldm.wan.model import Head, WanAttentionBlock, WanFeedForward, sinusoidal_embedding_1d


class MossSoundEffect(nn.Module):
    def __init__(self, dim=1536, in_dim=128, ffn_dim=8960, out_dim=128, text_dim=2048, freq_dim=256, eps=1e-6, patch_size=(1,), num_heads=12, num_layers=30, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.head_dim = dim // num_heads
        settings = dict(operations=operations, dtype=dtype, device=device)
        self.patch_embedding = operations.Conv1d(in_dim, dim, patch_size[0], stride=patch_size[0], dtype=dtype, device=device)
        self.text_embedding = WanFeedForward(operations.Linear(text_dim, dim, dtype=dtype, device=device), nn.GELU(approximate='tanh'), operations.Linear(dim, dim, dtype=dtype, device=device))
        self.time_embedding = nn.Sequential(operations.Linear(freq_dim, dim, dtype=dtype, device=device), nn.SiLU(), operations.Linear(dim, dim, dtype=dtype, device=device))
        self.time_projection = nn.Sequential(nn.SiLU(), operations.Linear(dim, 6 * dim, dtype=dtype, device=device))
        self.blocks = nn.ModuleList([WanAttentionBlock('t2v_cross_attn', dim, ffn_dim, num_heads, cross_attn_norm=True, eps=eps, operation_settings=settings) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps=eps, operation_settings=settings)

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        length = x.shape[-1]
        time = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=x.dtype))
        modulation = self.time_projection(time).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        x = self.patch_embedding(pad_to_patch_size(x, self.patch_size)).transpose(1, 2)
        frequencies = rope(torch.arange(x.shape[1], device=x.device).unsqueeze(0), self.head_dim, 10000).unsqueeze(2)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, transformer_options)
        for block in self.blocks:
            def core():
                nonlocal x
                x = block(x, modulation, frequencies, context, transformer_options=transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, malloc_scope='block')
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        x = self.head(x, time).unflatten(-1, (self.patch_size[0], -1))
        return x.permute(0, 3, 1, 2).flatten(2)[..., :length]
