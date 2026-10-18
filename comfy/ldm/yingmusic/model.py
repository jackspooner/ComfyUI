"""YingMusic Singer Plus DiT, adapted from ASLP-lab/YingMusic-Singer-Plus (CC BY 4.0)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_prefetch
import comfy.ops
from comfy.ldm.flux.layers import modulated_norm
from comfy.ldm.minimax_music.dit import RotaryEmbedding
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.flux.math import apply_rope


class GRN(nn.Module):
    def __init__(self, dim, dtype, device):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(1, 1, dim, dtype=dtype, device=device))
        self.beta = nn.Parameter(torch.empty(1, 1, dim, dtype=dtype, device=device))

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        norm = norm / (norm.mean(dim=-1, keepdim=True) + 1e-6)
        return torch.addcmul(x + comfy.ops.cast_to_input(self.beta, x), comfy.ops.cast_to_input(self.gamma, x), x * norm)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.dwconv = operations.Conv1d(dim, dim, 7, padding=3, groups=dim, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.pwconv1 = operations.Linear(dim, dim * 2, dtype=dtype, device=device)
        self.grn = GRN(dim * 2, dtype, device)
        self.pwconv2 = operations.Linear(dim * 2, dim, dtype=dtype, device=device)

    def forward(self, x):
        hidden = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.pwconv2(self.grn(F.gelu(self.pwconv1(self.norm(hidden)))))


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, depth, dtype, device, operations):
        super().__init__()
        self.text_embed = operations.Embedding(vocab_size, dim, dtype=dtype, device=device)
        self.text_blocks = nn.Sequential(*[ConvNeXtBlock(dim, dtype, device, operations) for _ in range(depth)])

    def forward(self, tokens, dtype):
        hidden = self.text_embed(tokens, out_dtype=dtype)
        dim = hidden.shape[-1]
        frequencies = 10000 ** (-torch.arange(0, dim, 2, device=hidden.device, dtype=torch.float32) / dim)
        positions = torch.arange(hidden.shape[1], device=hidden.device, dtype=torch.float32).clamp(max=4095)
        phases = torch.outer(positions, frequencies)
        hidden = hidden + torch.cat((phases.cos(), phases.sin()), dim=-1).to(hidden.dtype)
        return self.text_blocks(hidden)


class InputEmbedding(nn.Module):
    def __init__(self, dim, text_dim, dtype, device, operations):
        super().__init__()
        self.midi_proj = operations.Linear(128, 128, dtype=dtype, device=device)
        self.proj = operations.Linear(128 + text_dim + 128, dim, dtype=dtype, device=device)
        self.conv_pos_embed = nn.Module()
        self.conv_pos_embed.conv1d = nn.Sequential(
            operations.Conv1d(dim, dim, 31, groups=16, padding=15, dtype=dtype, device=device), nn.Mish(),
            operations.Conv1d(dim, dim, 31, groups=16, padding=15, dtype=dtype, device=device), nn.Mish(),
        )

    def forward(self, x, reference, text, midi):
        hidden = self.proj(torch.cat((x, reference, text, midi), dim=-1))
        return hidden + self.conv_pos_embed.conv1d(hidden.transpose(1, 2)).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads, dtype, device, operations):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.to_q = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_out = nn.Sequential(operations.Linear(dim, dim, dtype=dtype, device=device), nn.Identity())

    def forward(self, x, rope, transformer_options={}):
        batch, length, dim = x.shape
        q = self.to_q(x).reshape(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        k = self.to_k(x).reshape(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        v = self.to_v(x).reshape(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        q, k = apply_rope(q, k, rope)
        result = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                                     self.heads, skip_reshape=True, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
        return self.to_out(result)


class AdaNorm(nn.Module):
    def __init__(self, dim, count, dtype, device, operations):
        super().__init__()
        self.linear = operations.Linear(dim, dim * count, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dtype, device, operations):
        super().__init__()
        self.attn_norm = AdaNorm(dim, 6, dtype, device, operations)
        self.attn = Attention(dim, heads, dtype, device, operations)
        self.ff_norm = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.ff = nn.Module()
        self.ff.ff = nn.Sequential(
            nn.Sequential(operations.Linear(dim, dim * 2, dtype=dtype, device=device), nn.GELU(approximate="tanh")),
            nn.Identity(), operations.Linear(dim * 2, dim, dtype=dtype, device=device),
        )

    def forward(self, x, time, rope, transformer_options={}):
        shift, scale, gate, ff_shift, ff_scale, ff_gate = self.attn_norm.linear(F.silu(time))[:, None].chunk(6, dim=-1)
        x = torch.addcmul(x, gate, self.attn(modulated_norm(x, self.attn_norm.norm, scale, shift), rope, transformer_options))
        return torch.addcmul(x, ff_gate, self.ff.ff(modulated_norm(x, self.ff_norm, ff_scale, ff_shift)))


class YingMusic(nn.Module):
    def __init__(self, dim=1024, depth=22, heads=16, text_dim=512, text_depth=4, vocab_size=374,
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.time_embed = nn.Module()
        self.time_embed.time_mlp = nn.Sequential(
            operations.Linear(256, dim, dtype=dtype, device=device), nn.SiLU(),
            operations.Linear(dim, dim, dtype=dtype, device=device),
        )
        self.text_embed_p = TextEmbedding(vocab_size, text_dim, text_depth, dtype, device, operations)
        self.input_embed_with_midi = InputEmbedding(dim, text_dim, dtype, device, operations)
        self.rotary_embed = RotaryEmbedding(dim // heads, dtype, device)
        self.transformer_blocks = nn.ModuleList([DiTBlock(dim, heads, dtype, device, operations) for _ in range(depth)])
        self.norm_out = AdaNorm(dim, 2, dtype, device, operations)
        self.proj_out = operations.Linear(dim, 64, dtype=dtype, device=device)

    def preprocess_conditioning(self, tokens, midi, reference, unconditioned=False):
        # The null branch drops melody AFTER its biased projection, matching upstream CFG.
        tokens = torch.zeros_like(tokens) if unconditioned else tokens + 1
        text = self.text_embed_p(tokens, reference.dtype)
        midi = self.input_embed_with_midi.midi_proj(midi)
        if unconditioned:
            midi = torch.zeros_like(midi)
            reference = torch.zeros_like(reference)
        return torch.cat((reference.transpose(1, 2), text, midi), dim=-1)

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        reference, text, midi = context.split((64, self.text_embed_p.text_embed.embedding_dim, 128), dim=-1)
        hidden = self.input_embed_with_midi(x.transpose(1, 2), reference, text, midi)
        frequencies = torch.exp(torch.arange(128, device=x.device, dtype=torch.float32) * (-math.log(10000) / 127))
        phases = 1000 * timestep[:, None] * frequencies[None]
        time = self.time_embed.time_mlp(torch.cat((phases.sin(), phases.cos()), dim=-1).to(x.dtype))
        rope = self.rotary_embed.forward_from_seq_len(hidden.shape[1], x.device, x.dtype)
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), x.device, transformer_options)
        for block in self.transformer_blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, block, hidden.dtype)
            hidden = block(hidden, time, rope, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
        scale, shift = self.norm_out.linear(F.silu(time))[:, None].chunk(2, dim=-1)
        # ComfyUI integrates decreasing sigma; upstream integrates increasing flow time.
        return -self.proj_out(modulated_norm(hidden, self.norm_out.norm, scale, shift)).transpose(1, 2)
