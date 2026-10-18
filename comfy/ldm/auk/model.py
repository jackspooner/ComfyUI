"""AuK-Flash (Tencent, MIT): joint text/audio flow transformer."""
import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.flux.layers import modulated_norm
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, device):
        super().__init__()
        self.register_buffer('inv_freq', torch.empty(head_dim // 2, dtype=torch.float32, device=device))


def rotary(length, frequencies):
    # x-transformers rotates adjacent pairs, with frequencies repeated per pair.
    angles = torch.arange(length, device=frequencies.device, dtype=torch.float32)[:, None] * frequencies[None, :]
    cosine, sine = angles.cos(), angles.sin()
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(1, 1, length, frequencies.shape[0], 2, 2)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, operations, dtype, device):
        super().__init__()
        self.time_mlp = nn.Sequential(operations.Linear(256, dim, dtype=dtype, device=device), nn.SiLU(),
                                      operations.Linear(dim, dim, dtype=dtype, device=device))

    def forward(self, t, dtype):
        frequencies = torch.exp(torch.arange(128, device=t.device, dtype=torch.float32) * (-math.log(10000) / 127))
        angles = 1000 * t[:, None] * frequencies[None, :]
        return self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=-1).to(dtype))


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, operations, dtype, device):
        super().__init__()
        self.conv1d = nn.Sequential(operations.Conv1d(dim, dim, 31, padding=15, groups=16, dtype=dtype, device=device),
                                    nn.Mish(), operations.Conv1d(dim, dim, 31, padding=15, groups=16, dtype=dtype, device=device), nn.Mish())

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        if mask is not None:
            mask = mask[:, None, :]
            x = x.masked_fill(~mask, 0)
        for i, layer in enumerate(self.conv1d):
            x = layer(x)
            if mask is not None and i % 2 == 0:
                x = x.masked_fill(~mask, 0)
        return x.transpose(1, 2)


class AudioPromptEmbedding(nn.Module):
    def __init__(self, dim, latent_dim, operations, dtype, device):
        super().__init__()
        self.linear = operations.Linear(latent_dim, dim, dtype=dtype, device=device)
        self.conv_pos_embed = ConvPositionEmbedding(dim, operations, dtype, device)

    def forward(self, x, mask=None):
        x = self.linear(x)
        return x + self.conv_pos_embed(x, mask)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim, operations, dtype, device, final=False):
        super().__init__()
        self.final = final
        self.linear = operations.Linear(dim, dim * (2 if final else 6), dtype=dtype, device=device)
        self.norm = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

    def forward(self, x, t):
        modulation = self.linear(F.silu(t))[:, None, :]
        if self.final:
            scale, shift = modulation.chunk(2, dim=-1)
            return modulated_norm(x, self.norm, scale, shift)
        shift, scale, gate, ff_shift, ff_scale, ff_gate = modulation.chunk(6, dim=-1)
        return modulated_norm(x, self.norm, scale, shift), gate, ff_shift, ff_scale, ff_gate


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult, operations, dtype, device):
        super().__init__()
        inner = int(dim * ff_mult)
        self.linear_in = operations.Linear(dim, inner * 2, bias=False, dtype=dtype, device=device)
        self.linear_out = operations.Linear(inner, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        gate, value = self.linear_in(x).chunk(2, dim=-1)
        return self.linear_out(F.silu(gate) * value)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, joint, operations, dtype, device):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.comfy_attention = ComfyAttention()
        self.to_qkv = operations.Linear(dim, 3 * heads * head_dim, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(head_dim, eps=None, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=None, dtype=dtype, device=device)
        self.to_out = nn.Sequential(operations.Linear(heads * head_dim, dim, dtype=dtype, device=device), nn.Identity())
        if joint:
            self.to_qkv_c = operations.Linear(dim, 3 * heads * head_dim, dtype=dtype, device=device)
            self.c_q_norm = operations.RMSNorm(head_dim, eps=None, dtype=dtype, device=device)
            self.c_k_norm = operations.RMSNorm(head_dim, eps=None, dtype=dtype, device=device)
            self.to_out_c = operations.Linear(heads * head_dim, dim, dtype=dtype, device=device)

    def qkv(self, x, projection, q_norm, k_norm, rope):
        q, k, v = [value.reshape(x.shape[0], -1, self.heads, self.head_dim).transpose(1, 2) for value in projection(x).chunk(3, dim=-1)]
        q, k = apply_rope(q_norm(q), k_norm(k), rope)
        return q, k, v

    def forward(self, x, rope, c=None, c_rope=None, mask=None, c_mask=None, transformer_options={}):
        q, k, v = self.qkv(x, self.to_qkv, self.q_norm, self.k_norm, rope)
        attention_mask = mask
        if c is not None:
            cq, ck, cv = self.qkv(c, self.to_qkv_c, self.c_q_norm, self.c_k_norm, c_rope)
            q, k, v = torch.cat((q, cq), dim=2), torch.cat((k, ck), dim=2), torch.cat((v, cv), dim=2)
            if mask is not None:
                attention_mask = torch.cat((mask, c_mask), dim=1)
        if attention_mask is not None:
            attention_mask = torch.zeros_like(attention_mask, dtype=x.dtype).masked_fill(~attention_mask, torch.finfo(x.dtype).min)[:, None, None, :]
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        output = optimized_attention(q, k, v, self.heads, skip_reshape=True, mask=attention_mask,
                                     transformer_options=transformer_options, preferred_attention=self.comfy_attention)
        audio = self.to_out(output[:, :x.shape[1]])
        if mask is not None:
            audio = audio.masked_fill(~mask[:, :, None], 0)
        if c is None:
            return audio
        context = self.to_out_c(output[:, x.shape[1]:])
        if c_mask is not None:
            context = context.masked_fill(~c_mask[:, :, None], 0)
        return audio, context


class DoubleBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, ff_mult, operations, dtype, device):
        super().__init__()
        self.attn = Attention(dim, heads, head_dim, True, operations, dtype, device)
        for name in ('x', 'c'):
            setattr(self, 'attn_norm_' + name, AdaLayerNorm(dim, operations, dtype, device))
            setattr(self, 'ff_norm_' + name, operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device))
            setattr(self, 'ff_' + name, FeedForward(dim, ff_mult, operations, dtype, device))

    def forward(self, x, c, t, rope, c_rope, mask, c_mask, transformer_options):
        nx, gx, sx, ax, fx = self.attn_norm_x(x, t)
        nc, gc, sc, ac, fc = self.attn_norm_c(c, t)
        dx, dc = self.attn(nx, rope, nc, c_rope, mask, c_mask, transformer_options)
        x, c = torch.addcmul(x, gx, dx), torch.addcmul(c, gc, dc)
        x = torch.addcmul(x, fx, self.ff_x(modulated_norm(x, self.ff_norm_x, ax, sx)))
        c = torch.addcmul(c, fc, self.ff_c(modulated_norm(c, self.ff_norm_c, ac, sc)))
        return x, c


class SingleBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, ff_mult, operations, dtype, device):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim, operations, dtype, device)
        self.attn = Attention(dim, heads, head_dim, False, operations, dtype, device)
        self.ff_norm = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.ff = FeedForward(dim, ff_mult, operations, dtype, device)

    def forward(self, x, t, rope, mask, transformer_options):
        norm, gate, shift, scale, ff_gate = self.attn_norm(x, t)
        x = torch.addcmul(x, gate, self.attn(norm, rope, mask=mask, transformer_options=transformer_options))
        return torch.addcmul(x, ff_gate, self.ff(modulated_norm(x, self.ff_norm, scale, shift)))


class AuK(nn.Module):
    def __init__(self, dim=1536, heads=24, head_dim=64, ff_mult=2, text_hidden_dim=2048, latent_dim=64,
                 num_layers=10, num_single_layers=20, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.rotary_embed = RotaryEmbedding(head_dim, device)
        self.time_embed = TimestepEmbedding(dim, operations, dtype, device)
        self.txt_proj = operations.Linear(text_hidden_dim, dim, dtype=dtype, device=device)
        self.txt_norm = operations.RMSNorm(dim, eps=None, dtype=dtype, device=device)
        self.audio_embed = AudioPromptEmbedding(dim, latent_dim, operations, dtype, device)
        self.transformer_blocks = nn.ModuleList([DoubleBlock(dim, heads, head_dim, ff_mult, operations, dtype, device) for _ in range(num_layers)])
        self.single_transformer_blocks = nn.ModuleList([SingleBlock(dim, heads, head_dim, ff_mult, operations, dtype, device) for _ in range(num_single_layers)])
        self.norm_out = AdaLayerNorm(dim, operations, dtype, device, final=True)
        self.proj_out = operations.Linear(dim, latent_dim, dtype=dtype, device=device)

    def get_dynamic_vram__units(self):
        return list(self.transformer_blocks) + list(self.single_transformer_blocks), []

    def preprocess_text(self, context):
        return self.txt_norm(self.txt_proj(context))

    def forward(self, x, timestep, context, attention_mask=None, reference=None, reference_mask=None, transformer_options={}, **kwargs):
        t = self.time_embed(1 - timestep, x.dtype)
        c = context
        if reference_mask is not None:
            reference_mask = reference_mask.bool()
        x = self.audio_embed(x.transpose(1, 2))
        prompt_length = 0
        audio_mask = None
        if reference is not None and reference.shape[-1]:
            ref = self.audio_embed(reference.transpose(1, 2), reference_mask)
            prompt_length = ref.shape[1]
            x = torch.cat((ref, x), dim=1)
            audio_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            if reference_mask is not None:
                audio_mask[:, :prompt_length] = reference_mask
        c_mask = None if attention_mask is None else attention_mask.bool()
        if c_mask is None:
            c_mask = torch.ones(c.shape[:2], dtype=torch.bool, device=c.device)
        frequencies = comfy.model_management.cast_to(self.rotary_embed.inv_freq, dtype=torch.float32, device=x.device)
        rope = rotary(x.shape[1], frequencies)
        c_rope = rotary(c.shape[1], frequencies)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks) + list(self.single_transformer_blocks), x.device, transformer_options)
        for block in self.transformer_blocks:
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x, c
                x, c = block(x, c, t, rope, c_rope, audio_mask, c_mask, transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, malloc_scope="block")
        text_length = c.shape[1]
        x = torch.cat((c, x), dim=1)
        rope = rotary(x.shape[1], frequencies)
        mask = torch.cat((c_mask, audio_mask), dim=1) if audio_mask is not None else None
        for block in self.single_transformer_blocks:
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = block(x, t, rope, mask, transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, malloc_scope="block")
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
        x = x[:, text_length + prompt_length:]
        return -self.proj_out(self.norm_out(x, t)).transpose(1, 2)
