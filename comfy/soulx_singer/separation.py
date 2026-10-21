"""Inference-only MelBandRoformer, adapted from SoulX-Singer/ZFTurbo (MIT)."""

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class RMSNorm(nn.Module):
    def __init__(self, dim, dtype, device):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        self.scale = dim ** 0.5

    def forward(self, x):
        # Roformer clamps the L2 norm, rather than adding epsilon to the variance.
        return F.normalize(x, dim=-1) * self.scale * comfy.ops.cast_to_input(self.gamma, x)


class FeedForward(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim, dtype, device),
                                 operations.Linear(dim, dim * 4, dtype=dtype, device=device), nn.GELU(), nn.Identity(),
                                 operations.Linear(dim * 4, dim, dtype=dtype, device=device), nn.Identity())

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dtype, device, operations):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.comfy_attention = ComfyAttention()
        self.rotary_embed = nn.Module()
        self.rotary_embed.freqs = nn.Parameter(torch.empty(head_dim // 2, dtype=dtype, device=device))
        self.norm = RMSNorm(dim, dtype, device)
        self.to_qkv = operations.Linear(dim, heads * head_dim * 3, bias=False, dtype=dtype, device=device)
        self.to_gates = operations.Linear(dim, heads, dtype=dtype, device=device)
        self.to_out = nn.Sequential(operations.Linear(heads * head_dim, dim, bias=False, dtype=dtype, device=device), nn.Identity())

    def forward(self, x, position):
        x = self.norm(x)
        b, t, _ = x.shape
        q, k, v = self.to_qkv(x).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        phase = torch.outer(position, comfy.ops.cast_to_input(self.rotary_embed.freqs, position))
        cos, sin = phase.cos(), phase.sin()
        rotation = torch.stack((cos, -sin, sin, cos), dim=-1).unflatten(-1, (2, 2))[None, None]
        q, k = apply_rope(q, k, rotation)
        out = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                                  self.heads, skip_reshape=True, preferred_attention=self.comfy_attention)
        gates = self.to_gates(x).sigmoid().repeat_interleave(self.head_dim, dim=-1)
        return self.to_out(out * gates)


class Transformer(nn.Module):
    def __init__(self, dim, heads, head_dim, dtype, device, operations):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, head_dim, dtype, device, operations),
                                                   FeedForward(dim, dtype, device, operations)])])
        self.norm = RMSNorm(dim, dtype, device)

    def forward(self, x, position):
        for attn, ff in self.layers:
            x = x + attn(x, position)
            x = x + ff(x)
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim, widths, dtype, device, operations):
        super().__init__()
        self.widths = widths
        self.to_features = nn.ModuleList([nn.Sequential(RMSNorm(w, dtype, device),
                                                        operations.Linear(w, dim, dtype=dtype, device=device)) for w in widths])

    def forward(self, x):
        return torch.stack([net(band) for net, band in zip(self.to_features, x.split(self.widths, dim=-1))], dim=-2)


class MaskEstimator(nn.Module):
    def __init__(self, dim, widths, dtype, device, operations):
        super().__init__()
        self.to_freqs = nn.ModuleList([
            nn.Sequential(nn.Sequential(operations.Linear(dim, dim * 4, dtype=dtype, device=device), nn.Tanh(),
                                         operations.Linear(dim * 4, dim * 4, dtype=dtype, device=device), nn.Tanh(),
                                         operations.Linear(dim * 4, width * 2, dtype=dtype, device=device)), nn.GLU(dim=-1))
            for width in widths])

    def forward(self, x):
        return torch.cat([net(band) for net, band in zip(self.to_freqs, x.unbind(-2))], dim=-1)


class MelBandRoformer(nn.Module):
    def __init__(self, bands, stems=2, dim=384, depth=6, heads=8, head_dim=64,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.head_dim = head_dim
        self.bands = len(bands)
        indexes = [i * 2 + channel for band in bands for i in band for channel in range(2)]
        counts = [sum(i in band for band in bands) for i in range(1025) for channel in range(2)]
        self.register_buffer("freq_indices", torch.tensor(indexes, dtype=torch.long, device=device))
        self.register_buffer("num_bands_per_freq", torch.tensor(counts, dtype=torch.float32, device=device))
        widths = [len(band) * 4 for band in bands]
        self.band_split = BandSplit(dim, widths, dtype, device, operations)
        self.layers = nn.ModuleList([nn.ModuleList([Transformer(dim, heads, head_dim, dtype, device, operations),
                                                  Transformer(dim, heads, head_dim, dtype, device, operations)]) for _ in range(depth)])
        self.mask_estimators = nn.ModuleList([MaskEstimator(dim, widths, dtype, device, operations) for _ in range(stems)])

    def forward(self, audio, transformer_options={}):
        batch, channels, length = audio.shape
        window = torch.hann_window(2048, device=audio.device, dtype=torch.float32)
        spec = torch.stft(audio.float().reshape(batch * channels, length), 2048, 441, window=window, return_complex=True)
        frames = spec.shape[-1]
        spec = spec.reshape(batch, channels, 1025, frames).permute(0, 2, 1, 3).reshape(batch, 2050, frames)
        x = torch.view_as_real(spec[:, self.freq_indices]).permute(0, 2, 1, 3).flatten(-2).to(self.dtype)
        x = self.band_split(x)
        time_position = torch.arange(frames, device=x.device, dtype=torch.float32)
        freq_position = torch.arange(self.bands, device=x.device, dtype=torch.float32)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(x.device)
        for block in self.layers:
            comfy.model_management.throw_exception_if_processing_interrupted()
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="separation")
            hidden = block[0](x.transpose(1, 2).reshape(batch * self.bands, frames, -1), time_position)
            hidden = hidden.reshape(batch, self.bands, frames, -1).transpose(1, 2)
            result = block[1](hidden.reshape(batch * frames, self.bands, -1), freq_position).reshape(batch, frames, self.bands, -1)
            if torch.is_grad_enabled():
                x = result
            else:
                x.copy_(result)
            del result, hidden
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="separation")
        comfy.model_prefetch.malloc_graph_end()
        masks = torch.stack([net(x) for net in self.mask_estimators], dim=1)
        masks = masks.reshape(batch, len(self.mask_estimators), frames, -1, 2).permute(0, 1, 3, 2, 4).float().contiguous()
        masks = torch.view_as_complex(masks)
        summed = torch.zeros(batch, len(self.mask_estimators), 2050, frames, device=x.device, dtype=spec.dtype)
        summed.scatter_add_(2, self.freq_indices[None, None, :, None].expand_as(masks), masks)
        separated = spec[:, None] * summed / self.num_bands_per_freq[None, None, :, None]
        separated = separated.reshape(batch, len(self.mask_estimators), 1025, channels, frames).transpose(2, 3)
        return torch.istft(separated.flatten(0, 2), 2048, 441, window=window, length=length).reshape(batch, len(self.mask_estimators), channels, length)
