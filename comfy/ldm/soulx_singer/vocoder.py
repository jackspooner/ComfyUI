"""SoulX-Singer's released Vocos decoder, with native ComfyUI operations."""

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops


class ConvNeXt(nn.Module):
    def __init__(self, dim, intermediate, dtype, device, operations):
        super().__init__()
        self.dwconv = operations.Conv1d(dim, dim, 7, padding=3, groups=dim, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.pwconv1 = operations.Linear(dim, intermediate, dtype=dtype, device=device)
        self.pwconv2 = operations.Linear(intermediate, dim, dtype=dtype, device=device)
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        y = self.norm(self.dwconv(x).transpose(1, 2))
        y = self.pwconv2(F.gelu(self.pwconv1(y)))
        return x + (y * comfy.ops.cast_to_input(self.gamma, y)).transpose(1, 2)


class Vocos(nn.Module):
    def __init__(self, dim=1024, intermediate=4096, depth=30, n_fft=1920, hop=480,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.backbone = nn.Module()
        self.backbone.embed = operations.Conv1d(128, dim, 7, padding=3, dtype=dtype, device=device)
        self.backbone.norm = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.backbone.convnext = nn.ModuleList([ConvNeXt(dim, intermediate, dtype, device, operations) for _ in range(depth)])
        self.backbone.final_layer_norm = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.head = nn.Module()
        self.head.out = operations.Linear(dim, n_fft + 2, dtype=dtype, device=device)
        self.head.istft = nn.Module()
        self.head.istft.register_buffer("window", torch.empty(n_fft, dtype=torch.float32, device=device))

    def forward(self, mel, transformer_options={}):
        x = self.backbone.norm(self.backbone.embed(mel).transpose(1, 2)).transpose(1, 2)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.backbone.convnext), x.device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(x.device)
        for block in self.backbone.convnext:
            comfy.model_management.throw_exception_if_processing_interrupted()
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="vocos")
            result = block(x)
            if torch.is_grad_enabled():
                x = result
            else:
                x.copy_(result)
            del result
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="vocos")
        comfy.model_prefetch.malloc_graph_end()
        x = self.head.out(self.backbone.final_layer_norm(x.transpose(1, 2))).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        # FFT and overlap-add require float32 for the non-power-of-two 1920 window.
        spec = mag.exp().clamp(max=100).float() * torch.complex(phase.cos().float(), phase.sin().float())
        window = comfy.ops.cast_to_input(self.head.istft.window, spec.real)
        frames = torch.fft.irfft(spec, self.n_fft, dim=1) * window[None, :, None]
        length = (frames.shape[-1] - 1) * self.hop + self.n_fft
        pad = (self.n_fft - self.hop) // 2
        audio = F.fold(frames, (1, length), (1, self.n_fft), stride=(1, self.hop))[:, 0, 0, pad:-pad]
        envelope = F.fold(window.square()[None, :, None].expand(1, -1, frames.shape[-1]),
                          (1, length), (1, self.n_fft), stride=(1, self.hop))[0, 0, 0, pad:-pad]
        return (audio / envelope)[:, None]
