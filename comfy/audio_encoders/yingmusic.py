"""YingMusic melody encoder, adapted from ASLP-lab/YingMusic-Singer-Plus (CC BY 4.0)."""

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_prefetch
import comfy.ops
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class Attention(nn.Module):
    def __init__(self, dim, heads, dtype, device, operations):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_kv = operations.Linear(dim, dim * 2, bias=False, dtype=dtype, device=device)
        self.to_out = nn.Sequential(operations.Linear(dim, dim, dtype=dtype, device=device))

    def forward(self, x):
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        hidden = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                                     self.heads, preferred_attention=self.comfy_attention)
        return self.to_out(hidden)


class Convolution(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.pointwise_conv1 = operations.Conv1d(dim, dim * 2, 1, dtype=dtype, device=device)
        self.depthwise_conv = operations.Conv1d(dim, dim, 31, padding=15, groups=dim, dtype=dtype, device=device)
        self.norm = operations.BatchNorm2d(dim, dtype=dtype, device=device)
        self.pointwise_conv2 = operations.Conv1d(dim, dim, 1, dtype=dtype, device=device)

    def forward(self, x):
        hidden = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        hidden = self.norm(self.depthwise_conv(hidden).unsqueeze(-1)).squeeze(-1)
        return self.pointwise_conv2(F.silu(hidden)).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.ln1 = operations.Linear(dim, dim * 4, dtype=dtype, device=device)
        self.ln2 = operations.Linear(dim * 4, dim, dtype=dtype, device=device)

    def forward(self, x):
        return self.ln2(F.silu(self.ln1(x)))


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, dtype, device, operations):
        super().__init__()
        self.ffn1 = FeedForward(dim, dtype, device, operations)
        self.ffn2 = FeedForward(dim, dtype, device, operations)
        self.att = Attention(dim, heads, dtype, device, operations)
        self.conv = Convolution(dim, dtype, device, operations)
        self.norm1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm4 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.norm5 = operations.LayerNorm(dim, dtype=dtype, device=device)

    def forward(self, x):
        x = x + self.ffn1(self.norm1(x)) * .5
        x = x + self.att(self.norm2(x))
        x = x + self.conv(self.norm3(x))
        x = x + self.ffn2(self.norm4(x)) * .5
        return self.norm5(x)


class GatedConformer(nn.Module):
    def __init__(self, dim, heads, dtype, device, operations):
        super().__init__()
        self.att1 = ConformerBlock(dim, heads, dtype, device, operations)
        self.att2 = ConformerBlock(dim, heads, dtype, device, operations)
        self.glu1 = nn.Sequential(operations.Linear(dim, dim * 2, dtype=dtype, device=device), nn.GLU(dim=-1))
        self.glu2 = nn.Sequential(operations.Linear(dim, dim * 2, dtype=dtype, device=device), nn.GLU(dim=-1))

    def forward(self, midi, boundary):
        midi, boundary = self.att1(midi), self.att2(boundary)
        return midi + self.glu2(boundary), boundary + self.glu1(midi)


class MelodyEncoder(nn.Module):
    def __init__(self, dim=512, depth=8, heads=8, dtype=None, device=None, operations=None):
        super().__init__()
        self.register_buffer("mel_basis", torch.empty(80, 1025, dtype=dtype, device=device))
        self.inln = operations.Linear(80, dim, dtype=dtype, device=device)
        self.inln1 = operations.Linear(80, dim, dtype=dtype, device=device)
        self.outln = operations.Linear(dim, 128, dtype=dtype, device=device)
        self.cutheard = operations.Linear(dim, 1, dtype=dtype, device=device)
        self.cf_lay = nn.ModuleList([GatedConformer(dim, heads, dtype, device, operations) for _ in range(depth)])
        self.att1 = ConformerBlock(dim, heads, dtype, device, operations)
        self.att2 = ConformerBlock(dim, heads, dtype, device, operations)

    def forward(self, x, transformer_options={}):
        midi, boundary = self.inln(x), self.inln1(x)
        prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.cf_lay), x.device, transformer_options)
        for block in self.cf_lay:
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, block, x.dtype)
            midi, boundary = block(midi, boundary)
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
        return self.outln(self.att1(midi)), self.cutheard(self.att2(boundary))


def prepare_melody(logits, frames, reference_frames):
    melody = F.interpolate(logits.transpose(1, 2), size=frames, mode="linear", align_corners=False).transpose(1, 2).sigmoid()
    # Singer Plus uses deterministic one-frame-in-ten dropout, including at inference.
    melody[:, ::10] = 0
    melody[:, :reference_frames] = 0
    return melody


def spectrogram(audio, mel_basis):
    mono = audio.mean(dim=1)
    window = torch.hann_window(2048, device=mono.device, dtype=mono.dtype)
    spectrum = torch.stft(mono, n_fft=2048, hop_length=512, win_length=2048, window=window, return_complex=True)
    magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square())
    return torch.matmul(comfy.ops.cast_to_input(mel_basis, magnitude), magnitude).clamp(min=1e-5).log()
