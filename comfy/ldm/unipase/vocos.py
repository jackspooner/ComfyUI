"""UniPASE Adapter and DWO vocoder (MIT/Apache-2.0; see licenses)."""
import torch
from torch import nn
from torch.nn import functional as F
import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy.ops
from comfy.ldm.soulx_singer.vocoder import ConvNeXt
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class ResnetBlock(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        args = dict(dtype=dtype, device=device)
        self.norm1 = operations.GroupNorm(32, dim, eps=1e-6, **args)
        self.norm2 = operations.GroupNorm(32, dim, eps=1e-6, **args)
        self.conv1 = operations.Conv1d(dim, dim, 3, padding=1, **args)
        self.conv2 = operations.Conv1d(dim, dim, 3, padding=1, **args)

    def forward(self, x):
        return x + self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))


class AttnBlock(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        args = dict(dtype=dtype, device=device)
        self.norm = operations.GroupNorm(32, dim, eps=1e-6, **args)
        for name in ("q", "k", "v", "proj_out"):
            setattr(self, name, operations.Conv1d(dim, dim, 1, **args))
        self.comfy_attention = ComfyAttention()

    def forward(self, x):
        y = self.norm(x)
        q, k, v = [AttentionTensorContainer(proj(y).transpose(1, 2)) for proj in (self.q, self.k, self.v)]
        y = optimized_attention(q, k, v, 1, preferred_attention=self.comfy_attention, low_precision_attention=False).transpose(1, 2)
        return x + self.proj_out(y)


class Backbone(nn.Module):
    def __init__(self, cfg, dtype, device, operations):
        super().__init__()
        dim = cfg["dim"]
        args = dict(dtype=dtype, device=device)
        self.embed = operations.Conv1d(cfg["input_channels"], dim, 7, padding=3, **args)
        self.norm = operations.LayerNorm(dim, eps=1e-6, **args)
        self.final_layer_norm = operations.LayerNorm(dim, eps=1e-6, **args)
        self.convnext = nn.ModuleList([ConvNeXt(dim, cfg["intermediate_dim"], dtype, device, operations) for _ in range(cfg["num_layers"])])
        half = cfg.get("num_res", 4) // 2
        self.pos_net = nn.Sequential(*([ResnetBlock(dim, dtype, device, operations) for _ in range(half)] +
            [AttnBlock(dim, dtype, device, operations) for _ in range(cfg.get("num_attn", 1))] +
            [ResnetBlock(dim, dtype, device, operations) for _ in range(half)] +
            [operations.GroupNorm(32, dim, eps=1e-6, **args)]))

    def forward(self, x, transformer_options):
        x = self.norm(self.pos_net(self.embed(x)).transpose(1, 2)).transpose(1, 2)
        queue = prefetch.make_prefetch_queue(list(self.convnext), x.device, transformer_options)
        with prefetch.malloc_graph_scope(x.device):
            for block in self.convnext:
                mm.throw_exception_if_processing_interrupted()
                prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="unipase_vocos")
                result = block(x)
                if torch.is_grad_enabled():
                    x = result
                else:
                    x.copy_(result)
                del result
            prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="unipase_vocos")
        return self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)


class Adapter(nn.Module):
    def __init__(self, cfg, dtype=None, device=None, operations=None):
        super().__init__()
        args = dict(dtype=dtype, device=device)
        self.proj = operations.Linear(cfg["input_channels"], cfg["input_channels"], **args)
        self.decoder = Backbone(cfg, dtype, device, operations)
        self.head = operations.Linear(cfg["dim"], cfg["output_channels"], **args)

    def forward(self, acoustic, phonetic, transformer_options={}):
        x = self.decoder((self.proj(phonetic) + acoustic).transpose(1, 2), transformer_options)
        return self.head(x.transpose(1, 2)).transpose(1, 2)


class Vocoder(nn.Module):
    def __init__(self, cfg, dtype=None, device=None, operations=None):
        super().__init__()
        self.decoder = Backbone(cfg, dtype, device, operations)
        self.head = nn.Module()
        self.head.out = operations.Linear(cfg["dim"], cfg["n_fft"] + 2, dtype=dtype, device=device)
        self.head.istft = nn.Module()
        self.head.istft.register_buffer("window", torch.empty(cfg["n_fft"], dtype=dtype, device=device))
        self.n_fft, self.hop = cfg["n_fft"], cfg["hop_length"]

    def forward(self, features, transformer_options={}):
        x = self.head.out(self.decoder(features, transformer_options).transpose(1, 2)).transpose(1, 2)
        mag, phase = x.chunk(2, dim=1)
        spec = mag.clamp(-20, 5).exp() * torch.complex(phase.cos(), phase.sin())
        window = comfy.ops.cast_to_input(self.head.istft.window, spec.real)
        frames = torch.fft.irfft(spec, self.n_fft, dim=1) * window[None, :, None]
        length = (frames.shape[-1] - 1) * self.hop + self.n_fft
        pad = (self.n_fft - self.hop) // 2
        audio = F.fold(frames, (1, length), (1, self.n_fft), stride=(1, self.hop))[:, 0, 0, pad:-pad]
        envelope = F.fold(window.square()[None, :, None].expand(1, -1, frames.shape[-1]), (1, length), (1, self.n_fft), stride=(1, self.hop))[0, 0, 0, pad:-pad]
        return audio / envelope
