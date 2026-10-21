"""AuK's BigVGAN VAE inference path (Tencent/NVIDIA, MIT).

Weight normalization is folded by the offline converter. The training-only flow
is unused by both upstream reference encoding and waveform decoding.
"""
import torch
from torch import nn
from torch.nn import functional as F

import comfy.ops
import comfy.model_management
import comfy.model_prefetch
from comfy.ldm.minimax.audio_vae import SnakeBeta


class ConvPosition(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, operations, dtype, device):
        super().__init__()
        self.layer = operations.Conv1d(in_channels, out_channels, kernel, stride=stride,
                                       padding=(kernel - 1) // 2, dtype=dtype, device=device)

    def forward(self, x):
        return self.layer(x)


class ResStack(nn.Module):
    def __init__(self, channels, operations, dtype, device):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(
            nn.LeakyReLU(), operations.Conv1d(channels, channels, 3, dilation=2**i, padding=2**i, dtype=dtype, device=device),
            nn.LeakyReLU(), operations.Conv1d(channels, channels, 3, padding=1, dtype=dtype, device=device)) for i in range(6)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, operations, dtype, device):
        super().__init__()
        channels = [12, 24, 48, 96, 192, 384, 768]
        layers = [ConvPosition(1, 12, 3, 1, operations, dtype, device), nn.LeakyReLU(0.2)]
        for i, stride in enumerate([2, 2, 2, 3, 4, 5]):
            layers.extend([ConvPosition(channels[i], channels[i + 1], stride * 2, stride, operations, dtype, device),
                           ResStack(channels[i + 1], operations, dtype, device), nn.LeakyReLU(0.2)])
        layers.append(ConvPosition(768, 128, 3, 1, operations, dtype, device))
        self.generator = nn.Sequential(*layers)

    def forward(self, x):
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.generator), x.device, {})
        for layer in self.generator:
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = layer(x)
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope="block")
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
        return x


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('filter', torch.empty(1, 1, 12))

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (5, 5), mode='replicate')
        weight = comfy.ops.cast_to_input(self.filter, x).expand(channels, -1, -1)
        return (2 * F.conv_transpose1d(x, weight, stride=2, groups=channels))[..., 15:-15]


class LowPass(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('filter', torch.empty(1, 1, 12))

    def forward(self, x):
        channels = x.shape[1]
        weight = comfy.ops.cast_to_input(self.filter, x).expand(channels, -1, -1)
        return F.conv1d(F.pad(x, (11, 0), mode='replicate'), weight, stride=2, groups=channels)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.lowpass = LowPass()

    def forward(self, x):
        return self.lowpass(x)


class Activation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample()
        self.downsample = DownSample()

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock(nn.Module):
    def __init__(self, channels, kernel, operations, dtype, device):
        super().__init__()
        self.kernel = kernel
        self.convs1 = nn.ModuleList([operations.Conv1d(channels, channels, kernel, dilation=d, dtype=dtype, device=device) for d in (1, 3, 5)])
        self.convs2 = nn.ModuleList([operations.Conv1d(channels, channels, kernel, dtype=dtype, device=device) for _ in range(3)])
        self.activations = nn.ModuleList([Activation(channels) for _ in range(6)])

    def forward(self, x):
        for i, (c1, c2, dilation) in enumerate(zip(self.convs1, self.convs2, (1, 3, 5))):
            y = c1(F.pad(self.activations[2 * i](x), (dilation * (self.kernel - 1), 0)))
            y = c2(F.pad(self.activations[2 * i + 1](y), (self.kernel - 1, 0)))
            x = x + y
        return x


class AuKVAE(nn.Module):
    def __init__(self, operations=comfy.ops.disable_weight_init, dtype=None, device=None):
        super().__init__()
        self.register_buffer('global_mean', torch.empty(64, dtype=dtype, device=device))
        self.register_buffer('global_log_std', torch.empty(64, dtype=dtype, device=device))
        self.audio_encoder = Encoder(operations, dtype, device)
        self.conv_pre = operations.Conv1d(64, 1536, 7, padding=3, dtype=dtype, device=device)
        self.ups = nn.ModuleList([nn.ModuleList([operations.ConvTranspose1d(1536 // 2**i, 1536 // 2**(i + 1),
                                      2 * stride, stride=stride, dtype=dtype, device=device)]) for i, stride in enumerate((5, 4, 3, 2, 2, 2))])
        self.resblocks = nn.ModuleList([AMPBlock(1536 // 2**(i + 1), k, operations, dtype, device) for i in range(6) for k in (3, 7, 11)])
        self.activation_post = Activation(24)
        self.conv_post = operations.Conv1d(24, 1, 7, bias=False, dtype=dtype, device=device)

    def get_dynamic_vram__units(self):
        return [*self.audio_encoder.generator, *self.decoder_units()], []

    def encode(self, waveform):
        mean, log_std = self.audio_encoder(waveform).chunk(2, dim=1)
        latent = mean + torch.randn_like(mean) * log_std.exp()
        center = comfy.ops.cast_to_input(self.global_mean, latent)[None, :, None]
        scale = comfy.ops.cast_to_input(self.global_log_std, latent)[None, :, None].sqrt()
        return (latent - center) / scale

    def decoder_units(self):
        units = [self.conv_pre]
        for i, up in enumerate(self.ups):
            units.extend([up, *self.resblocks[3 * i:3 * i + 3]])
        return [*units, self.activation_post, self.conv_post]

    def decode(self, latent):
        center = comfy.ops.cast_to_input(self.global_mean, latent)[None, :, None]
        scale = comfy.ops.cast_to_input(self.global_log_std, latent)[None, :, None].sqrt()
        x = latent * scale + center
        queue = comfy.model_prefetch.make_prefetch_queue(self.decoder_units(), x.device, {})
        def pre():
            nonlocal x
            x = self.conv_pre(x)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, self.conv_pre, x.dtype, core=pre, malloc_scope="block")
        for i, (up, stride) in enumerate(zip(self.ups, (5, 4, 3, 2, 2, 2))):
            comfy.model_management.throw_exception_if_processing_interrupted()
            def upsample():
                nonlocal x
                x = up[0](x)[..., :-stride]
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, up, x.dtype, core=upsample, malloc_scope="block")
            result = None
            for block in self.resblocks[3 * i:3 * i + 3]:
                def residual():
                    nonlocal result
                    value = block(x)
                    result = value if result is None else result + value
                comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=residual, malloc_scope="block")
            x = result / 3
        def activate():
            nonlocal x
            x = self.activation_post(x)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, self.activation_post, x.dtype, core=activate, malloc_scope="block")
        def post():
            nonlocal x
            x = self.conv_post(F.pad(x, (6, 0)))
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, self.conv_post, x.dtype, core=post, malloc_scope="block")
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
        return x.clamp(-1, 1)
