"""Inference-only RVC HiFi-GAN, adapted from IAHispano/Applio (MIT).

Weight normalization is folded offline. Relative key AND value attention needs
position-dependent values, which the shared optimized-attention contract cannot
represent; that small encoder retains the exact relative-attention math.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy.ops


def conv(operations, cin, cout, kernel, **kwargs):
    return operations.Conv1d(cin, cout, kernel, padding=(kernel - 1) // 2, **kwargs)


class RelativeAttention(nn.Module):
    def __init__(self, channels, heads, operations):
        super().__init__()
        self.heads = heads
        self.dim = channels // heads
        for name in ("conv_q", "conv_k", "conv_v", "conv_o"):
            setattr(self, name, conv(operations, channels, channels, 1))
        self.emb_rel_k = nn.Parameter(torch.empty(1, 21, self.dim))
        self.emb_rel_v = nn.Parameter(torch.empty(1, 21, self.dim))

    def relative(self, parameter, x, length):
        embedding = comfy.ops.cast_to_input(parameter, x)
        embedding = F.pad(embedding, (0, 0, max(length - 11, 0), max(length - 11, 0)))
        start = max(11 - length, 0)
        return embedding[:, start:start + 2 * length - 1]

    def forward(self, x):
        b, channels, length = x.shape
        q, k, v = [p(x).reshape(b, self.heads, self.dim, length).transpose(2, 3)
                   for p in (self.conv_q, self.conv_k, self.conv_v)]
        q = q / math.sqrt(self.dim)
        relative = q @ self.relative(self.emb_rel_k, q, length).unsqueeze(0).transpose(-2, -1)
        relative = F.pad(relative, (0, 1)).reshape(b, self.heads, length * 2 * length)
        relative = F.pad(relative, (0, length - 1)).reshape(b, self.heads, length + 1, 2 * length - 1)
        weights = (q @ k.transpose(-2, -1) + relative[:, :, :length, length - 1:]).softmax(-1)
        relative_weights = F.pad(weights, (0, length - 1)).reshape(b, self.heads, -1)
        relative_weights = F.pad(relative_weights, (length, 0)).reshape(b, self.heads, length, 2 * length)[..., 1:]
        out = weights @ v + relative_weights @ self.relative(self.emb_rel_v, v, length).unsqueeze(0)
        return self.conv_o(out.transpose(2, 3).reshape(b, channels, length))


class ChannelNorm(nn.Module):
    def __init__(self, channels, operations):
        super().__init__()
        self.norm = operations.LayerNorm(channels, eps=1e-5)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class FFN(nn.Module):
    def __init__(self, channels, filters, kernel, operations):
        super().__init__()
        self.conv_1 = conv(operations, channels, filters, kernel)
        self.conv_2 = conv(operations, filters, channels, kernel)

    def forward(self, x):
        return self.conv_2(F.relu(self.conv_1(x)))


class Encoder(nn.Module):
    def __init__(self, channels, filters, heads, layers, kernel, operations):
        super().__init__()
        self.attn_layers = nn.ModuleList([RelativeAttention(channels, heads, operations) for _ in range(layers)])
        self.norm_layers_1 = nn.ModuleList([ChannelNorm(channels, operations) for _ in range(layers)])
        self.ffn_layers = nn.ModuleList([FFN(channels, filters, kernel, operations) for _ in range(layers)])
        self.norm_layers_2 = nn.ModuleList([ChannelNorm(channels, operations) for _ in range(layers)])

    def forward(self, x, transformer_options):
        blocks = [module for stage in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2) for module in stage]
        queue = prefetch.make_prefetch_queue(blocks, x.device, transformer_options)
        with prefetch.malloc_graph_scope(x.device):
            for attn, norm1, ffn, norm2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
                mm.throw_exception_if_processing_interrupted()
                prefetch.prefetch_queue_pop(queue, x.device, attn, x.dtype, malloc_scope="rvc_encoder")
                residual = x + attn(x)
                prefetch.prefetch_queue_pop(queue, x.device, norm1, x.dtype, malloc_scope="rvc_encoder")
                x = norm1(residual)
                prefetch.prefetch_queue_pop(queue, x.device, ffn, x.dtype, malloc_scope="rvc_encoder")
                residual = x + ffn(x)
                prefetch.prefetch_queue_pop(queue, x.device, norm2, x.dtype, malloc_scope="rvc_encoder")
                x = norm2(residual)
            prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="rvc_encoder")
        return x


class TextEncoder(nn.Module):
    def __init__(self, config, version, use_f0, operations):
        super().__init__()
        _, _, inter, hidden, filters, heads, layers, kernel = config[:8]
        self.hidden = hidden
        self.emb_phone = operations.Linear(256 if version == "v1" else 768, hidden)
        self.emb_pitch = operations.Embedding(256, hidden) if use_f0 else None
        self.encoder = Encoder(hidden, filters, heads, layers, kernel, operations)
        self.proj = conv(operations, hidden, inter * 2, 1)

    def forward(self, phone, pitch, transformer_options):
        x = self.emb_phone(phone)
        if self.emb_pitch is not None:
            x = x + self.emb_pitch(pitch, out_dtype=x.dtype)
        x = F.leaky_relu(x * math.sqrt(self.hidden), .1).transpose(1, 2)
        return self.proj(self.encoder(x, transformer_options)).chunk(2, dim=1)


class WaveNet(nn.Module):
    def __init__(self, hidden, gin, operations):
        super().__init__()
        self.hidden = hidden
        self.cond_layer = conv(operations, gin, 6 * hidden, 1)
        self.in_layers = nn.ModuleList([conv(operations, hidden, 2 * hidden, 5) for _ in range(3)])
        self.res_skip_layers = nn.ModuleList([conv(operations, hidden, hidden if i == 2 else 2 * hidden, 1) for i in range(3)])

    def forward(self, x, g):
        g = self.cond_layer(g)
        output = torch.zeros_like(x)
        for i, (layer, skip) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            a, b = (layer(x) + g[:, i * 2 * self.hidden:(i + 1) * 2 * self.hidden]).chunk(2, 1)
            result = skip(a.tanh() * b.sigmoid())
            if i < 2:
                residual, result = result.chunk(2, 1)
                x = x + residual
            output = output + result
        return output


class Coupling(nn.Module):
    def __init__(self, inter, hidden, gin, operations):
        super().__init__()
        self.pre = conv(operations, inter // 2, hidden, 1)
        self.enc = WaveNet(hidden, gin, operations)
        self.post = conv(operations, hidden, inter // 2, 1)

    def forward(self, x, g):
        a, b = x.chunk(2, 1)
        return torch.cat((a, b - self.post(self.enc(self.pre(a), g))), dim=1)


class Flow(nn.Module):
    def __init__(self, inter, hidden, gin, operations):
        super().__init__()
        self.flows = nn.ModuleList([Coupling(inter, hidden, gin, operations) if i % 2 == 0 else nn.Identity() for i in range(8)])

    def forward(self, x, g, transformer_options):
        blocks = [self.flows[i] for i in range(6, -1, -2)]
        queue = prefetch.make_prefetch_queue(blocks, x.device, transformer_options)
        with prefetch.malloc_graph_scope(x.device):
            for block in blocks:
                mm.throw_exception_if_processing_interrupted()
                prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="rvc_flow")
                x = block(x.flip(1), g)
            prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="rvc_flow")
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel, dilations, operations):
        super().__init__()
        self.convs1 = nn.ModuleList([operations.Conv1d(channels, channels, kernel, dilation=d, padding=(kernel - 1) * d // 2) for d in dilations])
        self.convs2 = nn.ModuleList([conv(operations, channels, channels, kernel) for _ in dilations])

    def forward(self, x):
        for a, b in zip(self.convs1, self.convs2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, .1)), .1))
        return x


class Source(nn.Module):
    def __init__(self, rate, operations):
        super().__init__()
        self.rate = rate
        self.l_linear = operations.Linear(1, 1)

    def forward(self, f0, factor, generator):
        f0 = f0.unsqueeze(-1)
        grid = torch.arange(1, factor + 1, device=f0.device, dtype=f0.dtype)
        phase = f0 / self.rate * grid
        remainder = (phase[:, :-1, -1:] + .5).fmod(1) - .5
        phase = phase + F.pad(remainder.cumsum(1).fmod(1), (0, 0, 1, 0))
        sine = (phase.reshape(f0.shape[0], -1, 1) * (2 * math.pi)).sin() * .1
        # Upstream consumes a zero fundamental phase draw before source noise.
        torch.rand(1, 1, 1, device=f0.device, generator=generator)
        voiced = F.interpolate((f0 > 0).to(f0.dtype).transpose(1, 2), scale_factor=factor).transpose(1, 2)
        noise = torch.randn(sine.shape, device=sine.device, dtype=sine.dtype, generator=generator)
        source = sine * voiced + noise * (voiced * .003 + (1 - voiced) * (.1 / 3))
        return self.l_linear(source).tanh().transpose(1, 2)


class Generator(nn.Module):
    def __init__(self, config, use_f0, operations):
        super().__init__()
        inter, kernels, dilations, rates, initial, upkernels, _, gin, rate = config[2], *config[10:]
        self.use_f0 = use_f0
        self.factor = math.prod(rates)
        self.num_kernels = len(kernels)
        self.conv_pre = conv(operations, inter, initial, 7)
        self.cond = conv(operations, gin, initial, 1)
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        if use_f0:
            self.m_source = Source(rate, operations)
        for i, (u, k) in enumerate(zip(rates, upkernels)):
            channels = initial // 2 ** (i + 1)
            padding = ((k - u) // 2) if not use_f0 or u % 2 == 0 else u // 2 + u % 2
            self.ups.append(operations.ConvTranspose1d(initial // 2 ** i, channels, k, u, padding=padding, output_padding=u % 2 if use_f0 else 0))
            if use_f0:
                stride = math.prod(rates[i + 1:])
                nk = 1 if stride == 1 else stride * 2 - stride % 2
                self.noise_convs.append(operations.Conv1d(1, channels, nk, stride=stride, padding=(nk - stride) // 2))
            self.resblocks.extend(ResBlock(channels, k, d, operations) for k, d in zip(kernels, dilations))
        self.conv_post = conv(operations, channels, 1, 7, bias=False)

    def forward(self, x, f0, g, generator, transformer_options):
        source = self.m_source(f0, self.factor, generator) if self.use_f0 else None
        x = self.conv_pre(x) + self.cond(g)
        stages = []
        for i, up in enumerate(self.ups):
            stages.append([up] + ([self.noise_convs[i]] if self.use_f0 else []) + list(self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]))
        queue = prefetch.make_prefetch_queue([block for stage in stages for block in stage], x.device, transformer_options)
        with prefetch.malloc_graph_scope(x.device):
            for i, up in enumerate(self.ups):
                mm.throw_exception_if_processing_interrupted()
                prefetch.prefetch_queue_pop(queue, x.device, up, x.dtype, malloc_scope="rvc_vocoder")
                x = up(F.leaky_relu(x, .1))
                if source is not None:
                    prefetch.prefetch_queue_pop(queue, x.device, self.noise_convs[i], x.dtype, malloc_scope="rvc_vocoder")
                    x = x + self.noise_convs[i](source)
                result = 0
                for block in self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]:
                    prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="rvc_vocoder")
                    result = result + block(x)
                x = result / self.num_kernels
            prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="rvc_vocoder")
        return self.conv_post(F.leaky_relu(x, .01)).tanh()


class Synthesizer(nn.Module):
    def __init__(self, config, version, use_f0, operations):
        super().__init__()
        self.enc_p = TextEncoder(config, version, use_f0, operations)
        self.emb_g = operations.Embedding(config[-3], config[-2])
        self.flow = Flow(config[2], config[3], config[-2], operations)
        self.dec = Generator(config, use_f0, operations)

    def forward(self, features, pitch, f0, speaker, generator, transformer_options):
        mean, logs = self.enc_p(features, pitch, transformer_options)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        z = mean + logs.exp() * noise * .66666
        g = self.emb_g(speaker, out_dtype=z.dtype).unsqueeze(-1)
        return self.dec(self.flow(z, g, transformer_options), f0, g, generator, transformer_options)
