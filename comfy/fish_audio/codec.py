"""Inference-only causal DAC codec, adapted from Fish Speech and Descript DAC."""
import torch
from torch import nn
from torch.nn import functional as F
import comfy.ops
import comfy_kitchen as ck
import comfy.model_management as mm
import comfy.model_prefetch as prefetch
from comfy.text_encoders.llama import RMSNorm
from comfy.ldm.modules.attention import ComfyAttention
from .model import attention, rotary_matrix
from .tower import TowerMLP
from types import SimpleNamespace


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride=1, dilation=1, groups=1, transpose=False, *, dtype, device, operations):
        super().__init__()
        self.transpose, self.stride = transpose, stride
        self.kernel = (kernel - 1) * dilation + 1
        cls = operations.ConvTranspose1d if transpose else operations.Conv1d
        self.conv = cls(inputs, outputs, kernel, stride=stride, dilation=dilation, groups=groups, dtype=dtype, device=device)

    def forward(self, x):
        pad = self.kernel - self.stride
        if self.transpose:
            x = self.conv(x)
            return x[..., :x.shape[-1] - pad] if pad else x
        extra = -x.shape[-1] % self.stride
        return self.conv(F.pad(x, (pad, extra))).contiguous()


class Snake(nn.Module):
    def __init__(self, channels, *, dtype, device, operations):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1, dtype=dtype, device=device))

    def forward(self, x):
        alpha = comfy.ops.cast_to_input(self.alpha, x)
        return x + (alpha + 1e-9).reciprocal() * (alpha * x).sin().square()


class Residual(nn.Module):
    def __init__(self, channels, dilation, **kw):
        super().__init__()
        self.block = nn.Sequential(Snake(channels, **kw), CausalConv(channels, channels, 7, dilation=dilation, **kw),
                                   Snake(channels, **kw), CausalConv(channels, channels, 1, **kw))

    def forward(self, x):
        return x + self.block(x)


class CodecAttention(nn.Module):
    def __init__(self, dim, **kw):
        super().__init__()
        operations = kw['operations']
        args = dict(dtype=kw['dtype'], device=kw['device'], bias=False)
        self.heads, self.dim = dim // 64, dim
        self.wqkv = operations.Linear(dim, 3 * dim, **args)
        self.wo = operations.Linear(dim, dim, **args)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, cos, sin, mask):
        batch, length, _ = x.shape
        q, k, v = self.wqkv(x).split(self.dim, -1)
        q = q.reshape(batch, length, self.heads, 64).transpose(1, 2)
        k = k.reshape(batch, length, self.heads, 64).transpose(1, 2)
        matrix = rotary_matrix(cos.float(), sin.float()).transpose(1, 2)
        q, k = ck.apply_rope(q, k, matrix)
        v = v.view(batch, length, self.heads, 64).transpose(1, 2)
        return self.wo(attention(q, k, v, mask, self.comfy_attention))


class LayerScale(nn.Module):
    def __init__(self, dim, **kw):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, dtype=kw['dtype'], device=kw['device']))

    def forward(self, x):
        return x * comfy.ops.cast_to_input(self.gamma, x)


class CodecBlock(nn.Module):
    def __init__(self, dim, **kw):
        super().__init__()
        self.attention = CodecAttention(dim, **kw)
        self.feed_forward = TowerMLP(SimpleNamespace(dim=dim, intermediate_size=3 * dim), kw['dtype'], kw['device'], kw['operations'])
        self.ffn_norm = RMSNorm(dim, eps=1e-5, dtype=kw['dtype'], device=kw['device'])
        self.attention_norm = RMSNorm(dim, eps=1e-5, dtype=kw['dtype'], device=kw['device'])
        self.attention_layer_scale = LayerScale(dim, **kw)
        self.ffn_layer_scale = LayerScale(dim, **kw)

    def forward(self, x, cos, sin, mask):
        x = x + self.attention_layer_scale(self.attention(self.attention_norm(x), cos, sin, mask))
        return x + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(x)))


class CodecTransformer(nn.Module):
    def __init__(self, dim, depth, window, **kw):
        super().__init__()
        self.window = window
        self.layers = nn.ModuleList(CodecBlock(dim, **kw) for _ in range(depth))
        self.norm = RMSNorm(dim, eps=1e-5, dtype=kw['dtype'], device=kw['device'])

    def forward(self, x):
        x = x.transpose(1, 2)
        length = x.shape[1]
        pos = torch.arange(length, device=x.device)
        phase = pos[:, None] * 10000 ** (-torch.arange(0, 64, 2, device=x.device, dtype=torch.float32) / 64)
        # Upstream codec RoPE tables are rounded to BF16 at construction.
        cos = phase.cos().to(torch.bfloat16)[None, :, None]
        sin = phase.sin().to(torch.bfloat16)[None, :, None]
        visible = (pos[None, :] <= pos[:, None]) & (pos[None, :] >= pos[:, None] - self.window + 1)
        mask = x.new_zeros(visible.shape).masked_fill_(~visible, float('-inf'))
        queue = prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": True})
        for layer in self.layers:
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = layer(x, cos, sin, mask)
            prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='fish_codec')
        prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='fish_codec')
        return self.norm(x).transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, dim, stride, transformer, **kw):
        super().__init__()
        self.block = nn.Sequential(*(Residual(dim // 2, d, **kw) for d in (1, 3, 9)), Snake(dim // 2, **kw),
                                   CausalConv(dim // 2, dim, 2 * stride, stride, **kw),
                                   CodecTransformer(dim, 4, 512, **kw) if transformer else nn.Identity())

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.block = nn.Sequential(CausalConv(1, 64, 7, **kw),
            *(EncoderBlock(128 * 2 ** i, stride, i == 3, **kw) for i, stride in enumerate((2, 4, 8, 8))),
            Snake(1024, **kw), CausalConv(1024, 1024, 3, **kw))

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, inputs, outputs, stride, **kw):
        super().__init__()
        self.block = nn.Sequential(Snake(inputs, **kw), CausalConv(inputs, outputs, 2 * stride, stride, transpose=True, **kw),
                                  *(Residual(outputs, d, **kw) for d in (1, 3, 9)))

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.model = nn.Sequential(CausalConv(1024, 1536, 7, **kw),
            *(DecoderBlock(1536 // 2 ** i, 1536 // 2 ** (i + 1), stride, **kw) for i, stride in enumerate((8, 8, 4, 2))),
            Snake(96, **kw), CausalConv(96, 1, 7, **kw), nn.Tanh())

    def forward(self, x):
        return self.model(x)


class ConvNext(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        args = dict(dtype=kw['dtype'], device=kw['device'])
        self.dwconv = CausalConv(1024, 1024, 7, groups=1024, **kw)
        self.norm = kw['operations'].LayerNorm(1024, eps=1e-6, **args)
        self.pwconv1 = kw['operations'].Linear(1024, 4096, **args)
        self.pwconv2 = kw['operations'].Linear(4096, 1024, **args)
        self.gamma = nn.Parameter(torch.empty(1024, **args))

    def forward(self, x):
        h = self.dwconv(x).transpose(1, 2)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + (h * comfy.ops.cast_to_input(self.gamma, h)).transpose(1, 2)


class VectorQuantizer(nn.Module):
    def __init__(self, size, **kw):
        super().__init__()
        args = dict(dtype=kw['dtype'], device=kw['device'])
        self.in_proj = kw['operations'].Conv1d(1024, 8, 1, **args)
        self.out_proj = kw['operations'].Conv1d(8, 1024, 1, **args)
        self.codebook = kw['operations'].Embedding(size, 8, **args)

    def decode(self, codes):
        return self.out_proj(self.codebook(codes).transpose(1, 2))

    def encode(self, x):
        latent = self.in_proj(x).transpose(1, 2)
        enc = F.normalize(latent.reshape(-1, 8))
        codebook = F.normalize(comfy.ops.cast_to_input(self.codebook.weight, latent))
        distances = enc.square().sum(-1, keepdim=True) - 2 * enc @ codebook.T + codebook.square().sum(-1)[None]
        codes = distances.argmin(-1).view(x.shape[0], -1)
        embedded = self.codebook(codes).transpose(1, 2)
        projected = latent.transpose(1, 2)
        return self.out_proj(projected + (embedded - projected)), codes


class ResidualQuantizer(nn.Module):
    def __init__(self, count, size, **kw):
        super().__init__()
        self.quantizers = nn.ModuleList(VectorQuantizer(size, **kw) for _ in range(count))


class Quantizer(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.semantic_quantizer = ResidualQuantizer(1, 4096, **kw)
        self.quantizer = ResidualQuantizer(9, 1024, **kw)
        self.downsample = nn.Sequential(*(nn.Sequential(CausalConv(1024, 1024, 2, 2, **kw), ConvNext(**kw)) for _ in range(2)))
        self.upsample = nn.Sequential(*(nn.Sequential(CausalConv(1024, 1024, 2, 2, transpose=True, **kw), ConvNext(**kw)) for _ in range(2)))
        self.pre_module = CodecTransformer(1024, 8, 128, **kw)
        self.post_module = CodecTransformer(1024, 8, 128, **kw)

    def encode(self, x):
        x = self.pre_module(self.downsample(x))
        codes = []
        for quantizer in (*self.semantic_quantizer.quantizers, *self.quantizer.quantizers):
            value, ids = quantizer.encode(x)
            x = x - value
            codes.append(ids)
        return torch.stack(codes, 1)

    def decode(self, codes):
        semantic = self.semantic_quantizer.quantizers[0].decode(codes[:, 0].clamp(max=4095))
        residual = 0.
        for i, quantizer in enumerate(self.quantizer.quantizers):
            # Generator residual heads have 4096 entries; DAC has 1024.
            residual = residual + quantizer.decode(codes[:, i + 1].clamp(max=1023))
        return self.upsample(self.post_module(semantic + residual))


class Codec(nn.Module):
    sample_rate = 44100
    frame_length = 2048

    def __init__(self, dtype, device, operations):
        super().__init__()
        kw = dict(dtype=dtype, device=device, operations=operations)
        self.encoder, self.quantizer, self.decoder = Encoder(**kw), Quantizer(**kw), Decoder(**kw)

    def get_dtype(self):
        return self.encoder.block[0].conv.weight.dtype

    def get_dynamic_vram__units(self):
        return ([*self.encoder.block[4].block[5].layers, *self.quantizer.pre_module.layers, *self.quantizer.post_module.layers], [])

    def encode(self, audio):
        return self.quantizer.encode(self.encoder(F.pad(audio, (0, -audio.shape[-1] % self.frame_length))))

    def decode(self, codes):
        return self.decoder(self.quantizer.decode(codes))
