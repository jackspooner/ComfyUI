"""DeWavLM-Omni inference, adapted from UniPASE/Microsoft WavLM (MIT)."""
import math
import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy.ops
from comfy.audio_encoders.wav2vec2 import ConvFeatureEncoder
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class Attention(nn.Module):
    def __init__(self, dim, heads, buckets, distance, relative_bias, dtype, device, operations):
        super().__init__()
        self.heads, self.buckets, self.distance = heads, buckets, distance
        args = dict(dtype=dtype, device=device)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, operations.Linear(dim, dim, **args))
        if relative_bias:
            self.relative_attention_bias = operations.Embedding(buckets, heads, **args)
        self.grep_linear = operations.Linear(dim // heads, 8, **args)
        self.grep_a = nn.Parameter(torch.empty(1, heads, 1, 1, **args))
        self.comfy_attention = ComfyAttention()

    def position_bias(self, x):
        position = torch.arange(x.shape[1], device=x.device)
        relative = position[None, :] - position[:, None]
        half = self.buckets // 2
        exact = half // 2
        distance = relative.abs()
        logarithmic = exact + (torch.log(distance.float().clamp(min=1) / exact) / math.log(self.distance / exact) * (half - exact)).long()
        buckets = (relative > 0).long() * half + torch.where(distance < exact, distance, logarithmic.clamp(max=half - 1))
        return self.relative_attention_bias(buckets).permute(2, 0, 1)[None]

    def forward(self, x, bias):
        b, t, dim = x.shape
        gate = self.grep_linear(x.reshape(b, t, self.heads, dim // self.heads).transpose(1, 2))
        a, c = gate.reshape(b, self.heads, t, 2, 4).sum(-1).sigmoid().chunk(2, dim=-1)
        mask = (a * (c * comfy.ops.cast_to_input(self.grep_a, a) - 1) + 2) * bias
        q, k, v = [AttentionTensorContainer(proj(x)) for proj in (self.q_proj, self.k_proj, self.v_proj)]
        return self.out_proj(optimized_attention(q, k, v, self.heads, mask=mask, preferred_attention=self.comfy_attention))


class Block(nn.Module):
    def __init__(self, cfg, first, dtype, device, operations):
        super().__init__()
        dim = cfg["encoder_embed_dim"]
        args = dict(dtype=dtype, device=device)
        self.self_attn = Attention(dim, cfg["encoder_attention_heads"], cfg["num_buckets"], cfg["max_distance"], first, dtype, device, operations)
        self.self_attn_layer_norm = operations.LayerNorm(dim, **args)
        self.final_layer_norm = operations.LayerNorm(dim, **args)
        self.fc1 = operations.Linear(dim, cfg["encoder_ffn_embed_dim"], **args)
        self.fc2 = operations.Linear(cfg["encoder_ffn_embed_dim"], dim, **args)
        self.norm_first = cfg["layer_norm_first"]

    def forward(self, x, bias):
        if self.norm_first:
            x = x + self.self_attn(self.self_attn_layer_norm(x), bias)
            return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        x = self.self_attn_layer_norm(x + self.self_attn(x, bias))
        return self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))


class DeWavLM(nn.Module):
    def __init__(self, cfg, dtype=None, device=None, operations=None):
        super().__init__()
        args = dict(dtype=dtype, device=device)
        dim = cfg["encoder_embed_dim"]
        self.feature_extractor = ConvFeatureEncoder(512, conv_bias=cfg["conv_bias"], conv_norm=cfg["extractor_mode"] == "layer_norm", operations=operations, **args)
        if not cfg["conv_bias"]:
            self.feature_extractor.conv_layers[0].conv.register_parameter("bias", None)
        self.layer_norm = operations.LayerNorm(512, **args)
        self.post_extract_proj = operations.Linear(512, dim, **args)
        self.mask_emb = nn.Parameter(torch.empty(dim, **args))
        self.encoder = nn.Module()
        self.encoder.pos_conv = nn.Sequential(operations.Conv1d(dim, dim, cfg["conv_pos"], padding=cfg["conv_pos"] // 2, groups=cfg["conv_pos_groups"], **args))
        self.encoder.layer_norm = operations.LayerNorm(dim, **args)
        self.encoder.layers = nn.ModuleList([Block(cfg, i == 0, dtype, device, operations) for i in range(cfg["encoder_layers"])])
        self.norm_first = cfg["layer_norm_first"]

    def forward(self, audio, enable_plc=True, transformer_options={}):
        # Upstream rounds to 320*k+80, including cropping the final partial frame.
        audio = F.pad(audio, (0, max(400, audio.shape[-1] // 320 * 320 + 80) - audio.shape[-1]))
        x = self.post_extract_proj(self.layer_norm(self.feature_extractor(audio)))
        if enable_plc:
            packets = audio[:, :x.shape[1] * 320].reshape(audio.shape[0], x.shape[1], 320)
            lost = (packets.abs() < 1e-7).float().mean(-1) >= .99
            x = torch.where(lost[..., None], comfy.ops.cast_to_input(self.mask_emb, x), x)
        position = F.gelu(self.encoder.pos_conv(x.transpose(1, 2))[..., :-1]).transpose(1, 2)
        x = x + position
        if not self.norm_first:
            x = self.encoder.layer_norm(x)
        bias = self.encoder.layers[0].self_attn.position_bias(x)
        queue = prefetch.make_prefetch_queue(list(self.encoder.layers), x.device, transformer_options)
        first = None
        with prefetch.malloc_graph_scope(x.device):
            for i, block in enumerate(self.encoder.layers):
                mm.throw_exception_if_processing_interrupted()
                prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="unipase_wavlm")
                x = block(x, bias)
                if i == 0:
                    first = x
            prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="unipase_wavlm")
        return F.layer_norm(first, first.shape[1:], eps=1e-6), F.layer_norm(x, x.shape[1:], eps=1e-6)
