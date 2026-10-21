"""Applio HuBERT/ContentVec, using the native Wav2Vec2 inference layers."""

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
from comfy.audio_encoders import wav2vec2
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


class ContentAttention(wav2vec2.Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, mask=None):
        q, k, v = [AttentionTensorContainer(p(x)) for p in (self.q_proj, self.k_proj, self.v_proj)]
        return self.out_proj(optimized_attention(q, k, v, self.num_heads, mask=mask, preferred_attention=self.comfy_attention))


class ContentEncoder(nn.Module):
    def __init__(self, dtype, device, operations):
        super().__init__()
        args = dict(dtype=dtype, device=device, operations=operations)
        self.feature_extractor = wav2vec2.ConvFeatureEncoder(512, conv_bias=False, conv_norm=False, **args)
        self.feature_projection = wav2vec2.FeatureProjection(512, 768, **args)
        self.encoder = nn.Module()
        self.encoder.pos_conv_embed = nn.Module()
        self.encoder.pos_conv_embed.conv = operations.Conv1d(768, 768, 128, padding=64, groups=16, dtype=dtype, device=device)
        self.encoder.layer_norm = operations.LayerNorm(768, eps=1e-5, dtype=dtype, device=device)
        self.encoder.layers = nn.ModuleList()
        for _ in range(12):
            block = wav2vec2.TransformerEncoderLayer(768, 12, do_stable_layer_norm=False, **args)
            block.attention = ContentAttention(768, 12, **args)
            self.encoder.layers.append(block)
        self.final_proj = operations.Linear(768, 256, dtype=dtype, device=device)

    def forward(self, audio, version, transformer_options):
        x = self.feature_projection(self.feature_extractor(audio))
        position = self.encoder.pos_conv_embed.conv(x.transpose(1, 2))[..., :-1]
        x = self.encoder.layer_norm(x + F.gelu(position).transpose(1, 2))
        queue = prefetch.make_prefetch_queue(list(self.encoder.layers), x.device, transformer_options)
        with prefetch.malloc_graph_scope(x.device):
            for block in self.encoder.layers:
                mm.throw_exception_if_processing_interrupted()
                prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="rvc_content")
                result = block(x)
                if torch.is_grad_enabled():
                    x = result
                else:
                    x.copy_(result)
                del result
            prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="rvc_content")
        return self.final_proj(x) if version == "v1" else x
