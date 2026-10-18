"""Native SoulX-Singer SVS/SVC (Soul-AILab, Apache-2.0; Amphion flow model, MIT)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.quant_ops
import comfy.rmsnorm
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.yingmusic.model import ConvNeXtBlock
from comfy.text_encoders.llama import precompute_freqs_cis


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.to_weight = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x, time):
        return comfy.rmsnorm.rms_norm(x, eps=1e-6) * self.to_weight(time)[:, None]


class Attention(nn.Module):
    def __init__(self, dim, heads, dtype, device, operations):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()
        self.q_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.k_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x, rope, transformer_options):
        b, t, dim = x.shape
        q, k, v = [p(x).reshape(b, t, self.heads, dim // self.heads).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj)]
        # Llama pairs the two halves of each head; Roformer's adjacent layout differs.
        q, k = comfy.quant_ops.ck.apply_rope_split_half(q, k, rope)
        out = optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                                  self.heads, skip_reshape=True, preferred_attention=self.comfy_attention,
                                  transformer_options=transformer_options)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim, dtype, device, operations):
        super().__init__()
        self.gate_proj = operations.Linear(dim, dim * 4, bias=False, dtype=dtype, device=device)
        self.up_proj = operations.Linear(dim, dim * 4, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(dim * 4, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, dim, heads, dtype, device, operations):
        super().__init__()
        self.input_layernorm = AdaptiveRMSNorm(dim, dtype, device, operations)
        self.post_attention_layernorm = AdaptiveRMSNorm(dim, dtype, device, operations)
        self.self_attn = Attention(dim, heads, dtype, device, operations)
        self.mlp = MLP(dim, dtype, device, operations)

    def forward(self, x, time, rope, transformer_options):
        x = x + self.self_attn(self.input_layernorm(x, time), rope, transformer_options)
        return x + self.mlp(self.post_attention_layernorm(x, time))


class DiffLlama(nn.Module):
    def __init__(self, dim=1024, depth=22, heads=16, mel_dim=128, dtype=None, device=None, operations=None):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // heads
        self.layers = nn.ModuleList([Block(dim, heads, dtype, device, operations) for _ in range(depth)])
        self.norm = AdaptiveRMSNorm(dim, dtype, device, operations)
        for name, dim_in, dim_out in [("diff_step_mlp", dim, dim), ("cond_mlp", dim, dim),
                                      ("mel_mlp", mel_dim, dim), ("mel_out_mlp", dim, mel_dim)]:
            setattr(self, name, nn.Sequential(operations.Linear(dim_in, dim * 4, dtype=dtype, device=device), nn.SiLU(),
                                             operations.Linear(dim * 4, dim_out, dtype=dtype, device=device)))

    def prepare_conditioning(self, cond):
        return self.cond_mlp(cond)

    def forward(self, mel, time, cond, transformer_options={}):
        x = self.mel_mlp(mel) + cond
        frequency = torch.exp(torch.arange(self.dim // 2, device=x.device, dtype=torch.float32)
                              * (-math.log(10000) / (self.dim // 2 - 1)))
        phase = time[:, None].float() * frequency[None]
        time = self.diff_step_mlp(torch.cat((phase.sin(), phase.cos()), dim=-1).to(x.dtype))
        cos, sin, _ = precompute_freqs_cis(self.head_dim, torch.arange(x.shape[1], device=x.device)[None], 10000.0, device=x.device)
        cos = cos[..., :self.head_dim // 2]
        rope = torch.stack((cos, -sin, sin, cos), dim=-1).unflatten(-1, (2, 2))
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(x.device)
        for block in self.layers:
            comfy.model_management.throw_exception_if_processing_interrupted()
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="block")
            result = block(x, time, rope, transformer_options)
            if torch.is_grad_enabled():
                x = result
            else:
                # Keep the block state outside the nested allocator scope.
                x.copy_(result)
            del result
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()
        return self.mel_out_mlp(self.norm(x, time))


class Singer(nn.Module):
    def __init__(self, svc=False, dim=1024, depth=22, heads=16, encoder_dim=512, encoder_depth=4,
                 vocab_size=3000, mel_dim=128, dtype=None, device=None, operations=None):
        super().__init__()
        self.svc = svc
        self.dtype = dtype
        self.f0_encoder = operations.Embedding(361, encoder_dim, dtype=dtype, device=device)
        if not svc:
            self.note_text_encoder = operations.Embedding(vocab_size, encoder_dim, dtype=dtype, device=device)
            self.note_pitch_encoder = operations.Embedding(256, encoder_dim, dtype=dtype, device=device)
            self.note_type_encoder = operations.Embedding(256, encoder_dim, dtype=dtype, device=device)
            self.preflow = nn.Sequential(*[ConvNeXtBlock(encoder_dim, dtype, device, operations) for _ in range(encoder_depth)])
        self.cond_emb = operations.Linear(encoder_dim, dim, dtype=dtype, device=device)
        self.diff_estimator = DiffLlama(dim, depth, heads, mel_dim, dtype, device, operations)

    def encode_score(self, phoneme, pitch, note_type, mel2note, f0):
        x = self.note_text_encoder(phoneme, out_dtype=self.dtype)
        x = x + self.note_pitch_encoder(pitch, out_dtype=self.dtype) + self.note_type_encoder(note_type, out_dtype=self.dtype)
        x = self.preflow(x)
        x = x.gather(1, mel2note[..., None].expand(-1, -1, x.shape[-1]))
        return x + self.f0_encoder(f0, out_dtype=self.dtype)

    def prepare_conditioning(self, features):
        return self.diff_estimator.prepare_conditioning(self.cond_emb(features))

    def forward(self, mel, time, cond, transformer_options={}):
        return self.diff_estimator(mel, time, cond, transformer_options)


def coarse_f0(f0, shift=0):
    bins = (1200 * torch.log2(f0.clamp(min=32.7031956625) / 32.7031956625) / 20 + 1).round().long().clamp(1, 360)
    return torch.where(f0 > 0, (bins + shift * 5).clamp(1, 360), 0)
