"""Drama 3 Qwen3.5-MoE backbone with execution-local attention state."""
import torch
from torch import nn
from torch.nn import functional as F
import comfy.ops
import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy_kitchen as ck
from comfy_kitchen.gated_delta import is_available as delta_available
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device
from comfy.text_encoders.llama import RMSNorm
from comfy.text_encoders.qwen35 import GatedDeltaNet as QwenDeltaNet
from .delta import chunk_gated_delta_rule


def attention(q, k, v, mask, preference):
    containers = tuple(map(AttentionTensorContainer, (q, k, v)))
    return optimized_attention_for_device(q.device, mask=True)(
        *containers, q.shape[1], mask=mask, skip_reshape=True, enable_gqa=True,
        preferred_attention=preference)


def rotary_matrix(cos, sin):
    return torch.stack((cos, -sin, sin, cos), -1).unflatten(-1, (2, 2))


def causal_cache(q, k, v, start, cache, capacity):
    end = start + q.shape[2]
    if not cache:
        shape = (*k.shape[:2], capacity, k.shape[-1])
        cache.extend((k.new_empty(shape), v.new_empty(shape)))
    cache[0][:, :, start:end] = k
    cache[1][:, :, start:end] = v
    visible = torch.arange(end, device=q.device)[None, :] <= torch.arange(start, end, device=q.device)[:, None]
    mask = q.new_zeros(visible.shape).masked_fill_(~visible, float('-inf'))
    return cache[0][:, :, :end], cache[1][:, :, :end], mask


class FullAttention(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        kw = dict(bias=False, dtype=dtype, device=device)
        width = self.heads * self.head_dim
        self.q_proj = operations.Linear(config.hidden_size, 2 * width, **kw)
        self.k_proj = operations.Linear(config.hidden_size, self.kv_heads * self.head_dim, **kw)
        self.v_proj = operations.Linear(config.hidden_size, self.kv_heads * self.head_dim, **kw)
        self.o_proj = operations.Linear(width, config.hidden_size, **kw)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=True, dtype=dtype, device=device)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=True, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def rotate(self, q, k, start):
        inv = self.rope_theta ** (-torch.arange(0, self.rotary_dim, 2, device=q.device, dtype=torch.float32) / self.rotary_dim)
        phase = torch.arange(start, start + q.shape[2], device=q.device, dtype=torch.float32)[:, None] * inv
        matrix = rotary_matrix(phase.cos(), phase.sin())[None, None]
        qr, kr = ck.apply_rope_split_half(q[..., :self.rotary_dim], k[..., :self.rotary_dim], matrix)
        return (torch.cat((qr, q[..., self.rotary_dim:]), -1),
                torch.cat((kr, k[..., self.rotary_dim:]), -1))

    def forward(self, x, start, cache, capacity):
        batch, length, _ = x.shape
        q, gate = self.q_proj(x).view(batch, length, self.heads, 2 * self.head_dim).chunk(2, -1)
        k = self.k_norm(self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim))
        q, k = self.rotate(self.q_norm(q).transpose(1, 2), k.transpose(1, 2), start)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        k, v, mask = causal_cache(q, k, v, start, cache, capacity)
        out = attention(q, k, v, mask, self.comfy_attention)
        return self.o_proj(out * gate.reshape(batch, length, -1).sigmoid())


class DeltaNet(QwenDeltaNet):
    def __init__(self, config, dtype, device, operations):
        # Reuse Qwen's managed projections; Fish has an additive L2 epsilon and
        # preserves recurrent state between prefill chunks.
        config.conv_kernel_size = config.linear_conv_kernel_dim
        super().__init__(config, device=device, dtype=dtype, ops=operations)

    def forward(self, x, start, cache, capacity):
        batch, length, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        history = self.conv_kernel_size - 1
        if not cache:
            cache.extend((mixed.new_zeros(batch, mixed.shape[1], history),
                          torch.zeros(batch, self.num_value_heads, self.key_head_dim, self.value_head_dim, device=x.device, dtype=torch.float32)))
        padded = torch.cat((cache[0], mixed), -1)
        cache[0].copy_(padded[..., -history:])
        with comfy.ops.CastBiasWeightContext(self.conv1d, x, offloadable=True) as (w, bias):
            mixed = F.silu(F.conv1d(padded, w, bias, groups=mixed.shape[1]))
        z = self.in_proj_z(x)
        decay = -mm.cast_to(self.A_log, device=x.device, dtype=torch.float32).exp()
        dt = mm.cast_to(self.dt_bias, device=x.device, dtype=torch.float32)
        if start and length == 1 and delta_available(x.device, self.key_head_dim, self.value_head_dim):
            with comfy.ops.CastBiasWeightContext(self.in_proj_a, x, offloadable=True) as (wa, _), comfy.ops.CastBiasWeightContext(self.in_proj_b, x, offloadable=True) as (wb, _):
                core = ck.gated_delta_decode_fused(mixed, x, wa, wb, dt, decay, cache[1], self.key_dim,
                    self.num_key_heads, self.key_head_dim ** -0.5, z,
                    comfy.ops.cast_to_input(self.norm.weight, x), self.norm.eps)
        else:
            q, k, v = mixed.transpose(1, 2).split((self.key_dim, self.key_dim, self.value_dim), -1)
            q = q.reshape(batch, length, self.num_key_heads, self.key_head_dim)
            k = k.reshape_as(q)
            groups = self.num_value_heads // self.num_key_heads
            q, k = q.repeat_interleave(groups, 2), k.repeat_interleave(groups, 2)
            v = v.reshape(batch, length, self.num_value_heads, self.value_head_dim)
            beta = self.in_proj_b(x).float().sigmoid()
            g = decay * F.softplus(self.in_proj_a(x).float() + dt)
            core, state = chunk_gated_delta_rule(q, k, v, g, beta, cache[1])
            cache[1].copy_(state)
            # Fish applies SiLU in fp32, after the weighted RMS output cast.
            weighted = RMSNorm.forward(self.norm, core.reshape(-1, self.value_head_dim))
            core = (weighted * F.silu(z.reshape(-1, self.value_head_dim).float())).to(x.dtype)
        return self.out_proj(core.reshape(batch, length, self.value_dim))


class DenseMLP(nn.Module):
    def __init__(self, hidden, intermediate, dtype, device, operations):
        super().__init__()
        kw = dict(bias=False, dtype=dtype, device=device)
        self.gate_proj = operations.Linear(hidden, intermediate, **kw)
        self.up_proj = operations.Linear(hidden, intermediate, **kw)
        self.down_proj = operations.Linear(intermediate, hidden, **kw)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        h, i, e = config.hidden_size, config.moe_intermediate_size, config.num_experts
        self.top_k, self.normalize = config.num_experts_per_tok, config.norm_topk_prob
        self.gate = operations.Linear(h, e, bias=False, dtype=dtype, device=device)
        self.experts = nn.Module()
        self.experts.gate_up_proj = operations.MoEExperts(e, h, 2 * i, bias=False, dtype=dtype, device=device)
        self.experts.down_proj = operations.MoEExperts(e, i, h, bias=False, dtype=dtype, device=device)
        self.shared_expert = DenseMLP(h, config.shared_expert_intermediate_size, dtype, device, operations)
        self.shared_expert_gate = operations.Linear(h, 1, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        weights, ids = self.gate(flat).float().softmax(-1).topk(self.top_k, -1)
        if self.normalize:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(flat.dtype)
        order = torch.argsort(ids.flatten(), stable=True)
        rows = order // self.top_k
        counts = torch.bincount(ids.flatten(), minlength=self.gate.out_features).tolist()
        result = flat.new_zeros(flat.shape, dtype=torch.float32)
        offset = 0
        # Fish sums routed contributions in fp32; the shared BF16-reduction helper
        # has a different result contract. Reuse its optimized expert banks.
        with self.experts.gate_up_proj.bank_resident(flat) as gate, self.experts.down_proj.bank_resident(flat) as down:
            for expert, count in enumerate(counts):
                if count:
                    selected = rows[offset:offset + count]
                    hidden = comfy.ops._swiglu_eager(gate.expert_linear(flat[selected], expert))
                    out = down.expert_linear(hidden, expert)
                    result.index_add_(0, selected, out.float() * weights.flatten()[order[offset:offset + count], None].float())
                    offset += count
        return (result.to(flat.dtype) + self.shared_expert_gate(flat).sigmoid() * self.shared_expert(flat)).view_as(x)


class Block(nn.Module):
    def __init__(self, config, index, dtype, device, operations):
        super().__init__()
        self.linear = config.layer_types[index] == 'linear_attention'
        name, cls = ('linear_attn', DeltaNet) if self.linear else ('self_attn', FullAttention)
        self.add_module(name, cls(config, dtype, device, operations))
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True, dtype=dtype, device=device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True, dtype=dtype, device=device)
        self.mlp = SparseMoE(config, dtype, device, operations)

    def forward(self, x, start, cache, capacity):
        mixer = self.linear_attn if self.linear else self.self_attn
        x = x + mixer(self.input_layernorm(x), start, cache, capacity)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.embed_tokens = operations.Embedding(config.vocab_size, config.hidden_size, dtype=dtype, device=device)
        self.layers = nn.ModuleList(Block(config, i, dtype, device, operations) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True, dtype=dtype, device=device)

    def forward(self, x, start, caches, capacity):
        queue = prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": True})
        for index, layer in enumerate(self.layers):
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = layer(x, start, caches[index], capacity)
            prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='drama3')
        prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='drama3')
        return self.norm(x)
