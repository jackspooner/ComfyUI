"""Fish codebook-MTP decoder. Adapted from Fish Speech 5415669; see NOTICE."""
import math
import torch
from torch import nn
from torch.nn import functional as F
import comfy.ops
import comfy_kitchen as ck
import comfy.model_management as mm
import comfy.model_prefetch as prefetch
from comfy.text_encoders.llama import RMSNorm
from comfy.ldm.modules.attention import ComfyAttention
from .model import attention, causal_cache, rotary_matrix


def sample(logits, temperature, top_k, top_p, generator):
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Drama 3 produced non-finite logits')
    values, ids = (logits.float() / max(temperature, 1e-5)).sort(descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
    dropped[..., 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_(generator=generator)
    return ids.gather(-1, (probabilities / noise).argmax(-1, keepdim=True)).squeeze(-1)


class TowerAttention(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.config = config
        width, kv = config.n_head * config.head_dim, config.n_local_heads * config.head_dim
        kw = dict(bias=False, dtype=dtype, device=device)
        self.wqkv = operations.Linear(config.dim, 2 * width + 2 * kv, **kw)
        self.wo = operations.Linear(width, config.dim, **kw)
        self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps, add=True, dtype=dtype, device=device)
        self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps, add=True, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def rotate(self, q, k, start):
        c = self.config
        phase = torch.arange(start, start + q.shape[2], device=q.device, dtype=torch.float32)[:, None] * c.rope_base ** (-torch.arange(0, c.rotary_dim, 2, device=q.device, dtype=torch.float32) / c.rotary_dim)
        matrix = rotary_matrix(phase.cos(), phase.sin())[None, None]
        qr, kr = ck.apply_rope(q[..., :c.rotary_dim], k[..., :c.rotary_dim], matrix)
        return (torch.cat((qr, q[..., c.rotary_dim:]), -1),
                torch.cat((kr, k[..., c.rotary_dim:]), -1))

    def forward(self, x, start, cache, capacity):
        c, length = self.config, len(x)
        width, kv = c.n_head * c.head_dim, c.n_local_heads * c.head_dim
        q, gate, k, v = self.wqkv(x).split((width, width, kv, kv), -1)
        q = self.q_norm(q.view(length, c.n_head, c.head_dim)).transpose(0, 1)[None]
        k = self.k_norm(k.view(length, c.n_local_heads, c.head_dim)).transpose(0, 1)[None]
        q, k = self.rotate(q, k, start)
        v = v.view(length, c.n_local_heads, c.head_dim).transpose(0, 1)[None]
        k, v, mask = causal_cache(q, k, v, start, cache, capacity)
        out = attention(q, k, v, mask, self.comfy_attention).reshape(length, width)
        return self.wo(out * gate.sigmoid())


class TowerMLP(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        kw = dict(bias=False, dtype=dtype, device=device)
        self.w1 = operations.Linear(config.dim, config.intermediate_size, **kw)
        self.w3 = operations.Linear(config.dim, config.intermediate_size, **kw)
        self.w2 = operations.Linear(config.intermediate_size, config.dim, **kw)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TowerBlock(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.attention = TowerAttention(config, dtype, device, operations)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps, add=True, dtype=dtype, device=device)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps, add=True, dtype=dtype, device=device)
        self.feed_forward = TowerMLP(config, dtype, device, operations)

    def forward(self, x, start, cache, capacity):
        x = x + self.attention(self.attention_norm(x), start, cache, capacity)
        return x + self.feed_forward(self.ffn_norm(x))


class TowerLayer(nn.Module):
    def __init__(self, config, shared, dtype, device, operations):
        super().__init__()
        self.shared = shared
        self.add_module('semantic_embedding' if shared else 'codebook_embedding', operations.Embedding(config.vocab_size, config.dim, dtype=dtype, device=device))
        for name in ('enorm', 'hnorm', 'final_layernorm'):
            self.add_module(name, RMSNorm(config.dim, eps=config.norm_eps, add=True, dtype=dtype, device=device))
        self.eh_proj = operations.Linear(2 * config.dim, config.dim, bias=False, dtype=dtype, device=device)
        self.transformer_layer = TowerBlock(config, dtype, device, operations)

    def forward(self, hidden, codes, semantic, start, cache, capacity, extra=None):
        embedding = self.semantic_embedding if self.shared else self.codebook_embedding
        e = torch.where(semantic[:, None], embedding(codes.clamp(0, embedding.num_embeddings - 1)), 0.)
        if extra is not None:
            e = e + extra
        mixed = self.eh_proj(torch.cat((self.enorm(e), self.hnorm(hidden)), -1))
        return self.final_layernorm(self.transformer_layer(mixed, start, cache, capacity))


class Tower(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        c = self.config = config.audio_decoder
        self.semantic_begin, self.semantic_end, self.eos = config.semantic_start_token_id, config.semantic_end_token_id, config.eos_token_id
        self.codebook_embeddings = operations.Embedding(c.vocab_size * c.num_codebooks, c.dim, dtype=dtype, device=device)
        self.shared_layers = nn.ModuleList(TowerLayer(c, True, dtype, device, operations) for _ in range(c.shared_enrichment_layers))
        self.residual_layers = nn.ModuleList(TowerLayer(c, False, dtype, device, operations) for _ in range(c.num_codebooks - 1))
        self.codebook_heads = nn.ModuleList(operations.Linear(c.dim, c.vocab_size, bias=False, dtype=dtype, device=device) for _ in range(c.num_codebooks - 1))

    def is_semantic(self, tokens):
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames, first, end):
        if first == end:
            return torch.zeros(len(frames), self.config.dim, device=frames.device, dtype=self.codebook_embeddings.weight.dtype)
        offsets = torch.arange(first, end, device=frames.device) * self.config.vocab_size
        return self.codebook_embeddings(frames[:, 1 + first:1 + end] + offsets).sum(1)

    def embed_backbone(self, frames, text):
        n = self.config.vq_backbone_codebooks
        mixed = (text + self.embedding_sum(frames, 1, 1 + n)) / math.sqrt(1 + n)
        return torch.where(self.is_semantic(frames[:, 0])[:, None], mixed, text)

    def run_shared(self, hidden, c0, semantic, start, frames, caches, capacity):
        unseen = self.embedding_sum(frames, 1 + self.config.vq_backbone_codebooks, self.config.num_codebooks)
        unseen = torch.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.)
        queue = prefetch.make_prefetch_queue(list(self.shared_layers), hidden.device, {"prefetch_dynamic_vbars": True})
        for i, layer in enumerate(self.shared_layers):
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal hidden
                hidden = layer(hidden, c0, semantic, start, caches[i], capacity, unseen if i == 0 else None)
            prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, malloc_scope='drama3_tower')
        prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope='drama3_tower')
        return hidden

    def residual(self, hidden, labels, semantic, start, caches, capacity, sampling=None):
        queue = prefetch.make_prefetch_queue(list(self.residual_layers), hidden.device, {"prefetch_dynamic_vbars": True})
        codes = [labels[:, 0]]
        for i, layer in enumerate(self.residual_layers):
            mm.throw_exception_if_processing_interrupted()
            def core():
                nonlocal hidden
                hidden = layer(hidden, codes[-1] if sampling else labels[:, i], semantic, start, caches[i], capacity)
            prefetch.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, malloc_scope='drama3_tower')
            if sampling:
                codes.append(sample(self.codebook_heads[i](hidden), *sampling))
        prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope='drama3_tower')
        return codes

    def prefill(self, hidden, start, frames, next_frames, caches, capacity):
        semantic = self.is_semantic(next_frames[:, 0])
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, start, frames, caches[0], capacity)
        self.residual(hidden, next_frames[:, 1:], semantic, start, caches[1], capacity)

    def decode(self, logits, hidden, start, frames, history, caches, capacity, settings, generator):
        candidates = torch.cat((torch.tensor([self.eos], device=hidden.device), torch.arange(self.semantic_begin, self.semantic_end + 1, device=hidden.device))).sort().values
        selected_logits = logits[:, candidates]
        first = candidates[sample(selected_logits, *settings, generator)]
        retry = candidates[sample(selected_logits, 1., 30, .95, generator)]
        repeated = (history[-16:] == first[:, None]).any(-1) & self.is_semantic(first)
        main = torch.where(repeated, retry, first)
        c0 = (main - self.semantic_begin).clamp(0, self.config.vocab_size - 1)
        semantic = self.is_semantic(main)
        hidden = self.run_shared(hidden, c0, semantic, start, frames, caches[0], capacity)
        codes = self.residual(hidden, c0[:, None], semantic, start, caches[1], capacity, (*settings, generator))
        return torch.stack((main, *codes), -1)
