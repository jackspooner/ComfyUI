# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
# Licensed under Apache-2.0; see ../ldm/audio/moss_audio_tokenizer.LICENSE.
"""MOSS Local v1.5: Qwen3 time model, NanoGPT2 book decoder and tied heads.

Adapted from OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 at
be7766a6735b98bd793f7c79fb720b4d0f5d13b8. Original Local 1.7B is separate.
"""
import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.utils
from comfy.console_progress import advance, report_progress
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device
from .llama import BaseLlama, Llama2_, Qwen3_8BConfig


PAD, AUDIO_START, AUDIO_END, AUDIO_GEN, AUDIO_PAD = 151643, 151669, 151670, 151656, 1024


def sample_v15_token(logits, generator, temperature=1., top_p=1., top_k=50, history=None, penalty=1.):
    if temperature <= 0:
        raise ValueError('temperature must be positive when do_sample=True.')
    scores = logits.float()
    if history is not None and penalty != 1.:
        scores = scores.clone()
        for row in range(scores.shape[0]):
            indices = torch.unique(history[row])
            indices = indices[(indices >= 0) & (indices < scores.shape[-1])]
            selected = scores[row, indices]
            scores[row, indices] = torch.where(selected < 0, selected * penalty, selected / penalty)
    scores = scores / temperature
    if top_k > 0 and top_k < scores.shape[-1]:
        scores = scores.masked_fill(scores < scores.topk(top_k, dim=-1).values[:, -1:], -torch.inf)
    if 0. < top_p < 1.:
        ordered, indices = scores.sort(descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scores = scores.masked_fill(torch.zeros_like(remove).scatter_(-1, indices, remove), -torch.inf)
    return torch.multinomial(scores.softmax(-1), 1, generator=generator).squeeze(-1)


class NanoAttention(nn.Module):
    def __init__(self, dim, heads, dtype, device, operations):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.c_attn = operations.Linear(dim, dim*3, bias=True, dtype=dtype, device=device)
        self.c_proj = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, cache):
        batch, length, dim = x.shape
        past = cache[2] if cache else 0
        q, k, v = self.c_attn(x).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        frequencies = 1. / (1000000. ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim))
        positions = torch.arange(past, past+length, device=x.device, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None]
        cos = angles.cos().repeat_interleave(2, -1)[None, None].to(q.dtype)
        sin = angles.sin().repeat_interleave(2, -1)[None, None].to(q.dtype)
        def rotary(value):
            rotated = torch.stack((-value[..., 1::2], value[..., ::2]), -1).reshape_as(value)
            return value*cos + rotated*sin
        q, k = rotary(q), rotary(k)
        if not cache:
            shape = (batch, self.heads, 13, self.head_dim)
            cache.extend([torch.empty(shape, device=x.device, dtype=k.dtype), torch.empty(shape, device=x.device, dtype=v.dtype), 0])
        used = past+length
        if used > cache[0].shape[2]:
            raise ValueError('MOSS Local v1.5 local cache exceeds its 13-position capacity')
        cache[0][:, :, past:used].copy_(k)
        cache[1][:, :, past:used].copy_(v)
        cache[2] = used
        mask = torch.full((length, used), -torch.inf, device=x.device, dtype=x.dtype).triu_(past+1)
        attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
        output = attention(AttentionTensorContainer(q), AttentionTensorContainer(cache[0][:, :, :used]),
            AttentionTensorContainer(cache[1][:, :, :used]), self.heads, mask=mask, skip_reshape=True,
            preferred_attention=self.comfy_attention)
        return self.c_proj(output)


class NanoMLP(nn.Module):
    def __init__(self, dim, inner, dtype, device, operations):
        super().__init__()
        self.fc_in = operations.Linear(dim, inner, bias=True, dtype=dtype, device=device)
        self.fc_out = operations.Linear(inner, dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.fc_out(F.silu(self.fc_in(x)))


class NanoBlock(nn.Module):
    def __init__(self, dim, inner, heads, dtype, device, operations):
        super().__init__()
        self.ln_1 = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.attn = NanoAttention(dim, heads, dtype, device, operations)
        self.ln_2 = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.mlp = NanoMLP(dim, inner, dtype, device, operations)

    def forward(self, x, cache):
        x = x + self.attn(self.ln_1(x), cache)
        return x + self.mlp(self.ln_2(x))


class NanoLocal(nn.Module):
    def __init__(self, dim, inner, heads, dtype, device, operations):
        super().__init__()
        self.h = nn.ModuleList([NanoBlock(dim, inner, heads, dtype, device, operations)])
        self.ln_f = operations.LayerNorm(dim, eps=1e-6, dtype=dtype, device=device)

    def forward(self, x, cache=None):
        cache = [[] for _ in self.h] if cache is None else cache
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.h), x.device, {'prefetch_dynamic_vbars': True})
        for i, layer in enumerate(self.h):
            comfy.model_management.throw_exception_if_processing_interrupted()
            def core():
                nonlocal x
                x = layer(x, cache[i])
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        return self.ln_f(x), cache


class MossTTSLocalV15(BaseLlama, nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        language = dict(hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
            num_attention_heads=32, num_key_value_heads=8, vocab_size=151936,
            max_position_embeddings=32768, lm_head=False)
        language.update(config_dict.get('language', {}))
        config = Qwen3_8BConfig(**language)
        config.prefetch_dynamic_vbars = True
        self.model = Llama2_(config, dtype=dtype, device=device, ops=operations)
        local = dict(dim=config.hidden_size, inner=9728, heads=32)
        local.update(config_dict.get('local', {}))
        self.local_transformer = NanoLocal(**local, dtype=dtype, device=device, operations=operations)
        self.audio_embeddings = nn.ModuleList([operations.Embedding(1024, config.hidden_size, dtype=dtype, device=device) for _ in range(12)])
        self.audio_lm_heads = nn.ModuleList([operations.Linear(config.hidden_size, 1024, bias=False, dtype=dtype, device=device) for _ in range(12)])
        self.text_lm_head = operations.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype, device=device)
        self.local_text_lm_head = operations.Linear(config.hidden_size, 2, bias=False, dtype=dtype, device=device)
        self.text_lm_head.weight = self.model.embed_tokens.weight
        for head, embedding in zip(self.audio_lm_heads, self.audio_embeddings):
            head.weight = embedding.weight
        self.num_layers, self.codebooks, self.dtype = config.num_hidden_layers, 12, dtype

    def combined_embeddings(self, input_ids):
        x = self.model.embed_tokens(input_ids[..., 0], out_dtype=self.dtype)
        for channel, embedding in enumerate(self.audio_embeddings):
            ids = input_ids[..., channel+1]
            valid = ids != AUDIO_PAD
            x = x + embedding(ids.masked_fill(~valid, 0), out_dtype=self.dtype)*valid[..., None]
        return x

    def global_hidden(self, input_ids, attention_mask, cache):
        # The pinned decoder numbers real positions independently of left
        # padding. Keep padded queries inert without changing shared Qwen math.
        valid = attention_mask[:, -input_ids.shape[1]:].bool()
        positions = attention_mask.long().cumsum(-1) - 1
        positions = positions.masked_fill(~attention_mask.bool(), 0)[:, -input_ids.shape[1]:]
        hidden, _, cache = self.model(None,
            embeds=self.combined_embeddings(input_ids) * valid[..., None],
            attention_mask=attention_mask, position_ids=positions, past_key_values=cache)
        return hidden * valid[..., None], cache

    @report_progress('MOSS Local v1.5', 'generating audio', unit='frames')
    def generate_local(self, input_ids, attention_mask, max_new_tokens, seed,
                       text_temperature=1., text_top_p=1., text_top_k=50,
                       audio_temperature=1.7, audio_top_p=.8, audio_top_k=25, audio_repetition_penalty=1.):
        batch, length, channels = input_ids.shape
        if channels != 13:
            raise ValueError('MOSS Local v1.5 requires exactly 12 audio codebooks')
        cache = self.model.init_kv_cache(batch, length+max_new_tokens, input_ids.device, self.dtype)
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        history = torch.empty(batch, length+max_new_tokens, channels, device=input_ids.device, dtype=torch.int64)
        history[:, :length] = input_ids
        mask = torch.ones(batch, length+max_new_tokens, device=input_ids.device, dtype=attention_mask.dtype)
        mask[:, :length] = attention_mask
        current, used = input_ids, length
        finished = torch.zeros(batch, device=input_ids.device, dtype=torch.bool)
        candidates = torch.tensor([AUDIO_GEN, AUDIO_END], device=input_ids.device)
        progress = comfy.utils.ProgressBar(max_new_tokens)
        for step in range(max_new_tokens):
            comfy.model_management.throw_exception_if_processing_interrupted()
            hidden, cache = self.global_hidden(current, mask[:, :used], cache)
            local, local_cache = self.local_transformer(hidden[:, -1:])
            local = local[:, -1]
            token = candidates[sample_v15_token(self.local_text_lm_head(local), generator,
                text_temperature, text_top_p, text_top_k)]
            continuing = (token == AUDIO_GEN) & ~finished
            finished |= token == AUDIO_END
            if not continuing.any():
                break
            next_tokens = []
            for channel in range(12):
                comfy.model_management.throw_exception_if_processing_interrupted()
                logits = self.audio_lm_heads[channel](local)
                previous = history[:, length:used, channel+1] if used > length else None
                audio = sample_v15_token(logits, generator, audio_temperature, audio_top_p,
                    audio_top_k, previous, audio_repetition_penalty)
                next_tokens.append(audio)
                if channel < 11:
                    local, local_cache = self.local_transformer(self.audio_embeddings[channel](audio, out_dtype=self.dtype)[:, None], local_cache)
                    local = local[:, -1]
            current = torch.full((batch, 1, channels), AUDIO_PAD, device=input_ids.device, dtype=torch.int64)
            current[:, 0, 0] = torch.where(continuing, AUDIO_GEN, PAD)
            current[:, 0, 1:] = torch.stack(next_tokens, -1).masked_fill(~continuing[:, None], AUDIO_PAD)
            history[:, used:used+1] = current
            mask[:, used] = continuing
            used += 1
            progress.update_absolute(step+1)
            advance()
        result = []
        for row in history[:, :used]:
            starts = torch.nonzero(row[:length, 0] == AUDIO_START).flatten()
            if not len(starts):
                raise ValueError('MOSS Local v1.5 prompt must contain its audio start marker')
            start = starts[-1].item()
            result.append((length-start-1, row[start:].to(device='cpu', copy=True)))
        return result
