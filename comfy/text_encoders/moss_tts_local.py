# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
# Licensed under Apache-2.0; see ../ldm/audio/moss_audio_tokenizer.LICENSE.
"""MOSS Local Transformer: global time steps and synchronous codebook decoding."""
import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention_for_device
from .llama import BaseLlama, Llama2_, Qwen3_8BConfig, RMSNorm, TransformerBlock
from .moss_tts_delay import AUDIO_END, AUDIO_PAD, AUDIO_START


def sample_local_token(logits, history, generator, temperature=1.0, top_p=1.0, top_k=50, penalty=1.0):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    if penalty != 1.0:
        selected = logits.gather(1, history)
        logits.scatter_(1, history, torch.where(selected < 0, selected * penalty, selected / penalty))
    logits = logits / temperature
    if top_k > 0:
        threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, indices = logits.sort(dim=-1, descending=False)
        remove = sorted_logits.softmax(-1).cumsum(-1) <= 1 - top_p
        remove[:, -1] = False
        logits = logits.masked_fill(torch.zeros_like(remove).scatter_(1, indices, remove), -torch.inf)
    return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)


class SpeechBridge(torch.nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, dtype, device, operations):
        super().__init__()
        self.gate_proj = operations.Linear(input_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = operations.Linear(input_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, output_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class LocalTransformer(torch.nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([TransformerBlock(config, i, dtype=dtype, device=device, ops=operations)
                                          for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, device=device)

    def forward(self, x, cache=None):
        cache = [[] for _ in self.layers] if cache is None else cache
        past = cache[0][2] if cache[0] else 0
        mask = None
        if x.shape[1] > 1:
            mask = torch.full((x.shape[1], past + x.shape[1]), -torch.inf, dtype=x.dtype, device=x.device).triu_(past + 1)
        attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": True})
        for i, layer in enumerate(self.layers):
            def core():
                nonlocal x
                x, cache[i] = layer(x, attention_mask=mask, optimized_attention=attention, past_key_value=cache[i])
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, malloc_scope='block')
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope='block')
        return self.norm(x), cache


class MossTTSLocal(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        language = dict(hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
                        num_attention_heads=16, num_key_value_heads=8, vocab_size=155648, lm_head=False)
        language.update(config_dict.get('language', {}))
        config = Qwen3_8BConfig(**language)
        config.prefetch_dynamic_vbars = True
        self.model = Llama2_(config, dtype=dtype, device=device, ops=operations)
        local = dict(language, hidden_size=1536, intermediate_size=8960, num_hidden_layers=4)
        local.update(config_dict.get('local', {}))
        local_config = Qwen3_8BConfig(**local)
        self.local_transformer = LocalTransformer(local_config, dtype, device, operations)
        self.num_layers = config.num_hidden_layers
        self.codebooks = config_dict.get('codebooks', 32)
        self.dtype = dtype
        inner = config_dict.get('bridge_size', 2048)
        self.emb_ext = torch.nn.ModuleList([operations.Embedding(1025, config.hidden_size, device=device, dtype=dtype) for _ in range(self.codebooks)])
        self.speech_embedding_to_local_mlp = SpeechBridge(config.hidden_size, inner, local_config.hidden_size, dtype, device, operations)
        self.local_to_speech_embedding_mlps = torch.nn.ModuleList([
            SpeechBridge(local_config.hidden_size, inner, config.hidden_size, dtype, device, operations) for _ in range(self.codebooks + 1)])
        self.layer_norm_before_lm_heads = torch.nn.ModuleList([
            RMSNorm(config.hidden_size, eps=1e-6, dtype=dtype, device=device) for _ in range(self.codebooks + 1)])
        self.lm_heads = torch.nn.ModuleList([operations.Linear(config.hidden_size, size, bias=False, device=device, dtype=dtype)
                                           for size in [config.vocab_size, *([1025] * self.codebooks)]])

    def combined_embeddings(self, input_ids):
        x = self.model.embed_tokens(input_ids[..., 0], out_dtype=self.dtype)
        for i, embedding in enumerate(self.emb_ext):
            x = x + embedding(input_ids[..., i + 1], out_dtype=self.dtype)
        return x

    def channel_logits(self, local_inputs, channel, cache=None):
        hidden, cache = self.local_transformer(local_inputs, cache)
        hidden = hidden[:, -1]
        hidden = self.local_to_speech_embedding_mlps[channel](hidden)
        return self.lm_heads[channel](self.layer_norm_before_lm_heads[channel](hidden)), cache

    def generate_local(self, input_ids, attention_mask, max_new_tokens, seed, text_temperature=1.5, text_top_p=1.0,
                       text_top_k=50, audio_temperature=1.0, audio_top_p=0.95, audio_top_k=50, audio_repetition_penalty=1.1):
        batch, length, channels = input_ids.shape
        capacity = length + max_new_tokens
        cache = self.model.init_kv_cache(batch, capacity, input_ids.device, self.dtype)
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        history = torch.empty((batch, capacity, channels), dtype=torch.int64, device=input_ids.device)
        history[:, :length] = input_ids
        mask = torch.ones((batch, capacity), dtype=attention_mask.dtype, device=input_ids.device)
        mask[:, :length] = attention_mask
        stopped = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        current = input_ids
        used = length
        progress = comfy.utils.ProgressBar(max_new_tokens)
        for step in range(max_new_tokens):
            comfy.model_management.throw_exception_if_processing_interrupted()
            hidden, _, cache = self.model(None, embeds=self.combined_embeddings(current), attention_mask=mask[:, :used], past_key_values=cache)
            local = self.speech_embedding_to_local_mlp(hidden[:, -1])
            local_cache = None
            next_tokens = []
            for q in range(channels):
                comfy.model_management.throw_exception_if_processing_interrupted()
                logits, local_cache = self.channel_logits(local[:, None], q, local_cache)
                if q:
                    logits[:, AUDIO_PAD] = -torch.inf
                token = sample_local_token(logits, history[:, :used, q], generator,
                    audio_temperature if q else text_temperature, audio_top_p if q else text_top_p,
                    audio_top_k if q else text_top_k, audio_repetition_penalty if q else 1.0)
                next_tokens.append(token.masked_fill(stopped, AUDIO_PAD if q else AUDIO_END))
                if q < channels - 1:
                    embedding = self.model.embed_tokens if q == 0 else self.emb_ext[q - 1]
                    local = self.speech_embedding_to_local_mlp(embedding(token, out_dtype=self.dtype))
            current = torch.stack(next_tokens, dim=-1)[:, None]
            history[:, used:used + 1] = current
            stopped |= current[:, 0, 0] == AUDIO_END
            used += 1
            progress.update(step + 1)
            if stopped.all():
                break
        outputs = []
        for row in history[:, :used]:
            start = max((i for i, token in enumerate(row[:, 0].tolist()) if token == AUDIO_START), default=-1)
            outputs.append((max(0, length - start - 1), row[start:].to(device='cpu', copy=True)))
        return outputs
