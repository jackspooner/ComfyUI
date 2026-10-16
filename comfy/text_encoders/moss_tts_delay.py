# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0; see
# ../ldm/audio/moss_audio_tokenizer.LICENSE for the full license.
"""Native MOSS delay generator, adapted from pinned VoiceGenerator 97521ec2."""
import torch

import comfy.model_management
import comfy.utils
from .llama import BaseLlama, Llama2_, Qwen3_8BConfig


PAD, IM_START, IM_END = 151643, 151644, 151645
AUDIO_START, AUDIO_END, AUDIO_GEN, AUDIO_DELAY = 151652, 151653, 151656, 151662
AUDIO_PAD = 1024


def sample_delay_token(logits, generator, do_sample=True, top_k=50, top_p=1.0, history=None, penalty=1.0):
    # The reference boolean-indexes audio heads before this operation, so its
    # repetition history is flattened across those heads (and batch items).
    if history is not None and penalty != 1.0:
        indices = torch.unique(history)
        selected = logits[:, indices]
        logits[:, indices] = torch.where(selected > 0, selected / penalty, selected * penalty)
    if not do_sample:
        return logits.argmax(dim=-1)
    if top_k > 0:
        values, indices = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        logits = torch.full_like(logits, -torch.inf).scatter_(-1, indices, values)
    if top_p < 1.0:
        probs, indices = torch.softmax(logits, dim=-1).sort(descending=True, dim=-1)
        remove = probs.cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.masked_fill(torch.zeros_like(remove).scatter_(-1, indices, remove), -torch.inf)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(-1)


class DelayPattern:
    def __init__(self, input_ids):
        self.codebooks = input_ids.shape[-1] - 1
        text = input_ids[..., 0].tolist()
        self.audio_lengths = []
        self.in_audio = []
        self.assistant_offsets = []
        for row in text:
            start = max((i for i, token in enumerate(row) if token == AUDIO_START), default=-1)
            continuation = row[-1] in (AUDIO_START, AUDIO_GEN) and start >= 0
            self.audio_lengths.append(len(row) - start if continuation else 0)
            self.in_audio.append(continuation)
            self.assistant_offsets.append(max((i for i, token in enumerate(row) if token == IM_START), default=-1) + 3)
        self.delays = [None] * len(text)
        self.stopped = [False] * len(text)

    def step(self, logits, history, step, generator, text_temperature=1.5, text_top_p=1.0, text_top_k=50,
             audio_temperature=1.5, audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.1):
        logits = [value / (text_temperature if text_temperature > 0 else 1) if i == 0
                  else value / (audio_temperature if audio_temperature > 0 else 1) for i, value in enumerate(logits)]
        text = [PAD] * len(self.stopped)
        sample_rows = []
        for i, (stopped, delay) in enumerate(zip(self.stopped, self.delays)):
            if stopped:
                continue
            if delay is not None and delay < self.codebooks:
                text[i] = AUDIO_DELAY
            elif delay == self.codebooks:
                text[i] = AUDIO_END
                self.in_audio[i] = False
            else:
                sample_rows.append(i)
            if self.in_audio[i]:
                allowed = logits[0][i, [AUDIO_GEN, AUDIO_DELAY]].clone()
                logits[0][i].fill_(-torch.inf)
                logits[0][i, [AUDIO_GEN, AUDIO_DELAY]] = allowed
            else:
                logits[0][i, [PAD, AUDIO_GEN, AUDIO_DELAY, AUDIO_END]] = -torch.inf
        if step == 0:
            logits[0][:, AUDIO_DELAY] = -torch.inf
        if step <= self.codebooks:
            logits[0][:, IM_END] = -torch.inf
        selected = sample_delay_token(logits[0][sample_rows], generator, text_temperature > 0, text_top_k, text_top_p).tolist()
        for i, token in zip(sample_rows, selected):
            text[i] = token
        for i, token in enumerate(text):
            if token == AUDIO_START:
                self.in_audio[i] = True
            if token == IM_END:
                self.stopped[i] = True
        audio = torch.full((len(text), self.codebooks), AUDIO_PAD, device=history.device, dtype=torch.int64)
        active = [[q < length and (delay is None or q >= delay) for q in range(self.codebooks)]
                  for length, delay in zip(self.audio_lengths, self.delays)]
        first = [i for i, row in enumerate(active) if row[0]]
        remaining = [(i, q) for i, row in enumerate(active) for q in range(1, self.codebooks) if row[q]]
        if first or remaining:
            first_logits = logits[1][first]
            first_logits[:, AUDIO_PAD] = -torch.inf
            audio[first, 0] = sample_delay_token(first_logits, generator, audio_temperature > 0, audio_top_k, audio_top_p, history[:, :, 1], audio_repetition_penalty)
            rows = [i for i, _ in remaining]
            heads = [q for _, q in remaining]
            other_logits = torch.stack(logits[2:], dim=1)[rows, [q - 1 for q in heads]]
            other_logits[:, AUDIO_PAD] = -torch.inf
            audio[rows, heads] = sample_delay_token(other_logits, generator, audio_temperature > 0, audio_top_k, audio_top_p, history[:, :, 2:], audio_repetition_penalty)
        for i, token in enumerate(text):
            if token in (AUDIO_START, AUDIO_GEN, AUDIO_DELAY):
                self.audio_lengths[i] += 1
            elif token == AUDIO_END:
                self.audio_lengths[i] = 0
            if self.delays[i] is None and token == AUDIO_DELAY:
                self.delays[i] = 0
            if self.delays[i] is not None:
                self.delays[i] += 1
                if self.delays[i] > self.codebooks:
                    self.delays[i] = None
        return torch.cat((torch.tensor(text, device=history.device)[:, None], audio), dim=-1)


class MossTTSDelay(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        language = dict(hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
                        num_attention_heads=16, num_key_value_heads=8, vocab_size=155648, lm_head=False)
        language.update(config_dict.get('language', {}))
        config = Qwen3_8BConfig(**language)
        config.prefetch_dynamic_vbars = True
        self.model = Llama2_(config, dtype=dtype, device=device, ops=operations)
        self.num_layers = config.num_hidden_layers
        self.codebooks = config_dict.get('codebooks', 16)
        self.dtype = dtype
        self.emb_ext = torch.nn.ModuleList([operations.Embedding(1025, config.hidden_size, device=device, dtype=dtype) for _ in range(self.codebooks)])
        self.lm_heads = torch.nn.ModuleList([operations.Linear(config.hidden_size, size, bias=False, device=device, dtype=dtype)
                                            for size in [config.vocab_size, *([1025] * self.codebooks)]])

    def combined_embeddings(self, input_ids):
        x = self.model.embed_tokens(input_ids[..., 0], out_dtype=self.dtype)
        for i, embedding in enumerate(self.emb_ext):
            x = x + embedding(input_ids[..., i + 1], out_dtype=self.dtype)
        return x

    def next_logits(self, input_ids, attention_mask, cache):
        hidden, _, cache = self.model(None, embeds=self.combined_embeddings(input_ids), attention_mask=attention_mask, past_key_values=cache)
        return [head(hidden[:, -1]) for head in self.lm_heads], cache

    def generate_delay(self, input_ids, attention_mask, max_new_tokens, seed, **sampling):
        batch, length, channels = input_ids.shape
        capacity = length + max_new_tokens
        cache = self.model.init_kv_cache(batch, capacity, input_ids.device, self.dtype)
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        history = torch.empty((batch, capacity, channels), dtype=torch.int64, device=input_ids.device)
        history[:, :length] = input_ids
        mask = torch.ones((batch, capacity), dtype=attention_mask.dtype, device=input_ids.device)
        mask[:, :length] = attention_mask
        pattern = DelayPattern(input_ids)
        current = input_ids
        used = length
        progress = comfy.utils.ProgressBar(max_new_tokens)
        for step in range(max_new_tokens):
            comfy.model_management.throw_exception_if_processing_interrupted()
            logits, cache = self.next_logits(current, mask[:, :used], cache)
            current = pattern.step(logits, history[:, :used], step, generator, **sampling)[:, None]
            history[:, used:used + 1] = current
            mask[:, used] = torch.tensor([not stopped for stopped in pattern.stopped], device=mask.device, dtype=mask.dtype)
            used += 1
            progress.update(step + 1)
            if all(pattern.stopped):
                break
        return [(length - start, history[i, start:used].to(device='cpu', copy=True)) for i, start in enumerate(pattern.assistant_offsets)]
