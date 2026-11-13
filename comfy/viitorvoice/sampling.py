# Adapted from viitor-ai/viitor-voice-nar 40359993534de0980eefff2d5a25bddc30f6bf58.
# Apache-2.0; see LICENSE.
"""Masked local-edit refinement with child-private tokens, schedules and RNG."""
import math

import torch
from torch.nn import functional as F

import comfy.model_management as mm
from .text import (estimate_ref_text_mask_len, insert_text_pause_anchors, make_mask_text,
                   normalize_nvv_tags, prepare_text_for_tokenizer, resolve_language,
                   tokenize_with_special_tokens, wrap_prompt_target_text)


def conditioning(tokenizer, config, text, target, reference, language):
    language = resolve_language(language)
    style = f'<|lang_start|>{language or "None"}<|lang_end|><|instruct_start|>None<|instruct_end|>'
    text = insert_text_pause_anchors(prepare_text_for_tokenizer(normalize_nvv_tags(text).strip(), tokenizer, language))
    if reference is not None:
        length = estimate_ref_text_mask_len(language, reference.shape[-1], config['audio_frame_rate'],
            config.get('no_ref_text_tokens_per_second', {}), config.get('no_ref_text_min_mask_tokens', 1),
            config.get('no_ref_text_max_mask_tokens', 256), config.get('no_ref_text_default_tokens_per_second', 4.))
        text = wrap_prompt_target_text(make_mask_text(length), text)
    def tokenize(value):
        return tokenize_with_special_tokens(value, tokenizer).repeat(target.shape[0], 1)
    parts = [tokenize(style), tokenize('<|text_start|>'+text+'<|text_end|>')]
    audio_start = sum(part.shape[-1] for part in parts)
    if reference is not None:
        parts.extend((reference.cpu(), torch.tensor(config['audio_separator_ids'])[:, None]))
    parts.append(target.cpu())
    ids = torch.cat(parts, -1)
    audio_mask = torch.arange(ids.shape[-1]) >= audio_start
    return ids, audio_mask


def schedule(total, steps, shift, device):
    # Match the pinned helper's num_step+1 argument and its own extra endpoint.
    times = torch.linspace(0, 1, steps+2, device=device)
    times = (shift*times/(1+(shift-1)*times)).tolist()
    remaining, result = total, []
    for step in range(steps):
        count = remaining if step == steps-1 else min(math.ceil(total*(times[step+1]-times[step])), remaining)
        result.append(count)
        remaining -= count
    return result


def gumbel(values, temperature, generator):
    scaled = values/temperature
    uniform = torch.rand(scaled.shape, dtype=scaled.dtype, device=scaled.device, generator=generator)
    return scaled-torch.log(-torch.log(uniform+1e-10)+1e-10)


def predict(conditional, unconditional, cfg, mask_id, separator_id, temperature, generator):
    probability = F.log_softmax(conditional.float(), -1)
    if cfg:
        probability = probability+cfg*(probability-F.log_softmax(unconditional.float(), -1))
    probability = F.log_softmax(probability, -1)
    probability[..., mask_id] = -torch.inf
    probability[..., separator_id] = -torch.inf
    if temperature:
        count = math.ceil(probability.shape[-1]*0.1)
        values, indices = probability.topk(count, dim=-1)
        filtered = torch.full_like(probability, -torch.inf).scatter_(-1, indices, values)
        tokens = gumbel(filtered, temperature, generator).argmax(-1)
    else:
        tokens = probability.argmax(-1)
    return tokens, probability.max(-1).values


def generate(model, tokenizer, config, target, editable, reference, text, language,
             steps, cfg, seed, options, progress, position_temperature=1., class_temperature=0.,
             t_shift=0.1, layer_penalty=5.):
    if not 1 <= steps <= 200 or not math.isfinite(cfg) or cfg < 0:
        raise ValueError('ViiTor generation requires 1 to 200 steps and a finite nonnegative guidance scale')
    if not 0 <= seed <= 0xffffffffffffffff or isinstance(seed, bool):
        raise ValueError('ViiTor seed must be an unsigned 64-bit integer')
    # Dynamic patchers may leave file-backed parameters on CPU. The private
    # input's device comes from the patcher's load_device at the caller.
    target = target.clone()
    device = target.device
    editable = editable.to(device=device, dtype=torch.bool)
    if editable.ndim != 1 or editable.numel() != target.shape[-1] or not editable.any():
        raise ValueError('ViiTor editable mask must select frames in the target draft')
    mask_ids = torch.tensor(config['audio_mask_ids'], device=device)[:, None]
    target = torch.where(editable[None], mask_ids, target)
    ids, audio = conditioning(tokenizer, config, text, target, reference, language)
    length, frames, books = ids.shape[-1], target.shape[-1], target.shape[0]
    branches = 2 if cfg else 1
    batch_ids = mask_ids[None].expand(branches, books, length).clone()
    batch_ids[0] = ids.to(device)
    audio_mask = torch.zeros(branches, length, device=device, dtype=torch.bool)
    audio_mask[0] = audio.to(device)
    allowed = torch.ones(branches, 1, length, length, device=device, dtype=torch.bool)
    if cfg:
        batch_ids[1, :, :frames] = target
        audio_mask[1, :frames] = True
        allowed[1] = False
        allowed[1, :, :frames, :frames] = True
        diagonal = torch.arange(frames, length, device=device)
        allowed[1, :, diagonal, diagonal] = True
    tokens = target[None].clone()
    pending = int(((tokens == mask_ids)&editable[None, None]).sum())
    counts = schedule(pending, steps, t_shift, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    layers = torch.arange(books, device=device)[None, :, None]
    pause_ids = torch.tensor(config['audio_pause_ids'], device=device)
    for step, count in enumerate(counts):
        mm.throw_exception_if_processing_interrupted()
        semantic, acoustic = model(batch_ids, audio_mask, allowed, options)
        predictions, confidence = [], []
        for book, logits in enumerate((semantic[:, None], acoustic)):
            conditional = logits[0:1, :, -frames:]
            unconditional = logits[1:2, :, :frames] if cfg else None
            choice, score = predict(conditional, unconditional, cfg, config['audio_mask_ids'][book],
                config['audio_separator_ids'][book], class_temperature, generator)
            predictions.append(choice)
            confidence.append(score)
        predictions, scores = torch.cat(predictions, 1), torch.cat(confidence, 1)-layers*layer_penalty
        if position_temperature:
            scores = gumbel(scores, position_temperature, generator)
        scores.masked_fill_(~editable[None, None], -torch.inf)
        scores.masked_fill_(tokens != mask_ids, -torch.inf)
        available = int(torch.isfinite(scores).sum())
        if not available:
            break
        count = min(count, available)
        if count:
            selected = scores.flatten().topk(count).indices
            flat = tokens.flatten().clone()
            flat[selected] = predictions.flatten()[selected]
            tokens = flat.reshape_as(tokens)
            semantic_selected = selected[selected < frames]
            pause_frames = semantic_selected[predictions[0, 0, semantic_selected] == pause_ids[0]]
            tokens[:, :, pause_frames] = pause_ids[None, :, None]
            tokens = torch.where(editable[None, None], tokens, target[None])
            batch_ids[0, :, -frames:] = tokens[0]
            if cfg:
                batch_ids[1, :, :frames] = tokens[0]
        progress(step+1, steps)
    if ((tokens == mask_ids)&editable[None, None]).any():
        raise RuntimeError('ViiTor local edit ended with unfilled mask tokens')
    return tokens[0].cpu()
