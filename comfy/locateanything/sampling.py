# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# Adapted from nvidia/LocateAnything-3B; NVIDIA License, see LICENSE.
"""Pinned hybrid/MTP pattern decoding, with request-private sampling state."""
import torch

TOKENS = dict(box_start=151668, box_end=151669, coord_start=151677, coord_end=152677,
              ref_start=151672, ref_end=151673, none=4064, null=152678, end=151645, mask=151676)


def sample(logits, history, generator, mode):
    scores = logits.clone()
    used = history.unique()
    selected = scores[..., used]
    scores[..., used] = torch.where(selected > 0, selected/1.1, selected*1.1)
    scores = scores/.7
    ordered, indices = scores.sort(descending=True, dim=-1)
    remove = ordered.softmax(-1).cumsum(-1) > .9
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    scores.masked_fill_(torch.zeros_like(remove).scatter_(-1, indices, remove), torch.finfo(scores.dtype).min)
    probabilities = scores.softmax(-1)
    values = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator).reshape(scores.shape[:-1])[0]
    if scores.shape[1] == 1:
        return values.tolist()
    p = probabilities[0]
    box = decode_box(p, mode)
    if box is not None:
        return box
    if p[0, TOKENS['ref_start']] >= .6:
        candidates = p[1:].topk(5, dim=-1).indices
        valid = (candidates < TOKENS['coord_start']) | (candidates > TOKENS['coord_end'])
        if valid.any(-1).all():
            return [TOKENS['ref_start'], *candidates.gather(-1, valid.long().argmax(-1, keepdim=True)).flatten().tolist()]
    return values.tolist()


def decode_box(p, mode):
    t = TOKENS
    if (p[0, t['box_start']] >= .7 and p[1, t['none']] > .2 and p[2, t['box_end']] > .2
            and p[3, t['null']] > .1 and p[4, t['null']] > .1):
        return [t['box_start'], t['none'], t['box_end'], t['null'], t['null'], t['null']]
    if p[5, [t['box_end'], t['null'], t['end']]].sum() < .2:
        return None
    probabilities, candidates = p[1:5].topk(4, dim=-1)
    valid = (candidates >= t['coord_start']) & (candidates <= t['coord_end'])
    if not valid.any(-1).all():
        return None
    first = valid.long().argmax(-1, keepdim=True)
    coords = candidates.gather(-1, first).flatten()
    if mode == 'hybrid':
        spread = candidates.masked_fill(~valid, -999999).max(-1).values-candidates.masked_fill(~valid, 999999).min(-1).values
        abnormal = (probabilities.gather(-1, first).flatten() < .9) & (valid.sum(-1) > 1) & (spread > 60)
        coords = coords.masked_fill(abnormal, 0)
    return [t['box_start'], *coords.tolist(), t['box_end']]


def pattern(values, mode):
    t = TOKENS
    if values[0] in (t['null'], t['end']):
        return 'end', [t['end']]
    if values[:2] == [t['box_start'], t['none']]:
        return 'box', [t['box_start'], t['none'], t['box_end']]
    if values[0] == t['box_start']:
        index = 1
        for value in values[1:5]:
            if not t['coord_start'] <= value <= t['coord_end']:
                break
            index += 1
        if index == 5 and values[5] == t['box_end']:
            return 'box', values
        if index == 3 and values[3] == t['box_end']:
            return 'box', values[:4]
        return ('box', values) if mode == 'fast' else ('error', values[:index])
    if t['null'] in values:
        values = values[:values.index(t['null'])]
    if values[-2:] == [t['ref_end'], t['ref_end']]:
        values = values[:-1]
    return 'ref', values


def generate(model, ids, visual, tokenizer, max_new_tokens, seed, mode, options, progress):
    t = TOKENS
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    initial = ids.shape[1]
    limit = min(16384, initial+max_new_tokens)
    cache = model.model.init_kv_cache(1, limit+6, ids.device, visual.dtype)
    mtp = mode != 'slow'
    first = True
    while ids.shape[1] < limit:
        past = cache[0][2] if cache[0] else 0
        prepared = ids
        if mtp:
            prepared = torch.cat((ids, ids[:, -1:], torch.full((1, 5), t['mask'], device=ids.device, dtype=ids.dtype)), -1)
        positions = torch.arange(past, prepared.shape[1], device=ids.device)[None]
        if mtp:
            positions[:, -6:] -= 1
        logits = model(prepared[:, past:], positions, cache, mtp, options, visual=visual if first else None)
        # Roll back speculative positions; keep the ordinary prefix under its
        # original position IDs. Each request owns all cache tensors.
        keep = ids.shape[1]
        cache = [(key, value, keep) for key, value, _ in cache]
        values = sample(logits, ids, generator, mode)
        if mtp:
            kind, values = pattern(values, mode)
        else:
            value = values[0]
            kind = 'end' if value == t['end'] else 'ar'
            if mode == 'hybrid':
                kind = 'box_end' if value == t['box_end'] else 'ar' if t['coord_start'] <= value <= t['coord_end'] or value == t['none'] else 'end'
        ids = torch.cat((ids, torch.tensor([values], device=ids.device, dtype=ids.dtype)), -1)
        first = False
        progress(len(values))
        if kind == 'end':
            break
        if mode == 'hybrid':
            mtp = False if kind == 'error' else True if kind == 'box_end' else mtp
    return tokenizer.decode(ids[0, initial:].tolist(), skip_special_tokens=False)
