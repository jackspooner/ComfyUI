"""Context planning for already formatted generation tokens, without vision inference."""
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from comfy.text_generation import ContextOverflowError


def sample_video(video, source_fps=24.0, fps=1.0):
    if not math.isfinite(source_fps) or source_fps <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("source_fps and fps must be finite positive numbers.")
    if video.ndim != 4 or video.shape[0] == 0:
        raise ValueError("Video requires a nonempty IMAGE batch [frames, height, width, channels].")
    # Choose the frame at each sampling instant; never duplicate source frames.
    rate = min(fps, source_fps)
    if rate == source_fps:
        indices = list(range(video.shape[0]))
    else:
        count = math.ceil(video.shape[0] * rate / source_fps)
        indices = [math.floor(i * source_fps / rate) for i in range(count)]
        indices = [index for index in indices if index < video.shape[0]]
    return video[indices], [i / source_fps for i in indices]


def timestamp(seconds):
    minutes, milliseconds = divmod(round(seconds * 1000), 60000)
    whole, fraction = divmod(milliseconds, 1000)
    return f"{minutes:02d}:{whole:02d}" + (f".{fraction:03d}".rstrip("0") if fraction else "")


def prepare_generation_tokens(tokens, model_limit, context_budget, max_length, count_tokens, image_geometry,
                              auto_resize=False, memory_budget=None, memory_estimate=None):
    if type(model_limit) is not int or model_limit <= 0:
        raise ValueError("This model does not expose a supported generation context limit.")
    if type(context_budget) is not int or not 0 <= context_budget <= model_limit:
        raise ValueError(f"context_budget must be 0 or between 1 and {model_limit} tokens.")
    if type(max_length) is not int or max_length < 0:
        raise ValueError("max_length must be a nonnegative integer.")
    limit = context_budget or model_limit
    reserve = max_length or 1
    if isinstance(tokens, dict):
        if len(tokens) != 1:
            raise ValueError("Media context planning requires one generative tokenizer stream.")
        key = next(iter(tokens))
        rows = tokens[key]
    else:
        rows = tokens
    images = [(r, i, entry[0]) for r, row in enumerate(rows) for i, entry in enumerate(row)
              if isinstance(entry[0], dict) and entry[0].get("type") == "image"]

    def candidate(budget=None, materialize=False):
        selected = [[(dict(entry[0]), *entry[1:]) if isinstance(entry[0], dict) else entry for entry in row] for row in rows]
        sizes = []
        for r, i, image in images:
            embed = dict(image)
            if budget is not None:
                h, w = image_geometry(image, budget)
                source = image.get("source", image["data"])
                if materialize:
                    pixels = source.movedim(-1, 1)
                    if pixels.shape[-2:] != (h, w):
                        pixels = F.interpolate(pixels.float(), (h, w), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
                    embed["data"] = pixels.movedim(1, -1)[..., :3]
                else:
                    embed["data"] = SimpleNamespace(shape=(source.shape[0], h, w, source.shape[-1]))
                # Geometry now owns the exact patch count; do not pad to an old budget.
                embed.pop("max_soft_tokens", None)
            embed.pop("source", None)
            embed.pop("expected_length", None)
            selected[r][i] = (embed, *rows[r][i][1:])
            sizes.append(list(embed["data"].shape[1:3]))
        value = {key: selected} if isinstance(tokens, dict) else selected
        count = count_tokens(value)
        memory = memory_estimate(value, count, reserve) if memory_estimate is not None else None
        return value, count, sizes, memory

    def fits(count, memory):
        return count + reserve <= limit and (memory_budget is None or memory <= memory_budget)

    budget = None
    if auto_resize and images:
        low, high = 1, limit
        _, smallest, _, memory = candidate(low)
        if not fits(smallest, memory):
            raise ContextOverflowError(f"Media cannot fit: minimum image sizes use {smallest} input tokens plus {reserve} output tokens within {limit}; estimated working memory {memory} bytes, available budget {memory_budget}. Reduce frames, text or output tokens.")
        # A shared per-image ceiling lets small/source-limited images leave room for larger ones.
        while low < high:
            middle = (low + high + 1) // 2
            _, count, _, memory = candidate(middle)
            if fits(count, memory):
                low = middle
            else:
                high = middle - 1
        budget = low
    selected, count, sizes, memory = candidate(budget, materialize=True)
    if count + reserve > limit:
        raise ContextOverflowError(f"Context overflow: input uses {count} tokens and needs {reserve} output tokens within {limit}. Enable auto_resize or reduce input/output tokens.")
    if memory_budget is not None and memory > memory_budget:
        raise ContextOverflowError(f"Text generation needs an estimated {memory} working bytes; the available budget is {memory_budget}. Reduce input or output tokens.")
    output = max_length or limit - count
    if max_length == 0 and memory_budget is not None:
        low, high = 1, output
        while low < high:
            middle = (low + high + 1) // 2
            if memory_estimate(selected, count, middle) <= memory_budget:
                low = middle
            else:
                high = middle - 1
        output = low
        memory = memory_estimate(selected, count, output)
    vram_limit = None
    if memory_budget is not None:
        low, high = count, limit - output
        while low < high:
            middle = (low + high + 1) // 2
            if memory_estimate(selected, middle, output) <= memory_budget:
                low = middle
            else:
                high = middle - 1
        vram_limit = low + output
    return selected, {
        "version": 1, "model_context_limit": model_limit, "context_limit": vram_limit or limit,
        "requested_context_limit": limit,
        "input_tokens": count, "output_allowance": output,
        "excluded_message_ranges": [], "auto_resize": auto_resize,
        "image_sizes": sizes, "image_token_ceiling": budget,
        "memory_budget_bytes": memory_budget, "estimated_working_bytes": memory,
        "vram_context_limit": vram_limit,
    }
