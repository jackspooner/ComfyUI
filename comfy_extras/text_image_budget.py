"""Plan a bounded image count using the selected CLIP's native context planner."""
import torch

from comfy.text_generation import ContextOverflowError


def plan_image_count(clip, prompt, system_prompt, width, height, maximum_images,
                     max_length, context_budget=0, reserve_tokens=1024):
    for name, value, minimum, maximum in (
        ("width", width, 1, 16384), ("height", height, 1, 16384),
        ("maximum_images", maximum_images, 1, 512),
        ("max_length", max_length, 1, 1048576),
        ("context_budget", context_budget, 0, 1048576),
        ("reserve_tokens", reserve_tokens, 0, 65536),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")

    # A meta tensor exercises the tokenizer's real image geometry without allocating
    # image pixels or running the vision encoder. Unsupported tokenizers fail here.
    image = torch.empty((1, height, width, 3), device="meta")

    def candidate(count):
        content = [{"type": "text", "text": prompt}]
        for _ in range(count):
            content.append({"type": "image", "data": image})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        tokens = clip.tokenize_chat(messages, thinking=False)
        _, info = clip.prepare_generation_tokens(
            tokens, context_budget, max_length + reserve_tokens, True,
            mtp=False, min_short_edge=min(width, height),
        )
        return info

    low, high = 0, maximum_images
    rejected = None
    while low < high:
        middle = (low + high + 1) // 2
        try:
            candidate(middle)
        except ContextOverflowError as error:
            rejected = str(error)
            high = middle - 1
        else:
            low = middle
    if low == 0:
        raise ContextOverflowError(f"No image fits at the requested size and output reservation. {rejected}")
    # Headroom is an observation, not a lease. Recheck the chosen candidate rather
    # than returning an earlier estimate after another planning call changed residency.
    info = candidate(low)
    return low, {
        "version": 1, "image_count": low, "maximum_images": maximum_images,
        "count_limit_reached": low == maximum_images,
        "source_width": width, "source_height": height,
        "requested_output_tokens": max_length, "reserved_extra_tokens": reserve_tokens,
        "context": info,
        "limitations": [
            "Estimate under the native planner and observed device headroom; not a memory reservation or an inference guarantee.",
            "No image pixels or vision features were evaluated. Native image geometry may round or upscale the supplied dimensions.",
            "Use the same model, prompts and output budget; include schema instructions and reserve room for frame labels and transport additions.",
        ],
    }
