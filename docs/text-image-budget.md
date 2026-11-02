# Image count planning

`TextImageBudget` (Plan Text Image Budget) accepts a complete native generation
CLIP and estimates the largest number of equal-size images, up to
`maximum_images` (1–512), that the existing context/memory planner admits.
Useful for video frame batches and multi-image comparison. Use the normal main
ComfyUI queue: planning loads weights through the ordinary memory owner and can
offload idle models. It performs no vision encoding or text generation.

Supply the downstream `prompt`, `system_prompt`, image `width`/`height`, and
positive `max_length`. `context_budget: 0` uses the model-supported maximum;
positive values only restrict it. `reserve_tokens` (default 1024) reserves extra
space for downstream labels/formatting. `json_schema`, when provided, adds the
same schema instruction as Generate Text. The caller must account for any
different provider formatting and retain the downstream context guard.

The node uses meta tensors to tokenize actual native image geometry and binary
searches feasible counts with a minimum short edge equal to the requested size's
short edge. Native grids may round or upscale; this does not promise exact pixel
area after native preprocessing. A tokenizer without meta-image support fails
explicitly. The first output is `image_count`; the second is `budget_json`, with
native context/memory diagnostics, requested dimensions and reservations, the
count ceiling flag and limitations. Save the JSON through SaveText when using
ordinary history or a workflow adapter. Planning is uncached so each queued call
observes current headroom.

Use the same checkpoint, text, output reservation and image geometry in the
consumer; thinking and MTP are disabled in planning. Match those controls in
generation. Replan for a changed prompt or batch size. A no-fit result raises
ContextOverflowError; other failures are not silently converted to smaller counts.
The result is an estimate, not a memory reservation, universal device maximum,
quality assessment or guarantee against concurrent allocations/fragmentation.

Verification: focused CPU tests cover bounded search, output reservation,
failure propagation, uncached node behavior and Gemma meta-image preparation.
Run `python -m pytest tests-unit/comfy_test/text_image_budget_test.py
tests-unit/comfy_test/text_media_budget_test.py -q`. Check live object-info after
deployment and verify a bounded normal-queue call before claiming activation.
