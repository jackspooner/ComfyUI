# Generate Text: video sampling and memory-aware sizing

This fork extends the existing **Generate Text** node (`TextGenerate`), retaining
its five output indices. The node's `define_schema` owns `/object_info/TextGenerate`;
there is no new HTTP route, remote request or model download. Discover the live
schema before authoring workflows. Source integration and live activation are
separate checks.

## Controls and examples

| Control | Meaning |
| --- | --- |
| `fps` | Frames analysed per source second; default 1. `4` means one frame every 0.25 seconds. |
| `source_fps` | Actual playback rate of the connected IMAGE frame batch; default 24. IMAGE tensors do not retain FPS metadata. Use the loader's output rate, including upstream resampling. |
| `auto_resize` | Default true selects the largest supported common per-image token ceiling under estimated device memory and token constraints. Explicit false preserves model-default resolution. Saved explicit false values remain false; omitted API inputs use true. Adapters without automatic planning support require false. |
| `min_short_edge` | Default 256 pixels. Every automatically sized image/frame must have a shorter side at least this large, rounded through model geometry. OOM retries retain this floor. Zero disables it. Ignored when `auto_resize=false`. |
| `max_length` | Output reservation, including thinking. A positive value is preserved even when media shrinks. Zero prioritizes media and uses the remaining affordable output capacity, with at least one token; use a positive reservation for useful captions. |
| `context_budget` | Optional additional total input/output token cap. Zero uses the model's configured maximum. It cannot raise the model's maximum or supply missing model capability information. |

For a 24 FPS frame batch, `source_fps=24, fps=4` selects indices 0, 6, 12, 18,
24, etc. Nonintegral rates use the source frame at each sampling instant; timestamps
reflect actual selected indices divided by source FPS. Rates above source FPS
select every source frame without duplication. Do not feed an already sampled
1 FPS batch with `source_fps=24`.

Example API node inputs (ordinary graph links for CLIP and IMAGE):

```json
{
  "class_type": "TextGenerate",
  "inputs": {
    "clip": ["model", 0], "video": ["frames", 0],
    "prompt": "Describe visible subjects, actions and camera movement chronologically. Do not invent audio.",
    "source_fps": 24, "fps": 4, "auto_resize": true, "min_short_edge": 256,
    "context_budget": 16384, "max_length": 512,
    "sampling_mode": "off", "thinking": false, "use_default_template": true
  }
}
```

Gemma4 retains native video markers and fractional timestamps. Other complete
image-chat adapters receive timestamped ordered image parts through their existing
chat formatter. Those adapters require the default template and no AUDIO input;
they are frame-based video descriptions, not a new temporal vision encoder.
Unsupported image/chat adapters fail explicitly instead of silently ignoring video.
The specialized **Generate LTX2 Prompt** schema retains its existing controls and
two outputs; it does not expose these new controls.

## Planning and model boundaries

Token limits come from the loaded generative model configuration, not the node's
UI maximum. Counts include actual formatted text/role/thinking markers and expanded
visual tokens. With `auto_resize` enabled, the CLIP owner loads through ordinary
model management, releases idle other models through that owner and measures free
device memory after the selected model's managed loading/staging. It subtracts the
memory owner's existing inference reserve. Currently resident, quantized or
offloaded weights affect this measurement; dynamic loading may stage rather than
fully populate GPU weights, and the existing owner retains eviction/reload policy.
Weights, KV caches and execution dtype are separate costs.

For text-only planning with a positive output reservation, adapters that expose
the generation-memory estimate now pass it to ordinary model loading before
measuring headroom. Qwen3.5/3.8 supplies that estimate for its hybrid cache and
workspace, including supported MTP state. This lets the existing offload owner
make room instead of rejecting a request solely because weights were loaded
without its output budget. The inference reserve remains separate, and planning
still rejects a request when measured space is insufficient. Raw media and
automatic output (`max_length=0`) retain their existing sizing order; their full
unplanned estimate is not reserved before sizing. Qwen generation also uses the
existing final working-memory check after sizing.

Isolated tests cover the 7,723-input / 20,000-output admission regression and
both successful and unavailable owner reservations. They do not establish that
the corresponding live Context-IR request fits the installed GPU; activation
and explicit model acceptance remain separate.

The model supplies cache/workspace estimates. Standard autoregressive caches,
Gemma4 sliding/global and shared KV caches, Qwen3.5 hybrid recurrent/full-attention
state and enabled MTP cache capacity are distinguished. Prefill/MLP, vision/audio
workspace and a temporary-work allowance are estimated conservatively. Planning
uses shape metadata, without vision inference or repeated trial generations.
CPU CLIP placement measures available RAM rather than claiming a VRAM budget.

The planner searches for the largest shared per-image token ceiling that fits
the measured working-memory budget and model/user token cap, then resizes pixels
once. Source-limited images leave space for larger images. It keeps the selected
frames and requested positive output reservation; it never silently lowers FPS,
shortens text, drops frames or reduces `max_length`. Explicit `drop_oldest_turns`
can still exclude complete old conversation pairs when the minimum media cannot
fit; the returned history and original image snapshots remain complete.

The default `min_short_edge=256` constrains both dimensions of every planned image
or frame to at least 256 pixels through the model's existing grid geometry. The
planner first finds the smallest common token ceiling that reaches this floor,
then maximizes resolution within the remaining limits. It does not manually round
or distort aspect ratios outside the model's preprocessing rules. Non-grid values
such as 300 may therefore produce a short edge of 320 or 336 pixels.

If the floor cannot fit, planning fails before materializing resized pixels or
running generation. Lower FPS, shorten/segment the clip, reduce text/output tokens
or explicitly lower the floor. Source/model limits are not overridden: a small
Qwen source that the native processor cannot enlarge to 256 fails clearly too.
This is a minimum input resolution, not a guarantee of recognizable detail in a
small or blurry original. `0` explicitly restores the native technical minima.
`auto_resize=false` bypasses this floor. Lower-level model preparation retains a
zero default for compatibility; the Generate Text node owns the 256-pixel default.

Implemented variable geometry: Qwen2.5-VL, Qwen3-VL, Qwen3.5/3.8, Qwen3-Omni,
Gemma4 and GLM-OCR. Qwen respects its native processor pixel/grid limits and
source detail. Gemma4 can use up to 1120 visual tokens per image/frame, including
the original pixels before the legacy 280/70-token resize. GLM-OCR retains its
one-megapixel/padded-grid path and task-specific limitations. Gemma3's fixed
visual token count cannot be reduced by shrinking pixels. Audio accounting and
workspace planning are implemented for Gemma4 and Qwen3-Omni; other audio adapters
reject automatic planning until they expose reliable counters/estimates.
Legacy audio with `auto_resize=false` keeps ignoring conversation budget controls.
Audio/video history remains unsupported. No hidden session or persistent tensor
cache is added. No model's trained context or vision capacity is extended.

Memory estimates are not a proof of the exact hardware maximum. Backend workspaces,
fragmentation, custom kernels and concurrent external allocations can change the
peak. The conservative window can leave spare VRAM. On an actual device OOM,
`auto_resize` clears failed-call references and unused allocator cache, halves the
working budget relative to the previous estimate and replans from original pixels,
up to three retries. It preserves `min_short_edge`, output allowance, sampling options and source
frames, and reports retries. No failed attempt publishes history/snapshots. If the
minimum media/output cannot fit or resolution cannot decrease, the call fails
with an actionable error. It never restarts services or retries an uncertain
queue submission. Non-OOM errors and cancellation propagate.

## Results and verification

`context_info` remains version 1 with its existing required fields. Planned calls
also report `requested_context_limit`, `vram_context_limit`, `memory_budget_bytes`,
`estimated_working_bytes`, `memory_device`, `memory_kind`, `auto_resize`,
`image_token_ceiling`, `image_sizes` (height/width), `min_short_edge`, and `oom_retries`.
`context_limit` is the lower logical/memory-derived estimate for this request;
`model_context_limit` remains the model maximum. The memory-derived limit depends
on selected media, output reservation and current headroom; do not reuse it as a
universal model specification. Video adds source/sample counts, source/requested
effective FPS and exact selected frame timestamps. Token-only planning has no
memory budget and reports null memory fields. Save `context_info` explicitly
with SaveText if a durable receipt is needed.

Unit verification:

```bash
COMFYUI_PYTHON=/path/to/prepared/python python3 scripts/setup_worktree.py
/path/to/prepared/python -m pytest tests-unit/comfy_test/text_media_budget_test.py tests-unit/comfy_test/image_history_test.py tests-unit/comfy_test/text_history_test.py tests-unit/comfy_test/text_generation_length_test.py -q
```

Tests cover fractional sampling, duplicate avoidance, exact grid counting,
memory-constrained sizing, quality-floor rejection, output preservation, fixed-resolution failure, original
pixels, metadata-only cache allocation checks, history exclusion and OOM recovery.
Live model/device acceptance is recorded separately; unit fixtures do not establish
all checkpoints, attention backends, maximum lengths or caption accuracy.
See [RTX 5090 acceptance](text-media-sizing-acceptance.md) for the tested revision,
prompt IDs, measured budgets and remaining coverage limits.
