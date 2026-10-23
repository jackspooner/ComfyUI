# Text media sizing acceptance — 2026-10-23

Source `fa68cb587` added the controls/planner. Serving checkout `a84d2544a` also
contained the separately integrated video-component extraction changes. The
ComfyUI user service was restarted only after idle queue and Workflows5 checks;
live `/object_info/TextGenerate` advertised `fps`, `source_fps` and `auto_resize`.
The existing flags, model roots and managed GPU ownership remained active.

Hardware: NVIDIA RTX 5090, 32,607 MiB reported VRAM. Native normal-queue tests used
the installed Gemma4 12B int8 convrot and Qwen3-VL 4B BF16 checkpoints. Each supplied
a static 640×360 blue-circle/orange-square fixture repeated at 24 FPS, selected at
`fps=4`, with `auto_resize=true`, greedy decoding, thinking/MTP disabled, and
`max_length=64`. All completed successfully with zero OOM retries and preserved
all 64 reserved output tokens. Captions identified the visible blue circle and
orange square. Repeated static frames do not establish motion-description quality.

| Case | Prompt ID | Frames | Input tokens | Effective context | Selected frame size (W×H) |
| --- | --- | ---: | ---: | ---: | --- |
| Gemma4, 8192-token cap | `deccfd10-7d80-4224-b332-6d77b01117fd` | 4 | 4481 | 8192 | 2112×1200 |
| Gemma4, no user token cap | `af213921-7f6c-4b31-8a87-2fd5012681a2` | 32 | 12869 | 13016 | 1248×720 |
| Gemma4, 384-token cap | `ccc57776-012b-45ef-9396-e50103cb2da1` | 4 | 281 | 384 | 480×240 |
| Qwen3-VL, 1024-token cap | `effaf463-46b0-48c0-bc08-9533ba10964a` | 4 | 926 | 1024 | 608×352 |

The uncapped Gemma4 run demonstrated memory-driven sizing: model maximum 131072,
effective estimate 13016, measured working budget about 16.33 GB and estimated
working requirement 16.16 GB. Its 32 frames remained present. The 384-token run
instead followed the explicit token cap. Native Gemma4 video markers and the
generic Qwen3-VL timestamped-image path both retained timestamps 0, .25, .5 and .75
seconds in the first second, with no second subsampling step.

These are current-headroom estimates under normal dynamic loading. Staged weights
need not all be resident at the time of measurement. Caches, managed resident
weights and subsequent dynamic eviction make budgets differ across runs; these
numbers are not permanent model capacities or proofs of maximum VRAM utilization.

Task-local reproduction script, exact graphs, submission receipts, history and
saved context/caption files are retained under
`output/text-media-sizing-acceptance/`; the fixture is under
`input/text-media-sizing-acceptance/`. `probe.py inspect` reads existing IDs and
checks frame counts, preserved output reservations, token ceilings and estimated
working memory. Do not resubmit its retained receipt as a new inference request.

The selected text/history/media/vision/attention/audio regression suite passed
274 tests with four external-reference skips. After concurrent integration,
the combined video-extraction and media-sizing suites passed 48 tests. Python
compilation and `git diff --check` passed. Ruff was unavailable in the prepared
runtime. Shared skill package validation, its 19 production tests, reconciliation
and discovery doctor passed.

Unit tests cover bounded OOM retry, fractional FPS, fixed-resolution failure,
metadata-only exact Gemma4 cache-size comparisons, hybrid/MTP accounting,
whole-turn exclusion and measured-headroom delegation. No real OOM was deliberately
forced. Other checkpoints, CPU/RAM placement, audio planning, attention backends,
long natural videos and full-context quality have no new live acceptance here.

## Minimum short edge acceptance

Serving revision `4348828a8` adds `min_short_edge`, default 256 pixels, including
OOM replanning. After an idle queue/Workflows5 check and managed restart, live
discovery reported `auto_resize=true` and the 256-pixel default. Using the same
Gemma4 int8 fixture, 24 source frames at 24 FPS and `fps=4` retained four frames
at 0, .25, .5 and .75 seconds, with all 64 output tokens reserved:

| Token cap | Prompt ID | Result |
| --- | --- | --- |
| 8192 | `b230ad7d-3b5c-46ab-9fd9-45317ca14d5a` | Four 2112×1200 frames; 4480 input tokens, 47 generated tokens |
| 384 | `0211aacf-7326-4541-a48a-520e011de659` | Four 480×288 frames; 320 input tokens, 18 generated tokens |
| 256 | `f3f2c926-a2b4-465c-ae9d-5c1703c998a3` | ContextOverflowError before generation: minimum media needs 320 input + 64 output tokens |

The earlier 384-token result above predates the floor and used a different prompt;
it does not describe the current default. The 256-token rejection retained the
floor and advised lower FPS, a shorter clip, fewer text/output tokens or an explicit
floor override. No real OOM was induced; unit tests verify retries retain the floor
and stop before another generation when it cannot fit. The combined selected
text/history/vision/audio/video regressions passed 308 tests with four skips in
both the task and integration worktrees.

RouterText revision `78fbc3e` also accepted an image call's typed sizing diagnostics,
reporting 1141 input/12 generated tokens and 64 reserved output tokens under an
effective 13315-token memory window versus the 131072-token model/request ceiling.
Prompt `c1979b0b-5118-4314-aada-31d5c94d04dc` returned the fixture's blue circle and
orange square. RouterText's build, 410 tests (one skip), lint, formatting and API
freshness checks passed in both worktrees. Exact task-local graphs, receipts,
context files and the RouterText response are retained under
`output/text-media-minimum-acceptance/`. These bounded static-image checks do not
establish natural-video understanding, exact maximum VRAM use or all model families.
