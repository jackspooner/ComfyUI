# Custom inference console progress

The fork provides `comfy.console_progress.ConsoleProgress` and
`report_progress` for inference without a console indicator. They write through
tqdm and the existing raw-log transport, including the log panel's initial
snapshot and reconnect stream. Carriage returns replace the active row. A scope
leaves one completion, cancellation or failure line; nested scopes reuse it.
This is source functionality, not evidence that a serving instance was updated.

```python
from comfy.console_progress import ConsoleProgress, advance, phase

with ConsoleProgress('My speech model', 'preparing audio', unit='tokens'):
    inputs = prepare_audio(audio)
    phase('decoding')
    for token in decode(inputs):
        advance()  # one actually produced token; EOS allowance is not a total
```

Use `total` only when the exact work is known, for example
`ConsoleProgress('OCR', 'recognizing', total=len(images), unit='images')`.
Unknown generation has a counter and elapsed time, without percentage or ETA.
For a blocking operation, omit the unit/total and name its current phase.
Ordinary updates redraw at most every 0.25 seconds. The scope's heartbeat updates
elapsed time once per second, including while waiting on an opaque service.
Do not create a heartbeat per token. Use a `with` block or the sync/async
`@report_progress('Model', 'phase', unit='iterations')` decorator so exceptions
and cancellation always stop/join the heartbeat and close the row.

`phase('recognizing', total=len(images), unit='images')` starts a new measured
phase at zero. `advance()` increments completed work; it does not set an absolute
position. Keep node `ProgressBar` hooks unchanged; use `update_absolute(step + 1)`
for loop indices. A generation budget is a limit, not a completion total.

## Coverage inventory

| Registered path/family | Console behavior |
| --- | --- |
| Native Whisper transcription and word alignment | Encoder/decoder phases; completed decoder calls across transcription, separate alignment scope |
| Native Qwen ASR / ForcedAligner | Encoding/decoding phases; accepted tokens / completed alignment chunks |
| Native MOSS Delay and Local speech (including compatibility adapters) | Completed global decoding iterations / frames; no percentage of token budget |
| Native Drama 3 | Prefill/generation phases and produced audio frames |
| Native SALMONN, PadCaptioner, InternVideo3 | Preparation/planning/decoding phases and completed decoding iterations |
| Native PaddleOCR | Completed images over exact image count, including microbatch windows |
| Native UniPASE / RVC | Completed restoration/conversion segments over exact segment count |
| SAM3 image detection | Completed images over batch size |
| SAM3 video / SAM3D Body / DA3 / Hunyuan3D / Stable3D / YingMusic extras | Delayed phase fallback; yield to any existing tqdm indicator |
| MoGe perspective/panorama/render, SDPose, SAM3D face expressions | Existing tqdm indicators retained |
| Native diffusion, CLIP/text generation, ACE-Step LM, YuE2 | Existing sampler/token indicators retained; no global ProgressBar console opt-in |
| All external custom-node registrations, V1 and V3 | Delayed phase fallback covering opaque calls, preprocessing, loading and decoding; yield to an existing tqdm indicator |
| Cutlery LocateAnything worker | Parent renders correlated preparing/generating IPC phases; startup is phase-only |
| Retained MOSS Local v1.5 / SoundEffect workers | Parent renders correlated loading/generation/stage IPC phases |
| Cutlery Runtime ASR, LM Studio, Higgs, XVC, AlphaFace, external 3D / legacy adapters | Custom-node phase fallback while waiting on existing service/client; no fabricated remote steps |
| FishAudioS2, VoxCPM2, Timbre Transfer, Breeze, DramaBox, remote/custom preprocessors | Custom-node fallback fills gaps and yields to existing bars; it does not infer token counts from opaque libraries |

The custom-node registration boundary wraps only the inference function, preserving
sync/async signatures and static/class descriptors. The fallback waits one second,
so short utility nodes stay quiet. It yields for the rest of that operation when
another tqdm bar appears. Native extra modules listed above explicitly opt into
the same fallback; standard sampler registration is unchanged. A registered name
is not proof of model assets or successful GPU execution. Unregistered direct calls
need an explicit scope. Existing third-party bar formatting remains its owner's
responsibility.

The fallback does not reserve a second tqdm row. The raw-log interceptor observes
existing carriage-return indicators before they write and yields the scoped row,
including when a short bar starts and finishes between heartbeat ticks.

## Worker forwarding and diagnostics

Workers keep stdout reserved for their existing JSONL protocol. An opted-in
request carries `request_id` and `report_progress: true`; phase frames use
`{"type":"progress","request_id":"…","phase":"generating audio"}`.
Terminal frames remain separate and echo correlation. The parent rejects stale
correlation before forwarding phase frames to `worker_progress`. Never treat
progress as success or renew a request's monotonic deadline when it arrives.
Cancellation polling, model ownership and terminal cleanup stay in their existing
owners. Workers exposing no iteration callback report real phases only.

The application's console logging handler uses `tqdm.write`, so warnings/errors
clear and restore the active row without changing file logging. Avoid per-token
INFO logs or `print` calls. The preparation-log suppression in `model_patcher.py`
remains independent of progress, including admission and full-unload behavior.

Run `tests-unit/comfy_test/console_progress_test.py` on CPU, affected native model
suites and both worker protocol suites. Main-queue GPU inference and actual panel
acceptance are separate rollout checks. Do not run full-checkpoint inference in
a test process, restart active work or replay an uncertain prompt to test display.
