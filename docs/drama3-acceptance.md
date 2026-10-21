# Drama 3 acceptance record

20 October 2026; Linux RTX 5090 (32,607 MiB), Python 3.13.16,
Torch 2.14.1+cu130, Comfy Kitchen 0.2.39. **The candidate fits this machine,
but speech quality acceptance failed. It is not approved for serving activation.**
Dialogue with a laughing cue and automatic long-form generation produced
near-silence, missing speech and length stops. Preserve this precision and the
failure evidence when investigating; do not substitute a runtime or trim history.

## Sources and converted packages

Fish Speech source is pinned to `541566932abbaebcb0a40e562a1f846738383f38`;
Drama 3 assets to `b2b5ee998034b2975720906c815e7fb0e9f007bd`. All eight
shards and required assets passed published LFS/Git hash and inventory checks.
The explicit source-commit archive matched the inspected source tree. Original
codec SHA256 is `74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f`.

| Package | Bytes | SHA256 |
| --- | ---: | --- |
| `text_encoders/drama3_nvfp4.safetensors` | 25,526,202,688 | `5ebf5df4fcc99ca9a5a4b10f5ac1d414880d8964c72bb4bb484b1438a8c5ef7d` |
| `vae/fish_dac_drama3.safetensors` | 782,872,050 | `0ab52383b66addd2b0405e8f816001dbe37313c29ee7ba97be313401925705e6` |

Sidecar manifests retain source/output hashes, tensor mappings, payload checks,
NVFP4 settings and errors for all 80 expert banks. Routed-weight relative RMSE
ranges from 0.08844 to 0.09493. Conversion streamed individual experts without a
complete BF16 model. Final strict real-package loading verified 53 generator and
20 codec managed blocks, NVFP4 layouts, fast-disk ownership and saved provenance.
The final load-only check used CPU with the hardware guard overridden; it was
not a CPU inference test.

## Component and reference evidence

Final selected CPU regressions passed **173 tests**, with four explicitly skipped
GPU cases covered separately by the Blackwell runs described below. The dedicated
Drama 3 CPU subset passed 20 checks. All four editable workflow layouts passed.

The dedicated tests cover upstream attention and DeltaNet prefill/cached decode,
shared/routed experts, normalization, tower alignment, sampling, source integrity,
packing/scales, history, containment and atomic publication failure. Blackwell GPU
checks additionally cover fused DeltaNet decode and a three-layer original FP8
reference with GPU-resident, CPU-cached and uncached streamed layers. The reference
harness was checked against upstream fixtures before use as a baseline.

Real original FP8 plain speech completed with 26 GPU-resident layers and 14
CPU-cached FP8 layers: 111 audio frames, EOS, 129.04 seconds generation. Both its
waveform and the native plain waveform transcribed exactly as “The quarterly
report is due on Wednesday. Please finish it by noon.” Source-codec versus
native FP32 decode of those actual codes had maximum absolute error
`1.341104507446289e-6`, RMSE `1.0143397588535663e-7`; the short reference
encoder fixture matched exactly after identical BF16 weight packaging.
A separate BF16 codec comparison had maximum absolute error `0.0069580078125`,
RMSE `0.001302648801356554`, and 95% encoder-code agreement. BF16 parity is
therefore not exact.

On actual first-layer text activations, BF16 normalization relative RMSE was
0.095%, DeltaNet prefill 0.174%, and cached decode 0.426%. Eight routed experts
showed 10.4–12.7% weight-chain error, 9.3–12.2% incremental activation error,
and 14.0–17.7% combined error. This isolates local error sources, not the cause
of every speech failure: it covers one layer/prompt and excludes upstream FP8
activation error. The original FP8 dialogue comparison was interrupted to release
this task's GPU reservation during serving-service recovery; it is incomplete.
Paired complex speech and human listening/speaker-identity acceptance remain open.

## Normal-queue speech results

The isolated candidate used the normal ComfyUI queue, separate output directory,
disabled custom nodes and native NVFP4/BF16 execution. All audio is mono 44.1 kHz.
Local Qwen3-ASR checks retained transcripts; these are intelligibility evidence,
not a human quality or voice-identity assessment.

| Case | Result |
| --- | --- |
| Plain / warm speech | EOS, 125 / 87 frames; plain transcript matches supplied words |
| Reference cloning | EOS, 80 frames; supplied words recognized; identity acceptance pending |
| English + Chinese | EOS, 115 frames; both languages transcribed correctly |
| Changed-word revision / reload | EOS, Thursday replaces Wednesday; prior take preserved |
| Delivery revision | EOS; words preserved; local audio description reports soft whisper-like delivery |
| Chained revision | EOS; Friday recognized; all earlier token chunks exactly preserved |
| Two-speaker dialogue with `[laughs]` | **Failed:** 1024-frame length stop, 47.55 seconds near-silence, empty transcript |
| Long form | **Failed:** three 1024-frame length stops, 142.66 seconds, 87.9% near-silent windows and substantial omitted text |

The editable Dialogue example now has simpler two-speaker text without the
laughing cue. That revised example has not been executed; do not infer a pass.
Raw clone-reference audio was not retained in this run; encoded reference tokens
are in the saved history. The case runner now saves the source clip in future runs.

## History, failures and lifecycle

Real save/reload and immutable branching passed, including exact preservation of
prior code chunks across one-, two- and three-take histories. Incompatible codec
identity, corrupt version and context overflow failed before generation. Targeted
prefill and decode cancellation returned no audio or updated history; subsequent
generation recovered. Explicit unload/reload and transition to native Qwen3-VL
4B text generation and back to Drama 3 completed through the same normal queue.

The final candidate run contained 14 successful jobs, three expected validation
errors and two targeted cancellations. Cold plain elapsed 42.71 seconds and warm
17.96 seconds. Cold peak process RSS was 7.10 GiB; the cumulative process high-water
mark across later jobs was 8.25 GiB. One-second whole-device sampling reached
28,500 MiB, including display and other processes, and began after the cold run.
Torch allocation counters exclude aimdo-managed model storage and are not whole
pipeline VRAM measurements. Full pipeline completion establishes local memory fit,
not realtime speed or arbitrary-context fit.

An earlier full-load attempt was killed by host OOM because fast-disk staging
ownership was missing. The integration now passes the existing storage owner's
fast-disk result to both patchers; the complete repeated-run suite above then
passed without that OOM. An initial validation-wrapper allocator initialization
failure was also corrected. Both failures are retained alongside successful runs.
The candidate and original-reference processes have exited, and their GPU holds
were released. This task did not restart ComfyUI or activate these source changes.

## Evidence and remaining gates

Local output copies, histories, DSP measurements and ASR results are retained in
`/home/jack/comfyui/output/drama3-native-20261020/`; transcripts are under
`output/router-audio-transcripts/`. Detailed prompt submissions, statuses,
resource samples, failure receipts and output hashes are retained in
`/home/jack/services/runtime/drama3-candidate/`. Original-reference fixtures,
waveforms and quantization measurements are under
`/home/jack/services/runtime/drama3-reference/`. These machine-local evidence
files and converted weights are not committed to Git.

Four editable workflows and matching API templates are installed in
`user/default/workflows/Drama 3/` and pass the layout checker. The installed Reload
example selects the retained plain history; freshly generated templates require
a history selection. Source/component checks, memory fit and lifecycle checks
have evidence. Dialogue/long-form quality, paired complex-reference evaluation,
human listening, speaker continuity and broader robustness remain unaccepted.
Serving activation remains separate from source integration.
