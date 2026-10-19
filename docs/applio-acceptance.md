# Applio / RVC acceptance

Recorded 19 October 2026 on the local RTX 5090 installation. Source integration,
model setup and live queued file conversion are available. **Explicit idle
`/free` unload/reload acceptance remains unrun** because unrelated generation is
active; no active/queued job was cancelled or restarted to obtain that window.
The native source and editable workflow are delivered with that limitation.

## Source and environment

- Applio reference: `5379cf261fa7d8085c3b077aa42dabca7c85310b`, MIT.
- [Pinned model inventory](applio-sources.json): Applio ContentVec and original RVC
  Kiki v1 / Zhanzhan v2 demo voices, both pitch-guided 40 kHz HiFi-GAN.
- Reused native SoulX RMVPE; no duplicate predictor download or external runner.
- ComfyUI source includes the concurrently integrated upstream v0.42.1 release.
  Python 3.13.16, Torch 2.14.1+cu130, NVIDIA driver 615.71.09, optional FAISS CPU
  1.15.1. Its downloaded wheel SHA256 is
  `f2c31b7f2f6647eb76829a5cfe3c398fb9346df9f26b1d4db35269c91eb58c33`.
- Original model/LFS hashes matched. Final converter validated complete native
  inventories, shapes and finite values for both voices and ContentVec, then read
  every saved tensor back bit-exact. Normalization is explicitly materialized in
  FP32. Model manifests and original licenses are retained beside local assets.
- Core code is on `main`; task branch `applio-native`. Implementation commits:
  `a87c529a2`, merge with updated main `ff4b84fe8`, compiler-buffer correction
  `f74489ece`, segment overlap `eecee481a`.

## Verification

`python -m pytest tests-unit/comfy_test/rvc_test.py tests-unit/comfy_test/soulx_singer_test.py -q`:
**27 passed**. Coverage includes relative-key/value attention, FP32 normalization
of half-source weights, pitch quantization, finite exact-neighbor retrieval,
pitch/non-pitch synthesis shapes, incompatible exports, preserved 50 Hz RMVPE
behavior with optional 100 Hz output, and segment duration/overlap continuity.
The complete ComfyUI suite and other accelerator platforms were not run.

`scripts/check_rvc.py` uses the pinned upstream code and real voice checkpoints on
CPU. Both generator waveforms and text-encoder statistics are bit-exact for the
fixed 35-frame feature/F0/seed fixture. With the final CPU-only harness,
ContentVec maximum error is `1.331e-5`, RMS `2.858e-6`, below its fixed `2e-5` gate.
This is component parity, not full long-recording parity with Applio's splitter.
A separate RMVPE comparison found median F0 error 0.0097 Hz on the five-second
source, with a 96 Hz worst edge discrepancy: upstream reflects the final mel
padding while the reused native predictor pads with zeros. RVC's one-second
context crop keeps the prediction boundary outside the emitted core; no claim of
bit-exact whole-pipeline pitch/audio parity is made.

The first GPU run failed allocation compilation. Reusing stable loop buffers and
prefetching complete encoder stages fixed it with the compiler enabled. Failed
and repaired histories are retained. No allocation-compilation or dynamic-VRAM
flag was disabled. Compiler graph breaks/rogues remain diagnostic observations;
they are not presented as zero-allocation or zero-overhead execution.

## Live queued inference and recovery

The source recording is the first five seconds of the installation's existing
`moss_audio_demo.wav`, copied to `input/applio_demo_source.wav`. It is an excerpt,
not a complete utterance. The editable library workflow is
`user/default/workflows/Applio - Native RVC Voice Conversion.json`; its geometry
checker passes. Live `/object_info` exposes the prepared model filenames and the
two native nodes. API graphs use the same controls and links as the editor graph.

| Check | Prompt ID | Result / active execution time |
| --- | --- | --- |
| Kiki final code, index 0.5, seed 42 | `7dc04b1b-debd-4bd7-ac3d-f0c364cd8629` | Success, 15.073 s |
| Kiki repeated execution, seed 43 | `60675c57-1af4-4e40-8d86-cc1c2c260409` | Success, 1.097 s; queued behind unrelated H3 work |
| Zhanzhan v2, no index | `14686453-1394-42a8-a915-edf709559ee7` | Success, 0.715 s |
| Targeted cancellation, 20-second source | `672ad277-9a8c-4c2a-b0a5-fe7ba0c916e6` | `execution_interrupted` in conversion node 3, 1.084 s |
| Recovery after cancellation, seed 44 | `fd126f1d-90fc-4a3f-b380-62e233204f15` | Success, 0.337 s; conversion was executed, not cached |

All successful clips contain 200,000 finite mono samples at 40 kHz, exactly five
seconds. The same native queue ran another model between the first and second
RVC runs; returning to RVC passed without a private residency owner. Runtime GPU
holds were released through their existing owner. Explicit idle memory release
and deterministic reload comparison remain separate pending gates.

The first final-code GPU run sampled total device memory every approximately
0.5 seconds: peak 2,378 MiB, including desktop and service baseline; before/after
were 1,461 / 2,269 MiB. This is a sampled total-device observation for a five-second
input, not a process-isolated maximum, CPU-RAM measurement, or universal VRAM
bound. Warm tests were interleaved with unrelated models, so they do not establish
controlled throughput. No constrained/low-VRAM or multi-hour recording test ran.

## Audio review and limits

Native Audio Flamingo 3 described the final Kiki clip as clear and understandable,
retaining the scene about white paint and a scarlet smokestack. Retained router
job `f9d64103-4880-427c-8342-57b4e3418347`, native prompt
`87cb0fae-6bf7-4b1e-afe0-c8327579bff3`. Earlier neutral source/output questions
also reported intelligible speech. A leading artifact question reported clicks
and abrupt cuts, which the neutral questions did not reproduce. Automated audio
review is inconsistent and does not certify human timbre, singing or English
quality. Listen to the saved FLACs and use the intended trained voice/encoder.
The v2 clip has finite/duration/GPU acceptance but no separate listening verdict.

32/48 kHz and non-pitch exports are implemented but have no real-checkpoint
acceptance here; the small non-pitch fixture is 32 kHz. Custom embedders,
MRF/RefineGAN, other F0 predictors, realtime and training are outside this port.
Long-file overlap behavior differs from upstream and needs extended listening.
The [operating guide](applio.md) records the interface and recovery limits.

## Evidence and development delivery

Local evidence is retained under `/home/jack/services/artifacts/applio-native/`:
`final-parity.json`, `lifecycle.json`, individual histories, converter manifests,
`impact.json`, prompts and setup/review receipts. Output FLACs remain in the normal
`output/audio` directory. Inputs, weights, user workflows and generated media are
not committed to source control.

The exact task and canonical graphs were prepared with embeddings disabled.
Cumulative change analysis against the pre-integration v0.42.1 main reports medium
risk (105 changed symbols, four device-dispatch flows); source review and relevant
regressions supplement it. Earlier analyses refused publication when code/HEAD
changed; fresh completed graphs superseded those receipts. RuntimeHost's existing
customized GitNexus workflow skills were preserved, with installation warnings.

The authoritative shared `comfyui` guidance is integrated (`e6722d8`) and discovery
checks pass. The complementary RuntimeHost placement guidance is committed and
pushed on `applio-native-placement` (`4b23f6b`), with all adapter checks passed.
Its DevTask `applio-native-placement-53537f35` remains blocked by serving-checkout
coordination/owner recovery; preserve it and resume through DevTask in an idle
window. That documentation-only branch does not add a RuntimeHost inference path.
