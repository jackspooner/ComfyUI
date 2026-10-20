# UniPASE acceptance record

20 October 2026, Linux x86_64, Python 3.13.16, PyTorch 2.14.1+cu130.
Source and checkpoint revisions and hashes are in [sources](unipase-sources.json)
and the local converter manifest. Code MIT, model-card Apache-2.0; component
notices are retained in the model directory.

## Verified offline

All four originals matched published SHA256s. Conversion validated 1,036 native
tensors, complete key/shape inventories, finite FP32 values and bit-exact saved
round trips after documented key mapping and positional weight-normalization
folding. Packaged SHA256:
`f4073752f4fbaa63c038401a3f4b05aa014a045851c8386c62a27269afba5157`.

79 targeted tests passed: UniPASE boundaries/math plus RVC and SoulX regressions.
UniPASE's 52 checks cover mono/stereo batches, 8/16/22.05/44.1/48 kHz source
rates, tiny inputs and 8/10/12/17/25-second boundaries, changed-rate sample count,
identity stitching, early cancellation, relative attention and PostNet normalization.
Fixture tests do not establish model quality or forced offload behavior.

`scripts/check_unipase.py` compared all three official noisy speech fixtures,
at 16 and 48 kHz output. Whole-clip CPU comparisons passed the predeclared maximum
absolute `.003` and relative-RMS `.015` gates. Observed maximum absolute difference
was `.000265`; worst relative RMS `.0000765` (0.00765%). Inputs were 4–14.61 seconds,
at 44.1/48 kHz. Retained WAVs and metrics:
`/home/jack/renders/unipase-acceptance/` (`*-before.wav`, `*-native.wav`,
`*-upstream.wav`, `cpu-parity.json`). The reference uses unchanged pinned numerical
code; only ESPnet's abstract base and PReLU lookup are replaced with PyTorch
identities. CPU comparison process peak RSS was 7,436,504 KiB with both complete
pipelines loaded; six comparisons took 54.8 seconds, not a serving RAM/VRAM bound.

The official short inference was used for parity, independently of our long-file
chunk orchestration. Neural speech parity therefore does not establish identical
long-form boundaries. Upstream's long script has short-input and changed-rate bugs;
our corrected orchestration has isolated boundary tests. Independent per-channel
processing extends upstream mono behavior; stereo spatial fidelity is unverified.

## Native GPU and lifecycle

The installed RTX 5090 uses driver 615.71.09, FP32 weights/compute, the selected
ComfyUI attention dispatcher, allocation compilation and dynamic VRAM. Native
UniPASE requests the existing full-precision attention contract; no global flags
were disabled. The final four-second official fixture completed in 15.92 seconds
(prompt `b383d3ed-68bc-4ef2-b9cd-668e47a3b5ac`), with 192,000 finite mono samples
at 48 kHz. CPU/native queue success is separate from human listening acceptance.

| Native check | Prompt ID | Outcome |
| --- | --- | --- |
| 17.125-second stereo, 44.1 kHz output, PLC off | `e9b4d434-f4bf-4bd6-987c-e11de0763388` | 755,212 samples/channel, two channels; 1.814 s warm |
| Targeted cancellation after first segment progress | `0de0f7d6-09cb-469a-a0f0-fbd806f8abf4` | Confirmed cancellation, execution_interrupted in enhance |
| Recovery, 16 kHz, PLC off | `5c2646d3-32c2-471e-80d4-32e612afd47c` | Success, 64,000 samples; PostNet bypassed |
| Explicit unload followed by 48 kHz reload | `1895765e-d6c2-4367-94ab-eee2b0cda4e5` | Success |

After idle queue checks, `/free` unloaded managed weights and released caches:
sampled total device usage fell from 4,517 to 2,331 MiB. The cold four-second run's
sampled peak was 5,524 MiB, before/after 1,961/4,375 MiB; service process RSS peaked
at 7,723,128 KiB. Samples were about one second apart and include desktop/service
baseline; these are not universal minima, process-isolated VRAM or low-memory
claims. Native GPU holds were confirmed released after successful and cancelled
runs. PostNet's LSTM loads fully through its patcher; no separate worker persists.

Pinned upstream GPU inference ran under a separately acquired validation hold only
after the native queue was idle and managed weights were explicitly unloaded.
The worker exited before exact-owner hold release/readback. The saved native FLAC
and normalized upstream GPU float WAV differed by relative RMS 0.00145952 (0.146%)
and maximum sample difference 0.00467062. Against upstream CPU the native GPU FLAC
had relative RMS 0.00332357 (0.332%) and maximum difference 0.01267083. CPU's strict
absolute 0.003 gate was not met by these cross-device/saved-audio comparisons;
GPU evidence establishes close measured agreement, not bit equality or that same
absolute tolerance. No waveform-perfect preservation or perceptual improvement
is claimed. Reference GPU Torch peak allocation was 3,330,563,072 bytes.

Before/after examples, including long stereo and upstream GPU audio, are retained
under `/home/jack/renders/unipase-acceptance/`. Native histories, cancellation
progress/receipt, resource samples and hold readbacks are under
`/home/jack/services/reports/unipase/`. The editable workflow is
`user/default/workflows/UniPASE - Native Speech Restoration.json`; its four functional
groups pass the geometry checker. Input example: `input/unipase_demo_source.wav`.

## Failure evidence and limits

The first GPU prompt `ecd3fe38-4aa6-4310-84d4-f6a84d5ad711` failed allocation
compilation in the encoder loop. Capturing retained layer-one features before the
repeated scope and reusing stable loop buffers corrected it. A subsequent run
reached Adapter attention, where INT8 attention rejected a 1024-wide single head;
the existing `low_precision_attention=False` contract fixed the FP32 path. Compiler
break/rogue diagnostics remain visible (six breaks/ten rogues on the cold final
run); successful compilation is not a zero-overhead claim.

One acceptance submit raced a managed restart and never acquired GPU ownership.
That interrupted check was retained and replaced only after confirming termination
and fresh startup. A separate live Drama 3 reference hold blocked startup; it was
left untouched until its owner released it. No foreign hold or unrelated job was
cancelled. Corrected verification waits for restart completion before submission.

No general music enhancement, universal synthetic-artifact repair, low-VRAM
capacity, other-accelerator acceptance or complete URGENT hybrid reproduction is
claimed. Stereo spatial fidelity, multi-hour input memory and human listening
quality remain unverified. Inference is offline; preserving content is a model
objective, not a guarantee.

RouterAudio is activated with integrated commit `4b0949d893da5355af429aefdbabba8ad3bb7658`.
Live discovery advertises audio_to_audio, required named source and the two controls.
Durable job `37ecb577-064a-4113-b907-7b232a109319` succeeded through native prompt
`065c5035-ca75-497d-9602-d1314088ed65`; identical-key replay returned the same job.
Its verified 135,941-byte FLAC SHA256 is
`3d7e3ea02e03c0eedc2f72e3c1672b3620b3e4ea577c5ca97d94d81a537249b9`.

MediaTools is activated at `3dd347dad1a35b62c810b5af27b6b7c955916661`, using
RouterClients `2f193c748a8d0d0f6f73cef19969187725363734` and RouterContracts
`1e4745d248e0126e9ba15b3bfad99ec430c9fc44`. Live selected discovery returns the
complete guide, audio-only input, required source, rate/PLC controls and dated
model provenance. MCP job `aa82c03b-be7e-4ef6-bf5e-00cdb77cf192` succeeded through
native prompt `3914224a-9ce8-4ef4-9b82-d27f5a9281de`. Identical-key replay returned
that same job. Metadata reads omit file paths; explicit collection saved a
135,938-byte FLAC with SHA256
`2cb09469eaa028e2d7465b214855c5cb9386bfe2520451fd54c0b0f81c6cd1e2`.
Independent decoding verified 192,000 finite mono samples at 48 kHz. The file is
retained under `/home/jack/services/reports/unipase/media-output/`; submission,
replay, discovery and verification receipts are in its parent directory.

MediaTools rejected the initial fixture outside configured MediaIO roots without
submitting inference. The successful check used an admitted test copy. Fixture
coverage additionally checks rejected paths, zero/false control preservation,
hash/size-verified collection and GET-only collection without regeneration.
The complete integrated MediaTools suite passed 425 tests across 47 files;
TypeScript/build, OpenAPI and lifecycle contract checks passed. Its broader live
all-model discovery smoke remains blocked by unrelated video route
`minimax-h3-ref2va` lacking release-date metadata; selected UniPASE discovery passed.
Setup reported existing npm audit findings and an esbuild install-script warning.
These checks do not establish Windows or perceptual-quality acceptance.
