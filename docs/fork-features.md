# Fork feature index

These guides cover this fork's integrations, workflow additions and retained
backports on top of [upstream ComfyUI](https://github.com/Comfy-Org/ComfyUI).
The current source base is v0.45.0. Earlier version numbers in acceptance
records identify the tested revision, not a promise about the current service.
For base capabilities, use the [upstream README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md).

Provision weights and optional dependencies separately. A source implementation,
CPU fixture pass, real-weight comparison, GPU lifecycle check and reviewed media
output are different evidence. Verify the running inventory before use and read
the linked guide's current limitations. Recorded RTX 5090 results do not certify
other devices, full-size inputs or all reference combinations.

## Images and video generation

| Addition | Guide and scope |
| --- | --- |
| H3 Ref2V and FL2V blueprints | [Editable subgraphs](h3-video-blueprints.md) expose distillation, attention, step overrides, duration, prompts and dimensions. They require provisioned native H3 weights and PDD/Cutlery nodes; artifact validation is not new media-quality acceptance. |
| H3 fractional masks | [Mask precision](h3-mask-precision.md) retains the 1/4096 grid and FP32 condition transport even with lower-precision compute. Only fully generated rows bypass mask timing; discover `minimax_h3_fractional_masks` on the running server. |
| H3 control and selective editing | [Control/editing](h3-control-editing.md) covers Fun ControlNet, sparse-attention composition, inpainting and LanPaint. Control residuals affect target video rows, leaving reference/keyframe/audio rows unchanged. CPU composition checks are separate from real-model acceptance. |
| H3 Contact Sheet | [Contact Sheet nodes](h3-contact-sheet.md) generate five ordered Turnaround views and a horizontal sheet. The separately licensed Turnaround LoRA is required; views are uncalibrated and need visual review. |
| Krea 2 editing | [Editing efficiency](krea2-edit-efficiency.md) covers grouped-query dispatch, target-only output projection and static text fusion. Update the paired Krea2 Edit source patch as documented; dynamic hooks retain per-step fusion. |
| Fibo 2.0 | [Setup and bounded acceptance](fibo2.md) documents the original v0.42.1 backport and later upstream changes. Turbo editing was checked; Dev, quantized weights, full parity and other devices are not covered by that acceptance. |
| Local orbit and Sol-H3 examples | [Optional workflow notes](fork-operations.md#optional-h3-workflow-examples) preserve the historical recipe and extension evidence without implying those local artifacts are bundled. |

### Dedicated MiniMax H3 Image

The fork includes the focused upstream
[native image integration](https://github.com/Comfy-Org/ComfyUI/commit/ede4e5fe8d15445c6fcd6828d70b202c22f9502c).
Use standard diffusion, minimax CLIP and VAE loaders with
`MiniMaxH3ImageConditioning`, the dedicated image checkpoint/VAE and the H3
Qwen3-VL encoder. This differs from video-derived still workflows and P-H3-Image.
Six Euler/simple steps at CFG 1 use the checkpoint's distilled grid. The publisher
supports up to 16 ordered references; with the VAE connected, each reference
conditions text and image latents. See the [publisher model card](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
for image-specific prompting and licensing. Source/CPU validation does not verify
maximum-size or 16-reference GPU quality. Use the ordinary queue after activation.

## Text, OCR and media understanding

| Addition | Guide and recorded limits |
| --- | --- |
| Native conversations | [Text/image history](text-conversations.md) and [acceptance](text-conversations-acceptance.md). History stores exact image attachments separately from JSON; callers save both. Audio/video use the single-prompt path rather than persistent media history. |
| Structured JSON | [JSON modes and support matrix](text-json.md). Prompt-level `json_schema` syntax extraction is distinct from opt-in token constraints and independent validation. Constrained mode requires its optional dependencies and `mtp=off`. Complete Gemma4 12B OYM BF16 has bounded JSON/writer checks; maximum capacity and perception accuracy are unverified. |
| Image-count planning | [Text image budget](text-image-budget.md) estimates same-size images within context and managed memory, reserving prompt/output space without encoding or generation. CPU checks do not establish maximum-size inference. |
| Media sizing | [Controls](text-media-sizing.md) and [acceptance](text-media-sizing-acceptance.md) cover video sampling, automatic sizing, output reservations and bounded OOM retries. Quality floors can cause explicit failure; recorded Gemma4/Qwen3-VL trials do not certify every adapter. |
| Qwen3-Omni | [Usage](qwen3-omni/usage.md), [status](qwen3-omni/status.md) and [conversion](qwen3-omni/quantization.md). Text/audio/image inputs produce text; video, speech output and diffusion conditioning are outside this integration. RTX 5090 lifecycle checks passed; the reference suite retains OCR/silence failures and prompt sensitivity (20/24 cases). |
| MOSS-Audio 8B | [Preparation and prompting](moss-audio.md). Local BF16 understanding through Load CLIP and Generate Text; recorded CPU/full-weight GPU comparisons include two-minute audio. Loading measurements and caption-quality limits remain in the guide. |
| Audio Flamingo 3 | [Setup](audio-flamingo.md). Native audio understanding/transcription; short-clip GPU and cancellation/reload checks passed, with broader quality and resource limits unverified. |
| PaddleOCR | [PP-OCRv6 setup and batching](paddleocr.md). Detection, CTC text, geometry and model scores remain separate per image. Bounded RTX 5090 trials showed faster warm batching, not a whole-workflow guarantee; a tiny badge remained unreadable. |
| GLM-OCR | [Conversion and limits](glm-ocr.md). Original-pixel image/crop recognition with padding and raw decoded text. Bounded GPU lifecycle checks passed, but badge accuracy failed; no production OCR route is claimed. |
| Video captioners | [Models and acceptance](video-captioners.md). SALMONN and InternVideo3 returned descriptions with mixed accuracy; InternVideo3 has no audio encoder or recursive tool loop. PadCaptioner failed usable-output acceptance with empty descriptions. Optional constrained JSON covers each answer/event caption; treat unaccepted paths as experimental. |

## Speech, music and audio processing

| Addition | Guide and recorded limits |
| --- | --- |
| MOSS speech and codec | [Native audio grounding](native-audio-grounding.md) covers VoiceGenerator, Base Delay, v1.5 acting, TTSD, Local Transformer and the MOSS v1 codec. Queue checks include reference encoding and continuation; broader speech quality and lifecycle acceptance remain pending. Local Transformer uses synchronous 32-book audio and has bounded CPU/GPU comparisons. |
| MOSS SoundEffect v2 | [Candidate setup](moss-soundeffect.md) covers offline conversion, Qwen3 conditioning and 48 kHz mono DAC. Component and lifecycle checks passed, but quality was mixed and the recorded candidate was not deployed. |
| Drama 3 | Built with Fish Audio. [Setup](drama3.md) and [acceptance](drama3-acceptance.md) cover references, long dialogue and saved take revisions, with NVFP4/MXFP8 experts and BF16 remaining weights. Initial target is Blackwell CUDA. NVFP4 dialogue/long-form quality failed; added MXFP8 conversion is not serving acceptance. |
| DramaBox loading | [Shared dual CLIP](dramabox-clip.md) loads Gemma 3 12B and audio components while preserving the existing loader's workflow conditioning. |
| AuK-Flash | [Generation](auk-flash.md) and [acceptance](auk-flash-acceptance.md) use native MODEL/CLIP/VAE, Euler sampling and an optional local prompt helper. Full-weight checks passed; listening and broader quality acceptance remain incomplete. |
| YingMusic Singer Plus | [Singing and reference editing](yingmusic.md) covers voice/melody references and Chinese/English lyrics, optional lyric preprocessing, separation and accompaniment mixing. Recorded RTX 5090 queue/lifecycle and reference-editing checks are bounded evidence. |
| SoulX-Singer | [Synthesis and conversion](soulx-singer.md) includes F0/note conditioning, RMVPE, separation and dereverberation. CPU and short-clip GPU checks passed, but the replacement-lyric quality gate failed; alignment, singer identity and subjective quality need review. |
| UniPASE | [Speech restoration](unipase.md) and [acceptance](unipase-acceptance.md) use four managed FP32 networks, rate preservation and bounded segments. Music enhancement and guaranteed repair of synthetic artifacts are outside validated scope. |
| Applio / RVC | [File conversion](applio.md) and [acceptance](applio-acceptance.md) cover v1/v2 HiFi-GAN voices, ContentVec, RMVPE and optional FAISS. Training, realtime capture, MRF/RefineGAN and other embedders are excluded; recorded idle-unload acceptance remained unrun. |

## Geometry and shared execution

- [MoGe geometry interchange](moge-depth.md#numeric-geometry-interchange):
  bounded pickle-free point/depth/normal archives, camera/FOV information,
  depth/normal/mask PNGs and optional GLB. Panorama depth is relative radial
  distance; perspective scale and camera remain estimates. The depth-PNG node
  uses 16-bit grayscale and range metadata.
- [Trellis2 existing-mesh texturing](trellis2-mesh-texturing.md): preserve mesh
  faces through centering/scaling and return the matching mesh/subdivisions for
  UV baking and export. The optional CPU converter needs Eigen 3, C++20 and a
  compatible PyTorch runtime; weights are separate.
- [SAM 3D Body precision](sam3d-body-precision.md): BF16 backbone with FP32
  prediction layers, FP32 fallback and explicit overrides. Attention honors the
  model's precision opt-out. CPU checks are not full-model visual acceptance.
- [Component leases](component-leases.md) and [scoped vision reuse](vision-feature-reuse.md):
  opt-in bounded reuse preserves private adapters, text passes, DeepStack and
  generation/KV state. Partial Qwen3-VL packages cannot generate text. ComfyUI
  retains loading/offload ownership; matching lifecycle integration and GPU
  acceptance are separate requirements.
- [Video extraction](video-extraction.md), [timing](run-timing.md),
  [queue controls](queue-control.md), [maintenance](maintenance.md) and
  [archive model cache](model-cache.md) document the shared operating contracts.

For implementation and offline conversion, read [native model integration](native-model-integration.md).
For startup, memory recovery and local-reference behavior, read [fork operations](fork-operations.md).
