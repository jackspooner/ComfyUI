## SAM 3D Body precision in this fork

The dense BF16 SAM 3D Body loader prefers BF16 backbone computation with FP32
prediction layers, avoiding the generic CUDA FP16 default. FP32 fallback and
explicit global dtype overrides remain available. See [precision, setup and acceptance limits](docs/sam3d-body-precision.md).
Comfy Kitchen INT8 attention honors the model's explicit precision opt-out for
all input dtypes, including BF16 DINOv3, through the existing SDPA path.
CPU regression checks and source availability do not establish full-model visual
acceptance or serving activation.

## Bounded native component reuse in this fork

This fork adds opt-in sequential CLIP leases, including shared full Qwen3-VL
backbones with private Krea2, Fibo2, FLUX.3 and selected 8B route adapters.
Native memory management retains loading/offload ownership. An explicit bounded
image-feature cache can reuse source vision across compatible nodes; generation
history and KV state stay per call. Partial Qwen3-VL packages cannot generate text.
See [API, compatibility, bounds and CPU acceptance](docs/component-leases.md).
Matching lifecycle integration and real-weight GPU acceptance are separate from
source availability.

## H3 control and selective editing in this fork

Native H3 Fun ControlNet now composes with native block-sparse attention in either
patch order, preserving control weights and cleanup while keeping the control
stream dense. See [checkpoint representations, guides, inpainting and LanPaint
extension composition](docs/h3-control-editing.md). CPU regression coverage is
separate from real-model acceptance and serving activation.
Control residuals update target video rows only, leaving reference/keyframe and
audio rows unchanged (`minimax_h3_control_target_only` feature). Repeated native
sparse nodes retain one prepare callback and a stable composed fallback chain.

## H3 Contact Sheet in this fork

Native [Contact Sheet nodes](docs/h3-contact-sheet.md) jointly generate five
Turnaround views and decode them independently, returning ordered images and a
horizontal contact sheet. They require the separate MiniMax-licensed Turnaround
LoRA and native H3 components. Views are uncalibrated and require visual review;
source and CPU fixture support do not establish serving or GPU acceptance.

## Local maintenance fencing in this fork

This fork adds opt-in local operator maintenance: persistently fence new native
prompt admission, drain all accepted pending/running prompts, and retain bounded
non-executable completion receipts across restart. The server starts still fenced
until the owning operation explicitly resumes. It does not pause tensors, restore
unfinished prompts, deploy code, or release GPU holds. Crashed unfinished work is
reported as interrupted/unknown, not replayed or declared safe.

Set both `COMFYUI_MAINTENANCE_DIR` and `COMFYUI_MAINTENANCE_TOKEN_FILE` at startup;
see [authentication, exact API, recovery and rollout limits](docs/maintenance.md).
`COMFYUI_MAINTENANCE_MAX_RECEIPTS` optionally increases the default 10,000-receipt
capacity, or accepts `unlimited` to remove the receipt-count cap. Retention and
unresolved-work safeguards remain unchanged; apply it only
through a safe restart with the same store and adequate disk/memory headroom.
Restored failures remain failed in Jobs/history, but omit optional error details
when the journal retained only a sanitized lifecycle marker. This keeps history
and its generated-asset previews readable without restoring private error data.
Without that configuration, ordinary queue behavior is unchanged. Source and
isolated CPU/HTTP tests are available; serving activation is a separate operation.
This extends [upstream ComfyUI](https://github.com/Comfy-Org/ComfyUI) without changing
its model executor; upstream usage and attribution remain below.

## Efficient video extraction in this fork

**Get Video Components** retains its existing outputs and precision by default,
with direct float-buffer assembly in place of stacking all frame tensors. Its
advanced controls select images/audio/metadata, retain every Nth frame, limit the
longest image edge, and optionally use faster 8-bit SDR RGB conversion. File
inputs skip unused streams and avoid converting discarded frames. HDR, float
and high-bit-depth inputs retain the existing conversion path. No new dependency,
external FFmpeg process or GPU decoder is required.

Start with **Load Video → Trim Video → Get Video Components**, then select
**Images only** if audio is unnecessary. Existing workflows need no changes.
See [controls, Python contract, memory limits and benchmarks](docs/video-extraction.md).
Restart an idle instance to load updated core code; source integration alone does
not update an already-running process.

## External image references in this fork

Start with `--allow-external-image-paths` to let **Load Image** and **Load Image
(as Mask)** read an absolute image filename anywhere the process can read. The
file stays in its original location; changing it changes subsequent executions,
and moving or deleting it breaks the reference. This is opt-in and does not relax
the generic annotated-path resolver or other file endpoints. `/features` reports
`external_image_paths`; `/view?filename=<URL-encoded-absolute-path>` serves only
verified image files with the existing preview/channel options. Relative input,
output and temporary references keep their existing behavior.

The matching Cutlery external-image preview extension restores these references
in frontend workflow views. RouterImage uses direct local references and imports
URL/inline bytes through `/upload/image`; imported inputs persist until deleted.
Old outputs referencing already-deleted temporary inputs are not repaired.

## MiniMax H3 blueprints in this fork

Native H3 fractional video/audio denoise masks retain a 1/4096 token grid and
FP32 condition transport, including with BF16/FP16 model compute. Only fully
generated rows bypass mask timing; near-one values remain conditioning. Running
support is advertised by `GET /features` as `minimax_h3_fractional_masks: true`. See
[mask precision, validation and activation limits](docs/h3-mask-precision.md).

**MiniMax H3 - Ref2V** and **MiniMax H3 - FL2V** provide editable video/audio
subgraphs with distillation and attention dropdowns, preset attention defaults,
an optional step-count override, duration in seconds, prompt, width and height.
They use the locally provisioned native H3 models and installed PDD/Cutlery nodes;
no weights are downloaded. See [controls, requirements and verification limits](docs/h3-video-blueprints.md).

## Native archive model cache in this fork

This fork adds an opt-in Linux NVMe cache for safetensors models stored in a
ComfyUI-shaped archive on a slower drive. Archive models appear as `archive/...`
in existing selectors. The loader stages a verified complete copy before use;
execution tracks reuse and releases affected cached results after 30 minutes of
inactivity. SQLite recovery preserves complete copies across restarts, and
capacity/free-space limits protect the NVMe. A local CLI queues prefetch, pin,
unpin and eviction requests and reports status.

Staging reuses small RAM buffers and batches persisted copy progress to once per
second without forcing a disk sync for each progress update. Completion logs separate
copy/hash, flush and verification times. Full checksum verification and durable
publication remain required; the default inactivity expiry remains 30 minutes.

Enable with `--model-cache-config /absolute/path/config.yaml`; see
[configuration, operation and limitations](docs/model-cache.md) and
[the example configuration](model_cache.example.yaml). It runs inside ComfyUI,
without an additional daemon or systemd unit. Cleanup occurs while ComfyUI runs
and on its next start after downtime. The feature is tested with isolated CPU
fixtures; activating source changes requires an idle service restart.
Custom loaders bypassing the common resolver and weight loader, background model
loads, GGUF and multi-file repositories are outside verified support.

## Native Fibo 2.0 backport

This fork backports upstream [Fibo 2.0 support](https://github.com/Comfy-Org/ComfyUI/pull/17240)
onto its v0.42.1 base. Load the diffusion model, Qwen3-VL 4B with CLIP type
`fibo2`, and Flux2 VAE, then use **TextEncodeFibo2** and the normal sampler.
Edit Turbo uses four steps and CFG 1; Dev also supports text-to-image.
See [model files, workflow, verification and limits](docs/fibo2.md).

## Native MOSS Local Transformer candidate

This fork adds offline BF16 packaging and native CLIP generation for MOSS Local
Transformer, using synchronous 32-book audio, the shared MOSS v1 codec, references
and continuation. Pinned CPU comparisons and short RTX 5090 generation/lifecycle
checks pass; broader quality and serving acceptance remain pending. See [conversion, explicit sampling defaults and limits](docs/native-audio-grounding.md#native-local-transformer).

This fork also includes a native MOSS SoundEffect v2 candidate with offline
checkpoint conversion, a Wan-based continuous flow model, Qwen3 conditioning and
a 48 kHz mono DAC VAE. CPU component/checkpoint comparisons and isolated RTX 5090
generation/lifecycle checks pass. Output quality is mixed, and it is not deployed.
See [setup, workflow and acceptance limits](docs/moss-soundeffect.md).

## Native Drama 3 in this fork

Built with Fish Audio. This fork adds native Drama 3 speech, speaker references,
long-form dialogue and explicit saved take histories for whole-take revisions.
An offline converter packages routed experts as NVFP4 and the remaining weights
as BF16; ComfyUI owns generation/codec loading and execution-local caches. The
initial target is a Blackwell CUDA GPU. The RTX 5090 candidate completes speech,
cloning, revisions and lifecycle checks within memory, but tested dialogue and
long-form quality failed; serving activation remains unapproved. See [setup and limitations](docs/drama3.md)
and the [acceptance record](docs/drama3-acceptance.md). Existing upstream capability
and attribution are described below.

<div align="center">

# ComfyUI

## Native UniPASE in this fork

This fork adds **Load UniPASE → UniPASE Speech Restoration** for local generative
speech enhancement with four managed FP32 networks, ordinary AUDIO inputs/outputs,
rate preservation and bounded long-recording segments. Provision weights explicitly;
no runtime downloads occur. See [setup, licensing and use](docs/unipase.md) and
[acceptance limits](docs/unipase-acceptance.md). Music enhancement and guaranteed
repair of every synthetic artifact are outside the validated scope.

## Native Applio / RVC in this fork

This fork adds **Load Applio / RVC Models → Applio / RVC Voice Convert** for local
file conversion with trained RVC v1/v2 HiFi-GAN voices, matching ContentVec, shared
RMVPE and optional FAISS retrieval, with context and overlap blending for segments.
Dedicated managed handles reuse ComfyUI's
patchers, casting, queue and media nodes; Applio's application is not launched.
See [offline setup, supported exports and limits](docs/applio.md) and
[acceptance evidence](docs/applio-acceptance.md). Training, realtime capture,
MRF/RefineGAN and other embedders are outside this path. The two 40 kHz demo voices
pass native GPU conversion and cancellation/recovery checks; explicit idle unload
acceptance remains unrun while unrelated generation occupies the queue.

## Stable release maintenance in this fork

This fork merges upstream stable [v0.43.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.43.0)
while retaining its native model, conversation and memory-recovery changes.
Upstream adds adaptive Qwen MTP trees, writable KV storage, fused Flux2 VAE
operations and Fibo/MOSS-Image fixes. Cloud-deployment uploads and their persistent
headless-use marker are excluded under this fork's no-internet policy;
source support does not establish local checkpoint or quality acceptance.
Use the release's pinned frontend, templates and Comfy Kitchen alongside a
compatible GPU runtime. See [maintenance, verification and rollback](docs/stable-release-maintenance.md).

## Native Audio Flamingo 3 in this fork

This fork adds native audio understanding and transcription with NVIDIA Audio Flamingo 3 through Load CLIP and Generate Text. It reuses ComfyUI model/attention/memory management and repackages a local HF snapshot without downloading during execution. See [setup, licensing, inputs and acceptance limits](docs/audio-flamingo.md). CPU reference checks and short-clip RTX 5090 inference, cancellation/reload and RouterAudio acceptance pass; broader resource and quality limits remain unverified.

**The most powerful and modular AI engine for content creation.**

## Native YingMusic Singer Plus in this fork

This fork adds the Singer Plus transformer and melody encoder to ComfyUI's native
loaders, attention and memory management. **Load Diffusion Model + Load Audio
Encoder + Load VAE → YingMusic Generate / Edit Singing** accepts voice and melody
audio references and Chinese/English lyrics. Weight conversion is local and explicit;
there is no singing inference worker. Optional phonemizer/espeak and Chinese ONNX
assets are used only for lyric preprocessing. See [setup, conversion and limitations](docs/yingmusic.md).
RTX 5090 generation, cancellation/reload and RouterAudio reference editing passed, including optional separation and accompaniment mixing. See the bounded acceptance evidence in the model documentation.

## Native PaddleOCR in this fork

This fork adds **Load PaddleOCR Models → PaddleOCR** for local PP-OCRv6 medium
text detection and CTC recognition. Native ComfyUI model management owns both
FP32 models. Outputs include text, region geometry, model scores and preprocessing
transforms; independent batch images remain separate. Default preprocessing avoids
enlargement, with an explicit upstream-resize comparison mode. See
[model setup, node contracts and measured limits](docs/paddleocr.md).
Optional detector-frame and recognition-crop microbatches now group identical
prepared shapes into real batched forwards. Defaults remain one; extra image
sockets accept different source dimensions without resizing them together.
CPU batching parity passes. RTX 5090 activation and a bounded 14-frame GPU trial
passed, with about 2.1× lower warm queue execution time at 2-frame/8-crop batching.
CUDA results can change numerically; this is not a whole-workflow speed guarantee.
Worktree preparation for this optional capability uses the normal setup command
with `--paddleocr` to check its preinstalled OpenCV/torchvision prerequisites.
Source integration requires activation before new nodes appear in a running
service. Checkpoint validation supports lazy weight allocation; OCR requests FP32
attention through the selected dispatcher. Recognition errors remain visible;
no production advert route is added. RTX 5090 advert tests read all five supplied
text lines correctly with official preprocessing; the tiny circular badge fails.

## Native GLM-OCR in this fork

This fork adds the official Z.ai GLM-OCR recognizer through **Load CLIP → Generate
Text**, using ComfyUI's native model loading, attention, prefetch and memory owners.
It processes one original-pixel image/crop with padding and no enlargement, and
adds a raw decoded-text output. See [setup, conversion and limits](docs/glm-ocr.md).
Bounded GPU lifecycle checks passed, but the Almay badge failed literal accuracy;
the production OCR route remains unconfigured. See the linked acceptance limits.

## Native text conversations in this fork

This fork also adds text/image conversation history and exact context budgets
to the existing **Load CLIP → Generate Text** path for native Qwen3.5/Qwen3.8 and
MOSS, plus complete Qwen-VL/Omni, Gemma3/4 and GLM-OCR image paths.
Templated image turns snapshot exact pixels into local immutable safetensors under
the configured system-user `text_history/images` directory;
replay needs both the JSON and attachments. Saving history belongs to the caller.
Audio/video inputs automatically use the single-prompt media path when no history
or structured content is supplied. Video respects its token cap and optional memory-aware sizing without enabling
conversation history; legacy audio retains its existing budget behavior.
**Generate Text** also accepts an optional `json_schema` (legacy API prompts may
omit it, including MOSS audio calls); this prompt-level JSON
mode extracts and syntax-checks the first JSON object or array, while schema conformance
remains model-dependent. Opt-in `response_format=json_schema` adds native
llguidance token constraints and independent schema validation on supported
complete MOSS/Qwen3-VL packages; it requires optional dependencies and `mtp=off`.
See [native constrained JSON](docs/text-json.md) for the exact subset, support
matrix and CPU-only acceptance limits (no full-model/GPU activation claimed). See
[native text conversations](docs/text-conversations.md) for prerequisites,
contracts, examples, limits and verification. This source capability requires
deployment before it appears in a running ComfyUI process. The
[local image-history acceptance record](docs/text-conversations-acceptance.md)
lists tested checkpoints, restart replay and coverage limits.

## Fork: Krea 2 inference

This fork builds on [upstream ComfyUI](https://github.com/Comfy-Org/ComfyUI)
and its [README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md).
Upstream provides the node graph, model loading, sampling and local image,
audio and video generation documented below; upstream credits and licenses remain.

Krea 2 uses the shared attention dispatcher's grouped-query support without
expanding K/V heads in the model, and projects only target tokens at the output.
The paired Krea2 Edit source-patch update prepares static text fusion once per
conditioning during sampling preparation. Update both repositories together;
existing workflow sockets and checkpoint layouts are unchanged. Hooks and dynamic
patches retain per-step fusion. See [Krea 2 editing efficiency](docs/krea2-edit-efficiency.md)
for setup, tests and limitations. These changes do not alter other model families.

## Local MiniMax H3 workflows and acceleration

This installation uses [upstream ComfyUI](https://github.com/Comfy-Org/ComfyUI)
and its [README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md):
the node graph UI/API, local image/audio/video inference, model loading and
memory management described below are upstream functionality.

The saved **MiniMax H3 - 360 Orbit** workflow uses Pablo Dawson's orbit LoRA
with the same image at both endpoints, 768×768, 73 frames and 28 steps. Open it
from the Workflows sidebar and press Run; the default input is the latest toy-house
image from this installation. It saves silent MP4 video under `output/360_orbit`.
See the [recipe, model requirements and recovery guide](docs/minimax-h3-360-orbit.md)
and the repository-owned [agent skill](skills/minimax-h3-360-orbit/SKILL.md).
The graph passed prompt validation and a complete house render on 7 October 2026
(73 frames at 24 fps, 768×768, no audio track; about 73 seconds on this host).
The author's training examples are human-centric, so object orbit quality is not
established by the model card.

The local Linux RTX 5090 installation additionally includes
[ComfyUI-Sol-H3](https://github.com/xmarre/ComfyUI-Sol-H3) v0.1.9, pinned at
`09a3320205512cd87f6110397fca1e25c19f5536`. Its Exact Runtime and experimental
Sana/CuTe sparse attention are available as MODEL patch nodes. In an existing
native H3 graph, connect the model after other patches through **Sol-H3 Exact
Runtime**, or **Sol-H3 SOL Attention (Experimental)**, before the sampler.
The latter includes Exact Runtime and defaults to one dense evaluation and two
dense leading layers. Sparse attention approximates the dense result.

GPU kernel and native model-patcher checks passed on 7 October 2026; full-sized
render quality and end-to-end speed have not been measured here. Existing router
`sol` selections still use the separate `SolAttnPatch` Triton extension. NVIDIA's
standalone Sol Engine is not installed by this node pack. See the
[local setup and recovery notes](../services/infrastructure/Workspace2LinuxDeployment/docs/minimax-h3-sol.md)
for dependencies, verification, limitations and rollback. Upstream credit and
licensing remain with the projects linked above.


[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]
[![][github-release-date-shield]][github-release-link]
[![][github-downloads-shield]][github-downloads-link]
[![][github-downloads-latest-shield]][github-downloads-link]

[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://discord.com/invite/comfyorg
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases
[github-release-date-shield]: https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat
[github-downloads-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat
[github-downloads-latest-shield]: https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest
[github-downloads-link]: https://github.com/comfyanonymous/ComfyUI/releases

<img width="1590" height="795" alt="ComfyUI Screenshot" src="https://github.com/user-attachments/assets/36e065e0-bfae-4456-8c7f-8369d5ea48a2" />
<br>
</div>

ComfyUI is the AI creation engine for visual professionals who demand control over every model, every parameter, and every output. Its powerful and modular node graph interface empowers creatives to generate images, videos, 3D models, audio, and more...
- ComfyUI natively supports the latest open-source state of the art models.
- It is by far the most optimized inference engine for diffusion models in the world with a focus on local consumer hardware.
- Can run even the biggest open source models on as low as 4GB vram + 8GB ram relatively quickly (saturating your GPU compute) using our state of the art asynchronous weight streaming technology.
- [Partner nodes](https://docs.comfy.org/tutorials/partner-nodes/overview#partner-nodes) provide access to the best closed source models such as Nano Banana, Seedance, Hunyuan3D, etc.
- It is available on Windows, Linux, and macOS, locally with our [desktop application](https://www.comfy.org/download), our [portable install](#installing) or on our [cloud](https://www.comfy.org/cloud).
- The most sophisticated workflows can be exposed through a simple UI thanks to App Mode.
- It integrates seamlessly into production pipelines with our API endpoints.

## Native Qwen3-Omni in this fork

Load CLIP (`qwen3_omni`) → Generate Text supports text, audio, complete image
batches and combined inputs to text. The offline converter preserves the selected
NVFP4 checkpoint's packed weights and embeds inference assets in one file. Native
RTX 5090 execution, cancellation/offload/model transitions and live queue runs
are verified. The reconstructed-reference suite has no new assertion failures;
both paths pass 20/24 cases, retaining checkpoint OCR/silence limitations and
prompt sensitivity. See [operation and saved workflow](docs/qwen3-omni/usage.md),
[measurements and live evidence](docs/qwen3-omni/status.md), and
[conversion/recovery](docs/qwen3-omni/quantization.md). Video, speech output,
diffusion conditioning remain outside this integration. Text/image history uses
the shared [conversation contract](docs/text-conversations.md); audio/video history is excluded.
Native packages may mix validated dense and NVFP4 projections; the offline
converter still targets the pinned source. [Compatibility and upstream assessment](docs/qwen3-omni/upstream-readiness.md)
records reference blockers, residual precision limits and proposed patch boundaries.

## Native MOSS-Audio in this fork

[Upstream ComfyUI](https://github.com/Comfy-Org/ComfyUI) and its
[README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md) describe the
node graph engine, local multimodal inference, APIs and memory management retained
here. This fork adds native MOSS-Audio 8B audio understanding through **Load CLIP →
Generate Text**, using a locally converted BF16 checkpoint and the existing AUDIO
socket. See [preparation, usage and verification status](docs/moss-audio.md).
CPU reference and full BF16 RTX 5090 comparisons pass, including two-minute audio.
The local service and saved example workflow are verified. See the linked document
for measured loading performance and caption-quality limits.

**Generate Text** accepts up to 1,048,576 requested output tokens. Set `max_length`
to `0` to use the model's remaining context after text and media expansion;
thinking and the final answer share that budget. Model context and available
memory still limit execution. See [token budgets and caption prompting](docs/moss-audio.md#token-budgets-and-caption-prompting).
Reasoning is separated at `</think>` even when the model omits the opening tag,
preventing reasoning or duplicate text from leaking into `generated_text`.

This fork adds `fps` (analysed frames per second), `source_fps` (the IMAGE batch's
actual playback rate), and enabled-by-default `auto_resize` to **Generate Text**. Explicit
`auto_resize=false` retains model-default sizing, including for adapters without
automatic planning support. Saved explicit values are preserved. For example,
`source_fps=24, fps=4` selects frames 0, 6, 12 and 18 in the first second.
`min_short_edge` defaults to 256 pixels during automatic sizing, including OOM
retries. Requests that cannot fit at that quality floor fail clearly; lower FPS,
shorten the clip or explicitly lower the floor (`0` disables it). Source/model
limits remain in effect; the floor does not force unsupported enlargement.
Automatic sizing uses measured device headroom after managed model loading,
model-aware cache/workspace estimates and the model/user token ceilings while
preserving a positive `max_length`. It reports the effective window, selected
sizes and estimates through `context_info`, with bounded smaller-media retries
on device OOM. This is conservative request-specific planning, not a universal
maximum-context benchmark. Qwen3.5/3.8 text planning reserves its requested
generation workspace through the existing loading/offload owner before checking
headroom; insufficient memory still fails without shortening the request.
See [media sizing, supported adapters and limits](docs/text-media-sizing.md).
The [RTX 5090 acceptance record](docs/text-media-sizing-acceptance.md) covers native
Gemma4 and Qwen3-VL queued runs, memory-derived and explicit token caps, and retained
output reservations; other models/backends still need separate live acceptance.

The shared Llama/Qwen attention path honors per-module attention preferences in
prefill and cached decoding. Explicit preferences use masked dispatch for fixed
caches; unconfigured fixed caches retain the existing flash decode path. See
[native audio integration status and verification](docs/native-audio-grounding.md).
The same guide covers pinned VoiceGenerator BF16 conversion and native speech:
**Load CLIP (moss_tts) → MOSS VoiceGenerator → Decode Audio Codes → Save Audio**.
Text and voice descriptions produce discrete codes for the separately loaded codec.
Offline BF16 conversion and native CLIP loading also support pinned base Delay,
v1.5 acting and TTSD bundles. Their speech nodes accept ordered voice references,
prompt controls and explicit continuation, with acting text normalization and
TTSD defaults. See the linked guide for complete queue inputs and acceptance limits.
The FP32 MOSS v1 codec has an offline converter and native **Load Audio Codec →
Encode Audio Codes → Decode Audio Codes** nodes, with integer payloads and
ComfyUI model/offload ownership. Short RTX 5090 queue runs passed VoiceGenerator,
Base Delay, acting and TTSD, including reference encoding and continuation.
Speech quality, broader accelerator lifecycle coverage and live-service adoption
remain pending.

## Native AuK-Flash

This fork adds an offline converter and native MODEL, Qwen-Omni conditioning CLIP
and audio VAE for [Tencent AuK-Flash](https://huggingface.co/tencent/AuK-Flash), plus
an optional local prompt helper. It reuses standard loading, Euler sampling and
audio output. [Setup, examples and verification status](docs/auk-flash.md) distinguish
passing full-weight CPU/GPU checks from pending listening and broader resource acceptance.
The local deployment includes these nodes; listening and broader quality acceptance
remain incomplete. Verify the running inventory when using another installation.

## Native SoulX-Singer in this fork

This fork adds native SoulX-Singer lyric synthesis with continuous F0 or note/score
conditioning, transcription-free singing voice conversion, RMVPE pitch extraction,
karaoke separation and dereverberation. It uses ComfyUI operations, selected
attention, managed loading/offloading and the memory compiler, with standard AUDIO
inputs/outputs and editable lyric/note timing. Runtime inference is offline.
See [pinned models, conversion, nodes, examples and acceptance limits](docs/soulx-singer.md).
Default-step full-weight CPU comparisons and short-clip RTX 5090 queue/lifecycle
checks pass. The retained song excerpt follows pitch but fails its replacement-lyric
quality gate; automatic alignment, exact singer identity and subjective singing
quality require separate review. The examples include the complete reference-song
path and original-backing mixing. Source/candidate acceptance does not restart
the serving installation. Use the existing worktree setup command with `--soulx`.

## Native model integration development

For model ports and checkpoint conversion, read the
[native integration guide](docs/native-model-integration.md). It covers component
selection, standard loaders and dedicated managed handles, offline conversion,
ComfyUI memory ownership, reference/lifecycle validation and delivery, with links
to implemented ports. Its Applio example describes scoping only; it adds no Applio
support. Prepare a development worktree as documented below; GPU acceptance needs
separately provisioned checkpoints and a compatible runtime.

`SaveMoGeDepthPNG` converts native MoGe output to the same 16-bit grayscale
format and depth-range metadata used by RuntimeHost. See
[MoGe metric depth PNG](docs/moge-depth.md) for encoding, workflow and recovery
details.

## Local stable-release maintenance

The selected upstream stable version is linked in [stable release maintenance](#stable-release-maintenance-in-this-fork).
Existing H3 inference is the local release acceptance workflow; new model variants
need their own weights and end-to-end acceptance.

This fork retains the upstream node engine, inference APIs and model support
attributed above, plus local pinned-memory and partial-prefetch recovery fixes.
The Linux installation uses its existing `.venv`, shared models outside Git and
`comfyui.service`. [Stable-release operations](docs/stable-release-maintenance.md)
describe verification, protected GPU dependencies and rollback. Develop changes
in a linked worktree with `COMFYUI_PYTHON=/path/to/prepared/python python3
scripts/setup_worktree.py`; it checks Python >=3.10, PyTorch, einops and safetensors
without installing packages or starting workloads. Use that interpreter for tests.
The root `.gitnexusignore` keeps local environments, installed node packages and
disabled third-party node sources outside the core code graph. Those sources
belong to separately owned repositories.

## Get Started

### Local

#### [Desktop Application](https://www.comfy.org/download)
- The easiest way to get started.
- Available on Windows & macOS.

#### [Manual Install](#manual-install-windows-linux)
Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

### Cloud

#### [Comfy Cloud](https://www.comfy.org/cloud)
- Our official paid cloud version for those who can't afford local hardware.

## Examples
See what ComfyUI can do with the [newer template workflows](https://comfy.org/workflows) or old [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Features
- A visual node graph for building and reusing image, video, audio, 3D, and text workflows without code.
- Reusable subgraphs, workflow templates, App Mode, and a local API for integrating workflows into applications.
- Efficient local execution with asynchronous queueing, partial graph re-execution, smart VRAM and RAM management, model offloading, and support for quantized models.
- Broad native model support. This is a representative list; browse the [workflow library](https://comfy.org/workflows/) for maintained, ready-to-run templates.
  - [Image generation](https://comfy.org/workflows/tag/text-to-image/): Stable Diffusion 1.5, SDXL, SD3.5, Flux.1, Flux.2, Qwen Image, Z-Image, Hunyuan Image 2.1, HiDream, Lumina Image 2.0, Chroma, Anima, LongCat Image, Ideogram 4, Krea 2, MageFlow, Microsoft Lens, PixelDiT, Kandinsky 5, Ernie Image and Moss Image.
  - [Image editing](https://comfy.org/workflows/tag/image-edit/): Flux Kontext, Flux.2 Klein, Qwen Image Edit, HiDream E1.1 and O1, OmniGen2, Boogu, JoyImage Edit, MageFlow Edit, and LongCat Image Edit.
  - [Video generation](https://comfy.org/workflows/tag/video-generation/): Wan 2.1 and 2.2, LTX-Video 2 and 2.3, HunyuanVideo 1.5, Kandinsky 5 Video, CogVideoX, Cosmos Predict2, Bernini-R, SCAIL 2, and Mochi.
  - [Audio and video generation](https://comfy.org/workflows/): MiniMax H3 and LTX-AV.
  - [Audio generation](https://comfy.org/workflows/tag/text-to-audio/): ACE-Step 1.5, Stable Audio 3, MiniMax Music 3 and Yue 2.
  - [3D and vision](https://comfy.org/workflows/): Hunyuan3D 2.1, TripoSplat, SeedVR2, SUPIR, Depth Anything 3, MoGe, SAM 3 and 3.1, RT-DETRv4, and BiRefNet.
  - [Text generation](https://comfy.org/workflows/tag/text-generation/): Gemma 3 and 4, Qwen3, Qwen3.5, and Qwen3-VL, including multimodal inputs.
- Load complete checkpoints or separate diffusion models, VAEs, text encoders, LoRAs, ControlNets, adapters, and upscalers from supported model formats.
- Built-in tools for inpainting, outpainting, reference conditioning, masks and compositing, model merging, upscaling, frame interpolation, segmentation, depth estimation, and media processing.
- Save and load workflows as JSON, or recover complete workflows and seeds from supported generated media.
- Runs fully offline: core does not download anything unless you request it. Use `--offline` to disable the optional paid [Comfy API nodes](https://docs.comfy.org/tutorials/api-nodes/overview) and force all built-in functionality to stay offline, or `--disable-partner-nodes` to only disable the API nodes.
- Extend ComfyUI with custom nodes
- Configure additional model locations with [`extra_model_paths.yaml`](extra_model_paths.yaml.example).
- Support for saving and loading high bit depth images and videos: 16 bit PNG images, 32 bit EXR, 10 bit AVIF are supported and more.
- Support for saving and loading HDR videos and images in various formats.


## Release Process

ComfyUI follows a weekly release cycle targeting Monday but this regularly changes because of model releases or large changes to the codebase. There are three interconnected repositories:

1. **[ComfyUI Core](https://github.com/comfyanonymous/ComfyUI)**
   - Releases a new major stable version (e.g., v0.7.0) roughly every 2 weeks.
   - Starting from v0.4.0 patch versions will be used for fixes backported onto the current stable release.
   - Minor versions will be used for releases off the master branch.
   - Patch versions may still be used for releases on the master branch in cases where a backport would not make sense.
   - Commits outside of the stable release tags may be very unstable and break many custom nodes.
   - Serves as the foundation for the desktop release

2. **[Comfy Desktop](https://github.com/Comfy-Org/Comfy-Desktop)**
   - Builds a new release using the latest stable core version

3. **[ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend)**
   - Every 2+ weeks frontend updates are merged into the core repository
   - Features are frozen for the upcoming core release
   - Development continues for the next release cycle

## Shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`              | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                          | Save workflow                                                                                                      |
| `Ctrl` + `O`                          | Load workflow                                                                                                      |
| `Ctrl` + `A`                          | Select all nodes                                                                                                   |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                              | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                                     |
| `H`                                  | Toggle visibility of history                                                                                       |
| `R`                                  | Refresh graph                                                                                                      |
| `F`                                  | Show/Hide menu                                                                                                      |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                                        |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| `Shift` + Drag                       | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

`Ctrl` can also be replaced with `Cmd` instead for macOS users

# Installing

## Windows and Mac

We highly recommend using the [desktop app](https://comfy.org/download):

### [Link to Download](https://comfy.org/download)

The desktop app is the easiest and best way to use ComfyUI for new users.

## Windows Portable

There is a portable standalone build for Windows that should work for running on Nvidia GPUs or for running on your CPU only. It is not recommended for regular users. Regular users should use the desktop app above.

[Direct link to download (nvidia)](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z)

Simply download, extract with [7-Zip](https://7-zip.org) or with the windows explorer on recent windows versions and run. For smaller models you normally only need to put the checkpoints (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints but many of the larger models have multiple files. Make sure to follow the instructions to know which subfolder to put them in ComfyUI\models\

If you have trouble extracting it, right click the file -> properties -> unblock

The portable above currently comes with python 3.13 and pytorch cuda 13.0. Update your Nvidia drivers if it doesn't start.

#### All Official Portable Downloads:

[Portable for AMD GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_amd.7z)

[Portable for Intel GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_intel.7z)

[Portable for Nvidia GPUs](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z) (supports 20 series and above).

[Portable for Nvidia GPUs with pytorch cuda 12.6 and python 3.12](https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia_cu126.7z) (Supports Nvidia 10 series and older GPUs, DO NOT USE THIS ON NEWER 20 SERIES AND ABOVE GPUS).

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.


## [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)

You can install and start ComfyUI using comfy-cli:
```bash
pip install comfy-cli
comfy install
```

## Manual Install (Windows, Linux)

Python 3.14 works but some custom nodes may have issues. The free threaded variant works but some dependencies will enable the GIL so it's not fully supported.

Python 3.13 is very well supported. If you have trouble with some custom node dependencies on 3.13 you can try 3.12

torch 2.7 is minimally supported but using a newer version is extremely recommended. Using a cu130 or above version of pytorch is required on Nvidia 20 series and above. Some features and optimizations might only work on newer versions. We generally recommend using the latest major version of pytorch with the latest cuda version unless it is less than 2 weeks old. If your pytorch is more than 6 months old, please update it.

### Instructions:

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae


### AMD GPUs (Linux)

AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2```

This is the command to install the nightly with ROCm 7.2 which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2```


### AMD GPUs (Windows, ROCm 10.0)

Use AMD's [multi-architecture PyTorch packages](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). The `device-*` extras install your GPU's kernels and the matching ROCm runtime automatically; a separate HIP SDK installation is not needed.

Use Windows 11, a current [AMD graphics driver](https://www.amd.com/en/support/download/drivers.html), and 64-bit Python 3.13.

The install command below uses `device-all` to install kernels for all supported GPUs. To reduce download size and disk usage, optionally replace **both** occurrences of `device-all` with the target for your GPU:

| GPU | Device extra |
| --- | --- |
| RX 9070 / XT, Radeon AI PRO R9700 | `device-gfx1201` |
| RX 9060 / XT | `device-gfx1200` |
| RX 7900 XT / XTX | `device-gfx1100` |
| RX 7700 XT / 7800 XT | `device-gfx1101` |
| RX 7600 / XT | `device-gfx1102` |
| Ryzen AI Max / Max+ (Strix Halo) | `device-gfx1151` |

**Note:** This table only lists examples. A GPU missing from it may still be supported: supported architectures include RDNA 2, RDNA 3, RDNA 3.5, and RDNA 4. Keep `device-all` to install kernels for all supported targets. For other models, see AMD's [GPU target table](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table) and [ROCm compatibility matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html).

**ROCm 10.0.0 with PyTorch 2.13:**

```bat
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-all]==2.13.0+rocm10.0.0" "torchvision[device-all]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
```

### Intel GPUs (Windows and Linux)

Intel Arc GPU users can install native PyTorch with torch.xpu support using pip. More information can be found [here](https://pytorch.org/docs/main/notes/get_start_xpu.html)

1. To install PyTorch xpu, use the following command:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu```

This is the command to install the Pytorch xpu nightly which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu```

### NVIDIA

Nvidia users should install stable pytorch using this command:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130```

This is the command to install pytorch nightly instead which might have performance improvements.

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Dependencies

Install the dependencies by opening your terminal inside the ComfyUI folder and:

```pip install -r requirements.txt```

After this you should have everything installed and can proceed to running ComfyUI.

### Others:

#### Apple Mac silicon

You can install ComfyUI in Apple Mac silicon (M1, M2, M3 or M4) with any recent macOS version.

1. Install pytorch nightly. For instructions, read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
1. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions for Windows and Linux.
1. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
1. Launch ComfyUI by running `python main.py`

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### Huawei Ascend NPUs

For models compatible with Huawei Ascend Extension for PyTorch (torch_npu). To get started, ensure your environment meets the prerequisites outlined on the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page. Here's a step-by-step guide tailored to your platform and installation method:

1. Begin by installing the recommended or newer kernel version for Linux as specified in the Installation page of torch-npu, if necessary.
2. Proceed with the installation of Ascend Basekit, which includes the driver, firmware, and CANN, following the instructions provided for your specific platform.
3. Next, install the necessary packages for torch-npu by adhering to the platform-specific instructions on the [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) page.
4. Finally, adhere to the [ComfyUI manual installation](#manual-install-windows-linux) guide for Linux. Once all components are installed, you can run ComfyUI as described earlier.

#### Cambricon MLUs

For models compatible with Cambricon Extension for PyTorch (torch_mlu). Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Cambricon CNToolkit by adhering to the platform-specific instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
2. Next, install the PyTorch(torch_mlu) following the instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html)
3. Launch ComfyUI by running `python main.py`

#### Iluvatar Corex

For models compatible with Iluvatar Extension for PyTorch. Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Iluvatar Corex Toolkit by adhering to the platform-specific instructions on the [Installation](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536)
2. Launch ComfyUI by running `python main.py`


## [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager/tree/manager-v4)

**ComfyUI-Manager** is an extension that allows you to easily install, update, and manage custom nodes for ComfyUI.

### Setup

1. Install the manager dependencies:
   ```bash
   pip install -r manager_requirements.txt
   ```

2. Enable the manager with the `--enable-manager` flag when running ComfyUI:
   ```bash
   python main.py --enable-manager
   ```

### Command Line Options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (implies `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |


# Running

```python main.py```

### For AMD cards not officially supported by ROCm

Try running it with this command if you have issues:

For 6700, 6600 and maybe other RDNA2 or older: ```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

For AMD 7600 and maybe other RDNA3 cards: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

### AMD ROCm Tips

You can try setting this env variable `PYTORCH_TUNABLEOP_ENABLED=1` which might speed things up at the cost of a very slow initial run.

# Notes

Only parts of the graph that have an output with all the correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.

Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.

To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```


## How to show high-quality previews?

Use ```--preview-method auto``` to enable previews.

The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI and launch it with `--preview-method taesd` to enable high-quality previews.

## How to use TLS/SSL?
Generate a self-signed certificate (not appropriate for shared/production use) and key by running the command: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"`

Use `--tls-keyfile key.pem --tls-certfile cert.pem` to enable TLS/SSL, the app will now be accessible with `https://...` instead of `http://...`.

> Note: Windows users can use [alexisrolland/docker-openssl](https://github.com/alexisrolland/docker-openssl) or one of the [3rd party binary distributions](https://wiki.openssl.org/index.php/Binaries) to run the command example above.
<br/><br/>If you use a container, note that the volume mount `-v` can be a relative path so `... -v ".\:/openssl-certs" ...` would create the key & cert files in the current directory of your command prompt or powershell terminal.

## Support and dev channel

[Discord](https://comfy.org/discord): Try the #help or #feedback channels.

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (it's like discord but open source).

See also: [https://www.comfy.org/](https://www.comfy.org/)

> _psst — we're hiring!_ Help build ComfyUI: [comfy.org/careers](https://www.comfy.org/careers)

## Frontend Development

As of August 15, 2024, we have transitioned to a new frontend, which is now hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend). The compiled JS files (from TS/Vue) are published to [pypi](https://pypi.org/project/comfyui-frontend-package) and installed as a dependency in ComfyUI.

### Reporting Issues and Requesting Features

For any bugs, issues, or feature requests related to the frontend, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend). This will help us manage and address frontend-specific concerns more efficiently.

### Using the Latest Frontend

The new frontend is now the default for ComfyUI. However, please note:

1. The frontend in the main ComfyUI repository is updated fortnightly.
2. Daily releases are available in the separate frontend repository.

To use the most up-to-date frontend version:

1. For the latest daily release, launch ComfyUI with this command line argument:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@latest
   ```

2. For a specific version, replace `latest` with the desired version number:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
   ```

This approach allows you to easily switch between the stable fortnightly release and the cutting-edge daily updates, or even specific versions for testing purposes.

# QA

### Which GPU should I buy for this?

[See this page for some recommendations](https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI)

## Worktree development

This fork also provides [scoped Qwen3-VL image-feature reuse](docs/vision-feature-reuse.md)
for paired conditioning encodes. The optional bounded scope preserves independent
text passes, private DeepStack outputs and ordinary execution outside the scope;
CPU fixtures cover its lifecycle and parity, while real-model GPU acceptance is
separate.

Run `COMFYUI_PYTHON=/path/to/prepared/comfyui/python python3 scripts/setup_worktree.py` before editing. The configured Python must be >=3.10 with PyTorch, einops and safetensors installed from the ComfyUI deployment's requirements. The check reuses that explicit runtime without copying dependencies, installing packages, loading models or connecting to a service. It is safe to rerun and leaves tracked files unchanged. GPU validation requires a compatible GPU/runtime and separately provisioned model files; unit tests use isolated fixtures.

## Local fork: pinned-memory and prefetch recovery

This fork is based on [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)
and its [upstream README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md).
Upstream supplies the node editor, local image/audio/video inference, model
loading and offloading, API and extension system documented above. Original
attribution and licensing remain unchanged.

The local repair makes a failed host-memory registration discard its temporary
allocation before evicting other registrations. Retrying allocates from the
buffer's current offset, avoiding stale pointers and invalid truncation. Partial
prefetch batches and their VBAR faults remain reachable to existing cleanup after
an exception. Successful sampling math and model interfaces are unchanged.

Activate core changes by restarting an idle instance; an already-running process
retains the old implementation and any stale pin state. This repair addresses
registration retry and exception cleanup, not every possible RAM/VRAM exhaustion.
It does not reduce the target resolution or the memory required by H3 refinement.
Third-party latent upscalers must separately use ComfyUI-managed model loading.

CPU regression checks (pytest and the prepared ComfyUI runtime required):

```bash
OMP_NUM_THREADS=1 "$COMFYUI_PYTHON" -m pytest -q tests-unit/comfy_test/test_pinned_memory_recovery.py tests-unit/comfy_test/test_prefetch_recovery.py
```

The tests inject registration/eviction and prefetch failures without GPU work.
Live CUDA inference and capacity limits require separate acceptance testing.

## Local fork: existing-mesh Trellis2 texturing

`Trellis2 VAE Encode Mesh` encodes an existing mesh (including a TripoSplat-derived
mesh) for native Trellis2 image-conditioned PBR texturing. It preserves faces while
centering/scaling the mesh, and provides exact surface subdivisions for texture
decoding. Use the returned mesh for UV unwrapping, baking and GLB export. The
optional CPU converter build requires Eigen 3, a C++20 compiler and the prepared
ComfyUI PyTorch runtime; weights are provisioned separately. See
[setup, workflow, contracts and limitations](docs/trellis2-mesh-texturing.md).
This extends the node engine, model loading and media workflows described in the
[upstream README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md).

## Local fork: prompt admission

`PromptExecutor.before_prompt_execution(self)` is a synchronous, no-op extension
hook invoked after interrupt and status initialization, before cache evaluation
or node execution. Overrides compose the previous hook and raise to refuse work.
Admission failures and Stop produce prompt-level history events with null node
identity and empty outputs, run lifecycle cleanup, and allow the next prompt.
Core owns no service settings or network calls.

Matching [Cutlery-Nodes](https://github.com/jackspooner/Cutlery-Nodes) uses this hook
for optional RuntimeHost GPU ownership on every queued CUDA-instance prompt,
including cached, CPU-only and remote-only work. CPU-configured instances bypass
GPU acquisition. Install both versions before restarting an idle instance; see
[GPU handoff, memory limits and recovery](https://github.com/jackspooner/Cutlery-Nodes/blob/main/docs/runtime-gpu-handoff.md).
Run `python -m pytest tests-unit/execution_test/test_prompt_admission.py -q` with
the prepared runtime; these checks do not start live GPU workloads.

Development prerequisites and focused native validation remain in [AGENTS.md](AGENTS.md#worktree-development). The [native integration guide](docs/native-model-integration.md) uses the prepared Python runtime directly; historical graph/lifecycle receipts are not setup requirements.
