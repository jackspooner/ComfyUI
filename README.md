# ComfyUI fork

This fork extends [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) with
additional native model integrations, editable workflows and local execution
controls. It is intended for people building local image, video, audio, text
and 3D pipelines who need the additions described here.

For the base node editor, general installation, supported hardware and ordinary
workflow usage, read the [upstream README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md).
The current source includes upstream **v0.45.0**; historical backports in the
guides describe their original integration, not the current base version.

## What this fork adds

- **Native model integrations:** additional speech, singing, music, OCR,
  audio understanding and video-captioning paths use ComfyUI's model loading,
  attention, offloading and queue. The [feature index](docs/fork-features.md)
  links each integration's weights, conversion, workflow and acceptance limits.
- **Image and video workflows:** editable H3 reference-to-video and
  first/last-frame blueprints, contact sheets, control/editing composition and
  fractional-mask fixes. Krea editing optimizations and selected upstream
  backports are documented separately from original fork features.
- **Text and reference handling:** explicit text/image conversation history,
  context and image-count planning, memory-aware media sizing, constrained JSON
  on supported models, bounded component leases and scoped vision-feature reuse.
- **Queue controls:** persistent pause/resume for exact pending prompt IDs,
  plus opt-in maintenance admission fences and retained completion receipts.
  Running prompts cannot be paused, and unknown work is never automatically replayed.
- **Memory and media handling:** pinned-memory registration and partial-prefetch
  recovery, an opt-in archive-to-NVMe model cache, selective video extraction,
  external image references, and geometry interchange/texturing helpers.
- **Diagnostics:** prompt-free local timing observations, stage coverage and
  bounded live updates for consumers such as MediaTools. Forecasts belong to
  the consumer; these observations do not establish complete GPU timings or
  authorize maintenance.
  Custom inference also has [single-line console progress](docs/custom-inference-progress.md)
  with truthful counters or phase/elapsed reporting in the log panel. Existing
  sampler indicators are preserved; matching worker extensions need activation
  alongside core before their IPC phases are available.

## Use this fork

Native Qwen3-ASR with forced alignment and Whisper transcription are implemented
through ComfyUI model patchers, layer offloading and prefetch, with execution-local
encoder buffers retained across their native allocation scopes. See
[native transcription](docs/native-transcription.md) for offline conversion,
Cutlery workflow requirements and the outstanding GPU/quality acceptance gates.
Repeated dynamic-VRAM preparations with unchanged state are logged at DEBUG;
initial preparation, changed state and preparation after a full unload remain at INFO.

Native MOSS speech packages use offline pinned conversion. The speech converter
can explicitly admit hash-verified linked archive weights; see
[native speech preparation](docs/native-audio-grounding.md#offline-voicegenerator-conversion).
Native SoundEffect v2 supports an exact continuous `linear_flow` scheduler for
arbitrary step counts and model-owned sigma shifts; see
[SoundEffect setup and limits](docs/moss-soundeffect.md).
The original discrete SoundEffect checkpoint has a separate pinned converter and
`MossDiscreteSoundEffect` node; see [its codec and prompt contract](docs/native-audio-grounding.md#original-discrete-soundeffect).
Local v1.5 has a separate native Qwen3/NanoGPT2 loader and 48 kHz stereo
Tokenizer v2 codec, preserving twelve-book references and continuation; see
[conversion and compatibility](docs/native-audio-grounding.md#local-v15-and-tokenizer-v2).
Source and CPU comparisons do not establish serving activation or full-model quality.
YingMusic also has a pinned native vocal separator using the existing Roformer
components and its exact chunk preparation, plus native compatibility package
loading; see [YingMusic setup and acceptance limits](docs/yingmusic.md).
LocateAnything now has pinned native MoonViT/projector and modified Qwen2
components with its original MTP masks, decoding modes and request-private state;
see [conversion, graph and acceptance limits](docs/locateanything.md).
ViiTorVoice-NAR also has native masked Qwen3 and DualCodec/W2V-BERT components,
composed with Qwen alignment and original-PCM splicing. See
[preparation, edit controls and pending acceptance](docs/viitorvoice.md).

Clone this repository's `main` branch:

```bash
git clone --branch main https://github.com/jackspooner/ComfyUI.git
cd ComfyUI
```

Follow the [upstream manual installation instructions](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md#manual-install-windows-linux)
for your platform, using **this checkout's** requirements and compatible
PyTorch/GPU runtime. Then launch this checkout with `python main.py`.
Upstream desktop, portable and cloud products do not install these fork additions.

Provision model files separately and follow the selected feature's guide before
building its workflow. Some integrations need offline checkpoint conversion,
optional Python dependencies, specific GPU capabilities or separately installed
custom nodes. There is no single hardware or VRAM requirement that covers them
all. The H3 blueprints, for example, require their documented PDD/Cutlery packs;
those packs and weights are not bundled.

Use the normal ComfyUI queue and retain the returned prompt ID. Inspect the
running node schemas and available model files before submitting examples;
source presence does not mean an existing service has loaded the feature.

## Limits and compatibility

CPU tests, bounded checkpoint/GPU checks, output quality and service activation
are distinct. Many recorded trials used an RTX 5090; they do not establish
equivalent support or performance on another device. Experimental integrations
and failed quality checks remain identified in the feature guides.

This fork excludes upstream cloud-deployment upload paths and their persistent
headless marker. Local SSO and partner-token defaults remain disabled; retained
partner nodes are separate, explicit external integrations. Model and LoRA
licenses remain separate from ComfyUI's code license.

A source update does not change a running process. Follow the maintenance guide
before activation, preserving accepted work, configuration, models and recovery
records. Queue absence alone does not resolve held or uncertain work.

## Fork documentation

- [Feature index](docs/fork-features.md): model and workflow additions by modality.
- [Fork operations](docs/fork-operations.md): image paths, memory recovery,
  prompt admission and optional local workflow examples.
- [Queue controls](docs/queue-control.md), [maintenance fences](docs/maintenance.md)
  and [model cache](docs/model-cache.md).
- [Timing observations](docs/run-timing.md) and [video extraction](docs/video-extraction.md).
- [Native integration development](docs/native-model-integration.md),
  [development prerequisites](AGENTS.md#worktree-development) and
  [stable-release maintenance](docs/stable-release-maintenance.md).

## Credits and license

The node engine and its original capabilities are maintained by
[Comfy-Org and upstream contributors](https://github.com/Comfy-Org/ComfyUI).
Linked integration guides credit model authors and upstream backports.
ComfyUI's [GNU GPL v3 license](LICENSE) and existing notices remain applicable;
consult each model's terms before use.
