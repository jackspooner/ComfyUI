# Native YingMusic Singer Plus

The shared MelBandRoformer selects attention through the core's device-aware
selector. An explicit CPU separator uses the existing CPU implementation even
when the server's default attention is CUDA-only. GPU execution and explicit
compatible per-model attention remain under native selection. Test full CPU
separation through the main queue as well as the normal GPU path; tiny fixtures
alone do not establish full-checkpoint readiness.

This fork adapts [ASLP-lab YingMusic Singer Plus](https://github.com/ASLP-lab/YingMusic-Singer-Plus), source revision `baa409c2e7e5e775f09b4e92a220808f4827d2cc`.
The DiT, melody Conformer and existing AudioOobleck VAE run inside ComfyUI.
Native loaders, ModelPatcher, attention selection, prefetch, queue progress and cancellation own execution.
The upstream CC BY 4.0 license and Amphion tokenizer credits are retained under `comfy/ldm/yingmusic/` and `comfy/text_encoders/yingmusic/`.

## Preparation

Provision the non-distilled `ASLP-lab/YingMusic-Singer-Plus` snapshot and pinned upstream source through Download Manager 2. Preserve their manifests; the converter does not download files.
Install the optional `requirements-yingmusic.txt` dependencies from staged, verified artifacts, plus the system espeak-ng shared library. These supplement the existing ComfyUI runtime. No alternate Torch or Transformers is required.

```sh
COMFYUI_PYTHON=/path/to/prepared/python python3 scripts/setup_worktree.py --yingmusic
/path/to/prepared/python scripts/convert_yingmusic.py --source /path/to/model-snapshot --source-code /path/to/upstream-source --destination /path/to/new-model-root
```

Conversion writes `yingmusic_singer_plus.safetensors` under `diffusion_models`, `audio_encoders` and `vae`, lyric assets under `text_encoders/yingmusic_singer_plus`, and a component hash manifest. It excludes the unused student MIDI network. Register the completed category directories through the existing ComfyUI model-root configuration. Existing destination files are never overwritten. If conversion fails, retain the original snapshot, inspect the exception and use a fresh destination after resolving it.

## Workflow

Use the standard diffusion, audio-encoder and VAE loaders with their converted files. Connect them to **YingMusic Generate / Edit Singing**, alongside ordinary AUDIO inputs for a voice example and the desired melody. The voice example's lyrics must describe that audio; target lyrics replace the melody's singing. Melody duration determines output duration. Chinese, English and mixed lyrics use the original pronunciation rules; arbitrary languages are not established.

The baseline is 32 Euler steps, guidance 3, time shift 0.5 and 0.5 seconds of reference silence. These are the upstream controls: guidance 3 corresponds to ordinary ComfyUI CFG 4. The native flow path reverses upstream time and velocity together. The model has 64-channel, 44.1 kHz audio latents at 2048 samples per frame. The output trims the voice prefix and trailing one-second conditioning region and applies the upstream 150 ms ending fade. Use Save Audio/Preview Audio to consume the result.

Use isolated vocals as references when possible. Vocal separation and accompaniment mixing remain explicit graph operations. Long lyrics must fit the available latent frames. The initial path supports a single reference pair per execution. Memory grows with reference duration; no universal VRAM requirement is established. CPU mathematical comparisons do not establish GPU performance or subjective singing quality.

## Native separator and compatibility packages

The matching source adds `YingMusicSeparatorLoader` → `YingMusicSeparateVocals`.
It reuses the native MelBandRoformer and patchers with YingMusic's exact one-stem
checkpoint and preparation. The checkpoint revision is
`9b3f444f2bccd77fbc03e32eb7c334fc96040b8d`, SHA256
`87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`.
Its 684 FP32 tensors occupy about 0.913 GB. The bundled separator's MIT credit
and license are retained in `comfy/yingmusic_separator.LICENSE`.

```sh
python -m scripts.convert_yingmusic --component separator --source /absolute/YingMusic-Singer-Plus --source-code /absolute/pinned-source --destination /absolute/YingMusic-Singer-Plus.comfyui-native
```

The converter verifies the pinned checkpoint, config and four implementation
files, the complete native shape inventory and every emitted tensor. It writes
`audio_encoders/yingmusic_separator.safetensors` and its `.manifest.json` beside
the unchanged original bundle. Existing artifacts are refused. Expose this file
through the configured native `audio_encoders` roots; retain its manifest.
The loader supports native selected-device or explicit CPU execution.

Preparation retains eight-second chunks, overlap two, batches of four, linear
edge fades, conditional reflected borders and final padding. The original
batch-level first/last-window decisions are preserved, including zero-counter
cleanup at edges. YingMusic's configured DC-bin suppression is retained; the
shared Roformer's existing default stays unchanged. Audio resamples to 44.1 kHz and mono duplicates to stereo;
accompaniment is the resampled original minus estimated vocals. This differs
from SoulX's five-second/Hann preparation. Native FP32 is the provisional quality
baseline; existing attention selection remains available. Full-model GPU kernel,
content, memory-pressure and listening comparisons are still required.

For existing Cutlery generation/segmentation IDs, also run the ordinary singer
converter with the same sibling destination. `YingMusicPackageLoader` composes
the existing native diffusion, VAE and audio-encoder loading with explicit
auto/CPU/native-GPU placement and returns their exact lyric-asset path. The
package VAE retains its conversion metadata and uses the standard file-backed
VAE reconstruction factory, including `SelectVAEDevice` retargeting. The selected
loader device is retained when reconstructing weights; the device selector owns
the clone's target device. Invalid VAE weights fail during package loading. The
generation node's optional `lyric_assets` input selects that converted asset set;
omitting it preserves configured text-encoder discovery. Existing standard-loader
graphs remain valid. Cutlery compatibility macros expand to inspectable component
graphs, preserving lyric/duration fitting, crop ordering, seeds, stems and mixing.
New RouterAudio v8 stages bind the exact separator file and manifest; accepted
adapters through v7 retain worker separation and their original checkpoints.

## Verification status

Tiny CPU package fixtures cover the loader's auto/CPU/native-GPU settings,
metadata, invalid VAE rejection and real patcher reconstruction through
`SelectVAEDevice`, including restoration of the original device. GPU resolution
is mocked and all tensors remain on CPU; these checks do not establish hardware
placement, full-checkpoint retargeting or multi-GPU acceptance.

The new separator's ten pinned CPU preparation fixtures match exactly across
short, boundary and multi-batch lengths. Native singing/SoulX regression checks
also pass; the full converted singing package loads on CPU with all three native
owners. Worker main-queue reference `6737d562-cd89-4889-8d75-b5742c9b6e89`
completed. These new source/package checks do not activate the separator or
establish native full-model quality. The following older singing acceptance
does not cover this separator migration.
An independent small complete Roformer comparison executes the pinned source
attention, rotary-embedding-torch 0.8.9 and released model bodies on CPU. It matches
the native waveform with DC suppression and verifies that omitting the filter
changes the output. This fixture loads no full checkpoint and uses no GPU.

Small FP32 CPU DiT comparisons match both upstream conditional branches within 0.000001 absolute error. A small melody Conformer comparison matches exactly; the deterministic melody dropout agrees. All 366 DiT and 635 melody-encoder tensor shapes match the full checkpoint, and standard native loader construction succeeds. The complete CPU ComfyUI queue also generates and saves stereo audio with English and mixed English/Chinese lyrics, including the 32-step baseline. CPU queue cancellation completed in 0.73 seconds and generation after explicit unload/reload succeeded. On 18 October 2026, RTX 5090 generation passed with the 32-step baseline, targeted cancellation completed in 0.55 seconds, and explicit unload/reload succeeded. RouterAudio passed both direct reference generation and optional CPU vocal separation plus native accompaniment mixing. The generated clip is finite stereo 44.1 kHz audio, about 7.06 seconds long. This is bounded integration acceptance, not a singing-quality, multilingual or low-VRAM benchmark.
