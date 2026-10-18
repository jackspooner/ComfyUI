# Native YingMusic Singer Plus

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

Use isolated vocals as references when possible. Vocal separation and accompaniment mixing remain separate audio operations; the native singing node does not silently run a separator. Long lyrics must fit the available latent frames. The initial path supports a single reference pair per execution. Memory grows with reference duration; no universal VRAM requirement is established. CPU mathematical comparisons do not establish GPU performance or subjective singing quality.

## Verification status

Small FP32 CPU DiT comparisons match both upstream conditional branches within 0.000001 absolute error. A small melody Conformer comparison matches exactly; the deterministic melody dropout agrees. All 366 DiT and 635 melody-encoder tensor shapes match the full checkpoint, and standard native loader construction succeeds. The complete CPU ComfyUI queue also generates and saves stereo audio with English and mixed English/Chinese lyrics, including the 32-step baseline. CPU queue cancellation completed in 0.73 seconds and generation after explicit unload/reload succeeded. On 18 October 2026, RTX 5090 generation passed with the 32-step baseline, targeted cancellation completed in 0.55 seconds, and explicit unload/reload succeeded. RouterAudio passed both direct reference generation and optional CPU vocal separation plus native accompaniment mixing. The generated clip is finite stereo 44.1 kHz audio, about 7.06 seconds long. This is bounded integration acceptance, not a singing-quality, multilingual or low-VRAM benchmark.
