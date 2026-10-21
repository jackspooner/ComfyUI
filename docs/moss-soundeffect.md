# Native MOSS SoundEffect v2 candidate

This fork implements the continuous-audio SoundEffect v2 architecture as native
ComfyUI MODEL, CLIP and VAE components. It does not use the discrete MOSS speech
codec. The serving deployment and existing model files are unchanged.

Upstream: [OpenMOSS MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), code revision
`934d6826b084c46a0d033402174d5f8ac4ed2519`; weights
[OpenMOSS-Team/MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
revision `e35df4d82fbe87fcd5d14e5d100e349c0c3c076d`. Both declare Apache-2.0;
the DAC architecture also carries Descript's MIT attribution in
`comfy/ldm/audio/moss_soundeffect.LICENSE`.

## Preparation

Install `requirements-moss-soundeffect.txt` into the explicitly selected runtime
using the normal dependency provisioning process. `ftfy` reproduces the upstream
Unicode/HTML prompt cleaning; the runtime performs no downloads. Run:

```sh
COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py --moss-soundeffect
python -m scripts.convert_moss_soundeffect --source /absolute/original-bundle --component transformer --destination /absolute/models/diffusion_models/moss_soundeffect_dit_native_fp32.safetensors
python -m scripts.convert_moss_soundeffect --source /absolute/original-bundle --component text_encoder --destination /absolute/models/text_encoders/moss_soundeffect_text_bf16.safetensors
python -m scripts.convert_moss_soundeffect --source /absolute/original-bundle --component vae --destination /absolute/models/vae/moss_soundeffect_vae_fp32.safetensors
```

The offline converter verifies the pinned SHA256 inventory before reading weights.
VAE input uses restricted `torch.load(weights_only=True, map_location='cpu', mmap=True)`;
legacy convolution weight normalization is folded in FP32. No upstream model code
is executed. Diffusers transformer names are mapped to native Wan names with every
tensor retained. The text checkpoint preserves the unused tied LM head as an accounted
tensor, but does not allocate it during conditioning. Each output has a verified
payload manifest. Existing outputs/manifests are refused; use a new destination
for retries and preserve originals. Conversion needs roughly 11.3 GB output space
plus a temporary VAE file and the original bundle; it is CPU-only.

| Component | Native payload | Initial compute |
| --- | ---: | --- |
| Flow transformer | 5,664,190,976 bytes | FP32 |
| Qwen3 conditioning encoder | 4,063,479,808 bytes plus tokenizer | BF16 |
| Continuous DAC VAE | 1,486,156,548 bytes | FP32 |

## Workflow

Use `UNETLoader`, `CLIPLoader(type=moss_soundeffect)` and `VAELoader` for the three
prepared files. Positive `CLIPTextEncode` text should include the duration suffix,
for example `Gentle rain on a wooden roof. duration: 5.0s`. Encode the negative
prompt separately without a duration suffix. Prompt cleaning applies `ftfy`, two
HTML unescapes and whitespace normalization. The encoder right-pads to 512 tokens
and zeros padding hidden states. The diffusion model still attends to all padded
positions, as upstream does. Longer prompts are preserved instead of truncated;
their quality and memory use are unverified.

Use `EmptyMossSoundEffectLatent(seconds=30, batch_size=1)` and `KSampler` with
100 steps, Euler, `simple`, CFG 4 and denoise 1. The model owns shift 5 and timestep
multiplier 1000. The reference always generates 1500 latent frames, then crops the
waveform; the node also permits other lengths without imposing a quality limit.
`VAEDecodeAudio(normalize=false)` preserves decoder amplitude. Finish with
`TrimAudioDuration(start_index=0, duration=5)` and `SaveAudio`. Output is 48 kHz mono,
128 latent channels and 960 samples per latent frame. Reference audio, speech
history and discrete audio codes are not inputs to this model.

The local editable example is `Audio Models/MOSS SoundEffect v2 - Native Candidate.json`
in the configured user workflow library, with a matching `.api.json`. It requires
the candidate nodes and converted model filenames; source support does not make
those nodes available on an older running server. Reload the saved workflow before
editing it. The 100-step simple schedule matches the reference grid; other step
counts use ComfyUI's discretized schedule and are not guaranteed to match upstream.

## Verification and limits

Run `python -m pytest tests-unit/comfy_test/moss_soundeffect_test.py -q`. Set
`MOSS_SOUNDEFFECT_SOURCE` to the pinned source's `moss_soundeffect_v2` directory
for independent tiny-model comparisons, and `MOSS_SOUNDEFFECT_WEIGHTS` to the
original bundle for tokenizer and actual DAC checkpoint tests. Tests use CPU
fixtures and do not start a service or mutate model files.

Tiny continuous encode/decode, Qwen text and flow-model comparisons pass, including
odd strides, padding and batch size two. Actual checkpoint DAC decode, FP32 Qwen
conditioning with zeroed padding, complete
transformer key coverage/native construction and FP32 transformer comparisons pass,
including first/last blocks and all 30 blocks on a three-frame fixture (absolute
and relative tolerance 2e-4). This is not full-length upstream pipeline parity.

Isolated RTX 5090/CUDA validation on 21 October 2026 passed cold/warm generation,
sampling cancellation, recovery, explicit unload, a VoiceGenerator transition and
SoundEffect reload. Four outputs are finite 48 kHz mono files with 240,000 samples.
Cold/warm wall times were 50.5/26.2 seconds including GPU-admission waits of
11.1/6.1 seconds; the 30-second latent was decoded before cropping to five seconds.
Process lifetime peak RSS reached 3.67 GB. Torch peaks were 2.25 GB allocated and
4.55 GB reserved, excluding aimdo-managed weights; end-of-prompt device usage
snapshots ranged from 14.7 to 28.8 GB and include other device allocations. These
are observations, not a peak-VRAM guarantee or validation of low-memory estimates.

Independent Audio Flamingo 3 described the dog fixture as clear rhythmic barking.
The quiet rain seed had peak amplitude 0.0102/RMS 0.000974 and was classified as
silent, so that fixture failed the content-quality check. Other rain seeds had
higher amplitude; listening quality and seed consistency remain unaccepted.
FP32 is the initial diffusion baseline and does not claim parity with upstream
BF16 autocast. Reduced precision, quantization, other devices, long prompts and
generic approximate VAE tiling remain unverified. No serving service was restarted
or activated. Failed loader attempts and successful normal-queue prompt histories
are retained in the local task evidence directory.
