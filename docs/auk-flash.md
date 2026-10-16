# Native AuK-Flash integration

This branch adds native offline AuK MODEL, Qwen-Omni conditioning CLIP and audio VAE
components. **Development status: not deployed or accepted for production.** CPU
full-checkpoint comparisons and complete offline conversion pass. Native GPU BF16
generation, repeatability, cancellation and unload/reload run successfully. Listening,
edit effectiveness and longer resource tests remain required before release.
No ComfyUI service restart is part of this change.

## Sources and requirements

The implementation follows [Tencent AuK](https://github.com/Tencent-Hunyuan/AuK)
and its [MIT-licensed model code and prompt templates](../comfy/ldm/auk/LICENSE). The VAE also credits NVIDIA
BigVGAN and HiFi-GAN. Source revisions:

| Component | Revision |
| --- | --- |
| AuK code | `9481474f107ee4d8dec959b3aed05b0627cebbf8` |
| [AuK-Flash weights](https://huggingface.co/tencent/AuK-Flash) | `575b92f0895f75180bf2cbd35f2e176c5732b8ed` |
| [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` |

Use this fork's prepared Python runtime and dependencies. No custom node package,
cloud API or inference-time download is required. Initial support is original
source precision and BF16 transformer/conditioning inference; the VAE uses FP32.
Quantized exports and other GPU backends have not been validated. Measured CPU process memory and numerical comparisons are recorded in the
[acceptance report](auk-flash-acceptance.md); these short-case measurements do not establish general hardware requirements.

## Offline conversion

Provision the complete pinned AuK snapshot and Qwen repository through Download
Manager 2. First inspect existing jobs, files and the Hugging Face cache: an `hf
download` without `--local-dir` writes a cache snapshot, even when another checkout
contains Git LFS pointers. Reuse completed snapshots and resume existing jobs;
do not queue another copy because a different directory contains pointers.

Run from the repository, using absolute local input directories and a **new output
root**:

```sh
python scripts/convert_auk.py \
  --auk /path/to/complete/AuK-Flash \
  --qwen /path/to/complete/Qwen2.5-Omni-3B \
  --destination /path/to/converted-auk
```

The converter checks pinned SHA256 values, shard headers and inventories, and all
native parameter shapes before writing. It preserves retained tensor values and
checks the written values exactly. VAE weight normalization is folded; the unused
training flow, Qwen vision/Talker/vocoder/logits weights and unused audio boundary
embedding are explicitly accounted for. AuK's fusion weights accompany the
conditioning encoder. Checkpoint rotary frequencies are retained exactly, including
their training-time rounding.

Outputs:

- `diffusion_models/auk_flash.safetensors`
- `text_encoders/auk_qwen25_omni_3b.safetensors`
- `vae/auk_audio_vae.safetensors`
- `auk_flash.manifest.json`: source hashes, revisions and every tensor's disposition.
- `auk_flash_sources/`: licenses, source configuration, tokenizer data and model cards.

Preserve the originals. Point ComfyUI's model-path configuration at the output root,
or place the three artifacts under the corresponding existing model directories.
Tokenizer/configuration data are embedded in the CLIP artifact. The manifest and
source bundle should accompany redistribution. Conversion needs additional disk
space for the outputs, one temporary output file and CPU working memory. It does
not overwrite existing outputs. If interrupted, preserve completed files for
inspection and retry into a new empty output root; a manifest is written only after
all components pass verification. Do not load a partial conversion as a complete set.

## Reference workflows

Editor workflows and matching API graphs are in `docs/workflows/`:

- [Direct synthesis](workflows/auk_flash_synthesis.json) · [API](workflows/auk_flash_synthesis_api.json)
- [Reference voice or editing](workflows/auk_flash_reference.json) · [API](workflows/auk_flash_reference_api.json)
- [Optional MOSS helper](workflows/auk_flash_helper.json) · [API](workflows/auk_flash_helper_api.json)

These are development examples; see the remaining acceptance checks below. Select the actual
installed filenames and replace `reference.wav` with your input. The MODEL comes
from Load Diffusion Model; Load CLIP uses type `auk`; Load VAE uses the converted
AuK VAE. Supply a complete instruction, for example:

- Synthesis: `Based on the following description: "A warm conversational voice", generate speech content "Hello there.".`
- Cloning: `Say the following with the same voice: "Hello there."`
- Editing: `Change the emotion to happy.` with reference audio.

AuK Conditioning produces only CONDITIONING. With audio, connect the AuK VAE; it
encodes a mono 24 kHz reference with a separate reproducible `reference_seed`, while
Qwen receives 16 kHz audio features. Reference processing does not trim silence or
normalize loudness. Extremely short references are padded only enough for encoding.
The VAE retains the full waveform before its encoder; the valid reference-frame
mask excludes any encoder tail beyond the original duration. Generic VAE Encode
Audio can encode mono material, but its stochastic result is not independently
seeded; use AuK Conditioning for the reference workflow.

Empty AuK Latent Audio allocates `[batch, 64, ceil(seconds × 50)]`. Output is mono
24 kHz and therefore rounded up to the next 20 ms. Start with one sample and a short
utterance. Longer speech and references use more activation memory; maximum duration remains unmeasured. The tested three-case GPU sequence peaked at
9.08 GB process VRAM; see the acceptance report for its limits.

Use **Euler + BasicGuider + ManualSigmas + SamplerCustomAdvanced** with exactly:

```text
1.0, 0.9238795042037964, 0.7071067690849304, 0.3826833963394165, 0.0
```

This performs four conditional evaluations. Upstream time is `1 − sigma`; native
flow output is the negative upstream velocity. Conventional CFG 1 is equivalent
to no extra guidance, but the regular KSampler's default schedule is not the Flash
schedule. Do not substitute it. Reference and target latents remain separate until
the model combines them internally. Conditioning is computed once before sampling.

Set VAE Decode Audio's **normalize=false** to preserve the upstream waveform
amplitude. Its default remains true for existing workflows. Save Audio writes the
result through ComfyUI's existing output handling.

## Optional local prompt helper

AuK Prompt Helper takes a **separate generative CLIP**; AuK's conditioning encoder
has no generation head. The example uses the already supported converted MOSS-Audio
8B checkpoint. See [MOSS setup](moss-audio.md); the helper never downloads it.

Inputs are the user request, optional reference audio, optional verbatim transcript,
optional explicit duration and a text-generation budget. It returns instruction
and seconds, using the existing native Generate Text interface. Speech/replacement
text must be copied exactly from the request; malformed output or changed text
produces an actionable error instead of silently changing the request. An explicit
duration takes precedence. Disconnect the example's duration input to use estimation.

The local adaptation uses upstream task templates, weighted UTF-8 speech estimates,
reference-transcript duration ratios, speed scaling, content edits, emotion factors
and nonverbal adjustments. A local timing ratio can adjust synthesis cadence.
It does not run upstream cloud classification/ASR, text normalization, VAD trimming
or loudness processing. Supply a transcript when reliable edit-duration estimation
matters. One live MOSS helper synthesis case passed speech preservation and explicit duration;
other task selections remain unvalidated. Helper quality is model-dependent; bypass it
by wiring your own instruction and seconds directly. MOSS adds substantial memory
and generation time, managed by the standard CLIP loader/offloader.

## Verification

Fast isolated checks:

```sh
python -m pytest tests-unit/comfy_test/auk_test.py -q
```

Pinned upstream comparisons require a separate reference environment with
Transformers 4.57.1, torchdiffeq 0.2.5 and x-transformers 1.31.14, plus the local
pinned AuK source. Do not install these into a serving environment to run tests.

```sh
AUK_REFERENCE_SOURCE=/path/to/pinned/AuK \
AUK_REFERENCE_WEIGHTS=/path/to/complete/AuK-Flash \
AUK_QWEN_SOURCE=/path/to/complete/Qwen2.5-Omni-3B \
PYTHONPATH=.:/path/to/reference/dependencies \
/path/to/reference/python -m pytest tests-unit/comfy_test/auk_reference_test.py -q
```

Current evidence: 34 AuK checks, including the full-checkpoint VAE comparison, and
47 existing model-detection/MOSS regression checks pass on Linux/PyTorch 2.14.1.
All three example API graphs validate and their editor layouts pass geometry checks.
Full-weight synthesis, cloning and editing captures retain the inputs, token IDs,
audio features, all Qwen hidden states, fusion output, reference latents, exact
initial noise, every transformer velocity/Euler state and decoded audio.

The standard loader/BasicGuider/Euler path runs all three cases with four
evaluations. Synthesis repeats exactly; cancellation interrupts the next evaluation.
Local MOSS transcribed upstream/native synthesis and cloning consistently. One live
MOSS helper request preserved the speech and explicit duration. These are automated
checks, not listening acceptance. The happy-emotion edit was described as neutral
for both implementations, so successful emotion control remains unproven.

See the [acceptance report](auk-flash-acceptance.md) for stage errors, precision,
hardware and measured CPU memory. The RTX 5090 GPU path was tested with BF16, Comfy Kitchen and dynamic VRAM,
including model transitions, repeatability, unload/reload and cancellation. Still
required: listening, edit effectiveness, longer repetitions and forced low-memory
pressure. Do not infer bitwise equality across dtypes or attention backends.
