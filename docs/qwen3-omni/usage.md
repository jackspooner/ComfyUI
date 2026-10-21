# Native Qwen3-Omni operation

[Project index](README.md) · [acceptance and deployment status](status.md)

The installed demo **Qwen3-Omni - Text Image Audio.json** is in
`user/default/workflows/`, with two images and speech from
`input/qwen3_omni_demo/` (audio attribution in `credits.txt`). It has final/raw
previews and saves both outputs. Its explicit three-line prompt correctly returns
red square, blue circle and speech in both native and reconstructed-reference
runs. A more general combined prompt failed in both; preserve that known source
limitation rather than treating fluent output as proof. The matching
[API prompt](workflows/demo-api.json) uses `"sampling_mode":"off"`; the Python
node method receives the normalized dictionary instead.

Use **Load CLIP** with type **qwen3_omni** and the converted
`qwen3_omni_30b_a3b_nvfp4.safetensors`, then connect its CLIP output to
**Generate Text**. The single artifact contains the tokenizer, configuration,
chat template and processor assets. Inference does not read the original HF
folder or access the internet. Install it in the configured text-encoder root;
discover the live loader's filename/type choices after activating the source.

Connect text to `prompt`, the complete image batch to `image`, and one ordinary
ComfyUI AUDIO clip to `audio`. Any combination is supported. The default template
places images in batch order, then audio, then the prompt. All supplied media are
consumed. The existing batch node may resize images to a common batch shape before
Omni receives them. A raw template must contain one `<|image_pad|>` placeholder
per image and one `<|audio_pad|>` per audio clip, with the checkpoint's surrounding
modality tokens; prefer the default template unless reproducing an exact template.

Initial settings are **max_length 512, Sampling Mode off, thinking false,
MTP off, use_default_template true, context_budget 0**, with history disconnected.
`max_length` is the new-token budget including EOS. Zero uses remaining context;
it can reserve a substantially larger KV cache even when EOS arrives early.
Expanded media plus output must fit 65,536 tokens. Overflow is refused before
loading/generation workspace allocation. There is no silent media truncation.
System-prompt override, sampling controls and raw-template selection use the
existing node behavior. Omni supports explicit text/image history through the shared version 2 contract.
It has no switchable thinking mode; keep that control disabled. Video, speech output and diffusion
conditioning are outside this integration.

Generate Text output 0 is the final text, output 1 is separated thinking, and
output 4 is raw model text. Keep a raw preview/save alongside the final preview:
unfinished reasoning is not promoted to a final answer. Output indices and other
models' history behavior are unchanged.

The native RTX 5090 policy uses three residual activation passes through existing
NVFP4 tensor-core operations, FP16 partial outputs, FP32 summation/final bias,
and BF16 results/KV cache. Weights remain packed. This differs from the publisher's
unverified W4A8/FP8-cache runtime. One-pass NVFP4 was rejected for speech/click
regressions; decoded-weight execution is a diagnostic reference. Other models'
quantization defaults are unchanged. Measured limits and checkpoint mistakes are
recorded in the status document; good prose is not proof of accurate perception.

Use the normal ComfyUI queue and retain prompt IDs/history and saved outputs.
Repeated generations keep features and KV state local to each call. Cancellation,
model loading/offloading and memory admission belong to ComfyUI. If admission
fails, shorten the input/output budget or release idle models through the owner;
do not disable the allocator, overwrite model weights, or replay an uncertain
queued request. Check queue/history before retrying a lost submission.

For installation/activation, preserve launch flags and serving dependencies,
check ComfyUI queue, Workflows5 and GPU ownership, and restart only in an idle
maintenance window. Verify the loader schema, then run text/audio/image/combined
prompts through the queue. On activation failure restore the previous application
revision/runtime and verify an existing workflow, retaining the artifact and
diagnostics. Offline conversion interruption/retry is documented in
[quantization](quantization.md#offline-converter); validation commands and isolated
GPU-window ownership are in [reference](reference.md).

Text/image conversations use [versioned native history](../text-conversations.md), including automatic first-turn snapshots. Audio/video history remains unsupported; single-prompt audio and combined inputs retain their existing path.
