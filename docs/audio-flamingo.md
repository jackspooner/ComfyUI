# Native Audio Flamingo 3

This fork implements NVIDIA Audio Flamingo 3 through standard Load CLIP and Generate Text. The native Whisper encoder, two-layer audio projector and Qwen2 decoder reuse ComfyUI operations, selected attention, ModelPatcher, prefetch, KV cache, cancellation and offloading. No Transformers model or standalone runner executes inference. Transformers supplies local tokenization and log-mel preprocessing only.

[NVIDIA's model](https://huggingface.co/nvidia/audio-flamingo-3-hf) is for non-commercial research under its linked NVIDIA OneWay Noncommercial License. The architecture follows the Apache-2.0 Transformers implementation, copyright NVIDIA Corporation and Hugging Face. Preserve the original model card and terms when repackaging weights.

## Offline preparation

Provision the complete `nvidia/audio-flamingo-3-hf` snapshot through Download Manager 2, including all four shards, index, config, processor and tokenizer files. The tested source revision is `7d4bae64ee29878af6504ae6f6bb3e40492838ad`. The normal ComfyUI environment supplies the needed libraries; no extra environment is required.

```
COMFYUI_PYTHON=/path/to/prepared/python python3 scripts/setup_worktree.py
/path/to/prepared/python scripts/convert_audio_flamingo.py --source /path/to/hf-snapshot --destination /path/to/models/text_encoders/audio_flamingo_3_bf16.safetensors --revision 7d4bae64ee29878af6504ae6f6bb3e40492838ad
```

The converter validates the exact configuration, all 830 BF16 tensor shapes, shard containment and payload ranges, then streams one checkpoint with its embedded tokenizer and a tensor-hash manifest. It uses the existing streaming conversion helper, preserves original files, refuses existing destinations and downloads nothing. Keep approximately 16 GB of free space for the converted checkpoint plus the original snapshot. Failed temporary files are cleaned; after fixing a missing/corrupt source, retry the same command.

## Use

Select `audio_flamingo` in Load CLIP with the converted file. Connect standard AUDIO and an instruction to Generate Text, then preview/save its text output. Start with sampling off and 256 maximum new tokens. Request a concise description of foreground/background sounds and speech; use “Transcribe the input speech.” for transcription. Descriptions may invent details and are not ground truth. Text-only prompts and explicit text conversation history also work; raw-template mode with audio requires one `<sound>` placeholder.

Input is downmixed/resampled to 16 kHz and processed in 30-second windows. All supplied windows are retained; unlike the upstream processor's ten-minute crop, the native path relies on the actual 32,768-token context limit. Output budget zero uses remaining context; large output budgets increase KV memory. The base checkpoint does not include the optional AF-Think adapter or voice generation. A thinking checkbox does not load those weights. Quantized variants, long-audio quality and device-specific capacity are not established by BF16 unit tests.

## Verification

Ten CPU checks cover reference encoder math with partial-window masks, pooling/positions, Qwen prefill and cached decoding, guarded detection, and real local processor/tokenizer parity across a window boundary. All 830 native tensor shapes match the complete checkpoint, and streaming conversion completed with payload hash verification. On 18 October 2026, the complete converted checkpoint passed RTX 5090 queue inference, speech transcription and music-description RouterAudio jobs, targeted cancellation (0.10 seconds), and explicit unload/reload. A transition from native YingMusic back to Audio Flamingo also passed. These are short-clip acceptance checks; long-audio quality, low-VRAM behavior and other GPU backends remain unverified. Discover live node schemas and file choices before claiming readiness. Submit through the normal ComfyUI queue; recover the returned prompt ID after a timeout before any retry.
