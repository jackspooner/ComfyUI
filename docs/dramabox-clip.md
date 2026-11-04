# DramaBox shared CLIP loading

ComfyUI includes a native `dramabox` CLIP type for the Gemma 3 12B encoder and
DramaBox audio-components checkpoint used by the DramaBox custom-node pipeline.
The type pairs exactly one Gemma encoder with one audio-components safetensors
file and returns the standard `CLIP` handle.

Use **Load CLIP (Dual)** with both files under `models/text_encoders/` and select
`type=dramabox`. The custom node's **DramaBox CLIP Loader** remains available for
saved workflows and GGUF support, but delegates model recognition and CLIP
construction to the shared loader.

The native adapter preserves the all-layer Gemma hidden states, Hugging Face
left-padding RoPE positions, DramaBox audio connector and tokenizer. The video
connector is intentionally omitted because DramaBox consumes audio conditioning
only. Quantization metadata and dtype detection still follow ComfyUI's shared
text-encoder path.

The loaded `CLIP` records the selected Gemma checkpoint path and filename plus
the DramaBox audio-components path. `DramaBoxSaveConditioning` uses that
attribution for its `encoder` metadata, so saved conditioning identifies the
source encoder instead of reporting `unknown`.

Model loading and CPU fixture checks do not establish real-checkpoint audio
quality or serving activation. The DramaBox inference pipeline and its staged
conditioning and sampling nodes remain separate from the shared text-encoder
loader.

Focused validation:

```sh
python -m pytest tests-unit/comfy_test/dramabox_clip_test.py -q
```
