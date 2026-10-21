# Native integration design

[Project index](README.md)

## Data flow

```text
Local HF shards + tokenizer + processor/config files
                     |
              offline converter
                     |
       ComfyUI artifact + provenance manifest
                     |
           Load CLIP (qwen3_omni, proposed)
                     |
Text + IMAGE + AUDIO -> Generate Text -> CLIP tokenizer/media preparation
                     -> Thinker prefill and cached token generation
                     -> decode -> generated_text / thinking
```

The artifact must include the complete Thinker language model and output head,
audio encoder, vision encoder and all required projection/merger weights.
Excluding speech-output modules does not exclude audio understanding. Preserve
the checkpoint's tokenizer, special tokens, template and processor settings.

## Existing owners and proposed changes

| Owner | Responsibility |
| --- | --- |
| `nodes.py`, `comfy/sd.py` | Add loader selection and unambiguous model detection/target construction |
| Proposed `comfy/text_encoders/qwen3_omni.py` | Omni architecture, modality embedding and position construction |
| Existing tokenizer interfaces | Turn prompt/media into the structured tokens consumed by the model adapter |
| `comfy_extras/nodes_textgen.py` | Existing user inputs, sampling controls and output text/reasoning separation |
| `comfy/text_encoders/llama.py` | Reuse compatible generation and expert execution helpers |
| `comfy/ops.py`, `comfy/quant_ops.py`, Comfy Kitchen | Weight formats, optimized operations and backend dispatch |
| Model patcher, model management and prefetch | Residency, cast/offload policy, loading and cancellation integration |
| Proposed offline script under `scripts/` | Local conversion, validation and provenance; no model execution or downloads |

The current node already passes image and audio to `clip.tokenize`, then calls
`clip.generate` and `clip.decode`. Aim to leave this interface unchanged. Extend
a shared helper only for a concrete reusable requirement demonstrated by the port.
Keep model-specific interpretation inside the tokenizer/model adapter.

Use Qwen3-VL and MOSS-Audio as integration examples, not as proof of identical
math or preprocessing. Validate audio feature extraction, image normalization,
feature injection, attention masks and multimodal positions against Omni itself.
The native MoE path must match Omni's routing and expert combination exactly.

## Resource and compatibility rules

Use existing optimized operations, `AttentionTensorContainer`/`ComfyAttention`
and memory compiler/prefetch patterns where required by repository guidance.
Keep KV caches and media feature caches scoped to one generation. Do not retain
large tensors on model instances across executions or introduce private residency
policy. Account for expanded media and generation cache in load estimates.

Detection must guard every key it reads and avoid stealing existing Qwen model
signatures. Packed dimensions cannot be treated as original linear input sizes.
Preserve existing optional generation arguments; unsupported acceleration options
must not change model semantics. Model/input failures use established ComfyUI
errors; missing local files never trigger background downloads.

No new HTTP/MCP interface is planned. Revisit the API-contract owner guidance if
implementation introduces one. Do not modify unrelated execution or queue layers.
