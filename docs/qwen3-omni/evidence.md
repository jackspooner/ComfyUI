# Evidence, decisions and open questions

[Project index](README.md)

Evidence date: 16 October 2026. Local source baseline:
`0d2d2221bcd9b4895cfb0930864a1a72e2888374`. Remote `main` URLs below are mutable;
pin full revisions during M0. The initial specification involved no payload
download or execution. Subsequent [execution evidence](status.md) records the
pinned download, actual layout and unresolved acceptance gates.

## Source observations

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| [Community model card](https://huggingface.co/shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4) | Publisher describes packed NVFP4 experts, quantized Thinker attention and BF16 media encoders | Correct payload layout, functioning runtime, speed or quality |
| [Community config](https://huggingface.co/shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4/blob/main/config.json) | Declared architecture, 48 Thinker layers, 128 experts/8 selected, W4A8 settings | Actual executed activation precision |
| [Quantization config](https://huggingface.co/shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4/blob/main/hf_quant_config.json) | ModelOpt 0.42.0 producer, NVFP4/group size 16 and FP8 cache declarations | Cache support in ComfyUI or preserved calibration behavior |
| [File inventory](https://huggingface.co/shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4/tree/main) | Seven safetensors shards plus index and processor/tokenizer assets; approximately 25.9 GB listed | Converted Thinker size, tensor completeness or peak VRAM |
| [Official Transformers Omni documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_omni_moe#text-only-generation) | A Thinker-only model can generate text from multimodal input | This community quant's reference runtime is working |
| [Official Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni) | Architecture, upstream implementation and model usage reference | Native ComfyUI integration |

Local source inspected directly:

- [Generate Text](../../comfy_extras/nodes_textgen.py): CLIP input and optional
  image/audio/video inputs; calls tokenize/generate/decode.
- [CLIP wrapper](../../comfy/sd.py): delegates generation through native model
  loading and quantized-operation context.
- [Quantized operations](../../comfy/ops.py) and
  [format registry](../../comfy/quant_ops.py): NVFP4 scales, expert banks and
  layout-based activation quantization are implemented.
- [Qwen3-VL](../../comfy/text_encoders/qwen3vl.py),
  [MOSS-Audio](../../comfy/text_encoders/moss_audio.py) and
  [MOSS conversion](../../scripts/convert_moss_audio.py): reusable integration
  precedents, not proven drop-in Omni implementations.

These links are source evidence, not proof that a running service contains those
versions. The initial canonical GitNexus ensure was refused with
`validation-refused`: candidate analysis lacked a complete authenticated database
and receipt. No graph completeness claim is made here; owner recovery is required
before graph-dependent implementation. During document preparation, the separate
task worktree's owner ensure successfully reused an authenticated index at the
baseline commit, with embeddings disabled and no model requests. This does not
retroactively certify the earlier canonical checkout or a future implementation
checkout; ensure each exact checkout again before graph use.

## Decisions

| ID | Decision | Rationale |
| --- | --- | --- |
| D1 | Target the named community NVFP4 Instruct checkpoint | User-selected source; avoid an unnecessary full BF16 conversion prerequisite |
| D2 | Require text, audio, image and combined inputs | User clarified that media support is part of the completed result |
| D3 | Keep Thinker and input encoders; omit speech output | Matches the requested text output and avoids unused modules |
| D4 | Use native CLIP plus existing Generate Text | Existing socket and lifecycle contracts already fit the goal |
| D5 | Establish quantization feasibility before full port | Storage compatibility does not establish numerical execution compatibility |
| D6 | Validate BF16 first only at small reference/operation scope where useful | A full original BF16 model is not a mandatory initial download |

## Questions to resolve with evidence

- Are experts fused or split in the actual current shards, and are their scales
  per expert, per tensor, or stored in another layout?
- Does a pinned source runtime execute the advertised W4A8 and FP8-cache scheme?
- Can packed weights/scales map exactly into native layouts without requantization?
- Which available kernel path preserves acceptable quality and performance?
- Which calibration tensors are essential, and how will an alternate path be
  explicitly represented and tested?
- What is the actual Thinker-only footprint with both media encoders, realistic
  expanded inputs and generation cache on the target GPU?
- Can current native preprocessing and generation helpers support Omni's exact
  modality positions and feature injection with narrow changes?

Resolve these in milestone evidence records. Do not convert unanswered questions
into claims of supported behavior in README, skills or workflow descriptions.
