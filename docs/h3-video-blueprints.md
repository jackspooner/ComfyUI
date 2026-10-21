# MiniMax H3 blueprints

Open `blueprints/MiniMax H3 - Ref2V.json` or `blueprints/MiniMax H3 - FL2V.json`
as a workflow, or add the named blueprint from ComfyUI's subgraph library.
The installation also has copies in `user/default/workflows/` and
`user/default/subgraphs/`. Existing blueprint catalogues may be cached until
the next service start; open the saved workflow directly in that case.
Reload a previously opened workflow before editing its updated disk copy.

Both blueprints expose **Distillation**, **Attention**, **Override steps**,
**Steps**, **Duration (seconds)**, **Prompt**, **Width**, **Height**, **Seed**,
and the diffusion-model, text-encoder and video/audio-VAE selectors. Connect
their VIDEO output to Save Video. The workflow-library copies already include
Save Video; matching API prompts are in `user/default/workflows/api/`.
The output includes generated stereo audio. The blueprint does not save files
or submit work by itself.

## Presets and overrides

| Distillation | Default steps | Automatic attention | Video/audio shift |
| --- | ---: | --- | --- |
| None | 30 | Native Sol over Comfy Kitchen INT8 | 12 / 3 |
| LightX2V 8-step | 8 | Native Sol over Comfy Kitchen INT8 | 12 / 3 |
| LightX2V 4-step | 4 | Native Sol over Comfy Kitchen INT8 | Ref2V: 12 / 3; FL2V: 6 / 3 |
| PDD | 8 | Comfy Kitchen INT8 dense | 12 / 3 |

PDD's automatic dense choice is a conservative blueprint default, not a claim
that PDD requires dense attention. Explicit **Sol sparse**, **Comfy Kitchen
INT8 (dense)** and **PyTorch (dense)** override Automatic for every preset.
Sol uses the native BlockSparseAttention defaults: tau 1.3, start/end 0.2/1.0,
12288 minimum tokens, 256 extra tokens and exact text/reference/audio sinks.
Warmup and ineligible short calls intentionally remain dense.

**Override steps = false** ignores the Steps value and uses the preset count.
With **Override steps = true**, the integer drives the selected schedule.
PDD supports only 4, 6 or 8 evaluations through its native `nfe` contract;
other values fail in the existing PDD node. Other presets pass the override
through to BasicScheduler. Changing steps does not change the selected adapter.
PDD always uses Euler and its own trained sigma output; other presets use
`er_sde`, with `beta` for Ref2V and `sgm_uniform` for FL2V. Model branches are
lazy: selecting None does not load an adapter, and PDD never stacks turbo LoRAs.

Duration is converted at 24 fps and rounded upward to the native `17k+5`
frame grid; 5 seconds produces 124 frames (about 5.17 seconds). Width and
height use the native node's canvas contract. Longer clips and large canvases
increase local RAM/VRAM use; the advertised trained duration is about 5–15 seconds.

## References

Ref2V exposes three optional image references, one video-frame sequence at
24 fps, its optional soundtrack and one standalone audio reference. Use native
labels `<Picture 1>`, `<Picture 2>`, `<Video 1>` and `<Audio 1>` in the prompt.
Number only connected references in presentation order. A video soundtrack
precedes standalone audio, so when both are supplied they are `<Audio 1>` and
`<Audio 2>`. Reference sizing defaults to `match`; `max` costs more sampling
time. Additional references can be added to the inner conditioning node's
existing Autogrow inputs and exposed as extra subgraph ports.

FL2V exposes optional first and last frames. Connect both for first-to-last
generation, or just the first for image-to-video. Its prompt should describe
the motion between those images and any intended audio. Neither blueprint
requires a reference for the native text-only path.

## Requirements and validation

These are workflow artifacts for this fork, based on the native H3 nodes from
[upstream ComfyUI](https://github.com/Comfy-Org/ComfyUI). They need a frontend
with subgraph promotion, CustomCombo, native sparse attention and the
ComfyUI-MiniMax-H3-PDD-Acc and Cutlery-Nodes packs. All node types must be installed
even when a branch is inactive. No new backend nodes or runtime defaults are added.

Defaults use the installed `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
or `minimax_h3_fl2va_pruned_int8_convrot.safetensors`,
`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`,
`minimax_h3_video_vae_fp16.safetensors` and
`minimax_h3_audio_vae_fp32.safetensors`. Turbo adapters are under `minimax/`
in the LoRA selector; PDD uses its separate selector and corresponding trunk.
Use original H3 trunks, not already distilled/merged alternatives. FastH3/VSA
and SLA checkpoints are outside these blueprints because matching weights are
not installed. GPU support for the selected attention implementation is required;
PyTorch dense is the portable override. Provision missing weights explicitly.

Validation covers live node/filename discovery, saved graph/link and promoted
widget consistency, preset/override routing, CPU prompt validation and layout.
It does not establish frontend widget round-trip behavior or new GPU/media
quality acceptance. No generation, service restart or model download is needed
to install these artifacts.
