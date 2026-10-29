# H3 structural control and selective editing

The native `ModelPatchLoader` and `MiniMaxH3FunControlNetApply` support Fun
ControlNet Union v1/v2. This fork preserves ControlNet block patches when native
`BlockSparseAttention` is applied afterward, including their weight and cleanup
hooks. The control stream uses dense attention rather than the base stream's
sparse routing. Apply sparse before control on older deployments. This does not
certify third-party sparse patches; SolAttn Morton token reordering must remain
disabled for raster-order control injection.

Repeated native sparse nodes compose in node order: the outer node tries its
policy first and delegates ineligible calls through its predecessor. Only the
outer node reinstalls attention at prepare time, so repeated sampling does not
grow the chain. Each patch retains its own cleanup callback; obsolete attention
closures are weakly tracked. Dense backend overrides remain opaque callables.

Control residuals are added only to packed target `video` rows. Text,
keyframe/reference image, reference audio and target audio rows receive no direct
residual. `/features` advertises `minimax_h3_control_target_only: true` for this
contract. This permits a compiler to preserve reference conditioning while
adding structural control; it does not certify RefMod workflow execution or
visual quality. Base/control AdaLN width matching still applies.

## Checkpoint and guide selection

The [Alibaba Union 2.0 model card](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0)
specifies ten control blocks at base layers 0,5,...,45, 49 input channels and
`post_norm` inpainting. Eight guide representations share one checkpoint: Canny,
depth, HED, MLSD, pose, scribble, layout and grayscale. These are preprocessed RGB
frame batches, not a node enum or an automatic estimator. Layout uses colored
subject boxes on white backgrounds. The control branch does not directly control
audio. The weights use the MiniMax H3 Community License, linked from that card;
retain its terms independently of ComfyUI's license.

Match the base representation: pruned curve-basis H3 needs a compatible
`adaln_basis` control patch such as the [Kijai model-patch conversion](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main/model_patches).
The original full-width AdaLN checkpoint is not interchangeable. A selector name
alone is not readiness: inspect safetensors metadata and tensor inventory.
For v2 require ten complete control blocks, `minimax_h3_fun_controlnet=adaln_basis`
for a curve-basis base, and `inpaint_masked_pixel_mode=post_norm`. Preserve
quantization metadata. No downloads occur while loading or applying a patch.

Load from `model_patches`, then connect `model`, `model_patch`, `vae`, `strength`,
`start_percent`, `end_percent` to `MiniMaxH3FunControlNetApply`. Optional
`control_video` is IMAGE; `mask` is MASK with white/1 regenerating; `source_video`
is IMAGE and is read only with a mask. The node fits guide/source frames and
holds their last frame when short; use aligned media when temporal identity
matters. Strength zero or absent guide/mask is a no-op. The source is conditioning,
not a promise of pixel-exact preservation. The primary model recommends CFG 1.

## LanPaint is a separate native extension

[scraed/LanPaint](https://github.com/scraed/LanPaint) supplies its own ComfyUI
samplers and AV nodes under GPLv3. Source inspected at
`2d7912f9a5efe5ece8de334c7ca18317b8288c39` (2026-10-29); it is not bundled with
this core. Install a pinned extension in the custom-node owner and activate only
through the deployment procedure. It requires the existing H3 model, text encoder,
video/audio VAEs and native torch, torchaudio and video decoding dependencies;
it adds no model checkpoint. Its sampler performs additional Langevin iterations,
so ordinary sampling estimates do not establish its runtime or VRAM needs.

Compose `LoadVideo` → `LanPaint_AVEncode(video,vae,audio_vae,mask,audio_mask)` →
`LanPaint_SamplerCustomAdvanced` → `LanPaint_AVDecode` → `SaveVideo`.
Use the normal noise, guider, sampler and sigmas; the custom sampler also takes
`LanPaint_NumSteps`, `LanPaint_Lambda`, `LanPaint_StepSize`, `LanPaint_PromptMode`
and `LanPaint_Info`. The encode output is a nested video/audio LATENT with nested
noise masks. The decode node takes the same original VIDEO and masks, plus
`blend_overlap` (odd pixels, default 11) and `audio_crossfade` (seconds, default .02),
and returns VIDEO and AUDIO. AVEncode requires a soundtrack; Cutlery's source
preparation helper supplies explicit silence when the source has none.

The spatial mask is `[frames,height,width]`, and the audio mask is `[frames]`
at source frame rate. A zero audio mask keeps the soundtrack; a zero spatial
mask keeps the picture during final blending. Cutlery's `CutleryVideoEditMasks`
can prepare these from explicit intensity masks, admitted per-frame PNGs and
audio intervals without the interactive editor. Do not use image alpha as the
regeneration mask. Preserve the source timing and frame identity in the caller.
The upstream pin above point-samples masks and can lose single-frame strokes.
The separately maintained coverage fix max-pools all source cells intersecting
each normalized latent bin, then retains the existing two-slice temporal dilation.
It preserves soft maxima and narrow spatial strokes, including first/last frames;
it does not claim an exact causal-VAE receptive field. Keep the original masks
for final blending. Confirm that extension revision independently before relying
on sparse painted frames; a core feature does not identify extension code.

The upstream decoder truncates to the shorter source/decoded length. For 24 fps
off-grid inputs, `CutleryH3PadEditSource` pads picture, soundtrack and masks to the
next `17*n+5` grid before encode. Decode against its original-length source output
and the unpadded masks to recover original duration. It supplies silence for a
source without audio; omit the synthetic soundtrack on export when no audio edits
were requested. Arbitrary-FPS joint retiming is not supplied by that helper.
Boundary blending changes nearby pixels/samples, and lossy output encoding can
change even preserved regions. Use lossless decoded comparisons for preservation
checks; no byte-exact preservation claim follows from successful generation.

## Verification and limits

CPU regressions cover both sparse/control orders, eligible and dense fallback,
control weight/cleanup delegation, and exclusion of the base attention override
without mutating its options:

```sh
python -m pytest tests-unit/comfy_test/test_minimax_h3_control_sparse.py tests-unit/comfy_test/test_minimax_h3_model.py -q
```

Use the prepared runtime in CPU mode. These fixtures do not load production
weights or prove GPU control adherence, LanPaint quality, cancellation/recovery,
resource estimates, or activation. Run real checkpoint and normal-queue behavioral
acceptance separately, retaining source hashes, masks, settings and prompt IDs.
The independent [community compatibility investigation](https://github.com/wyzborrero/ComfyUI-H3-FunControl)
explains patch-order and representation failure modes; its measured performance
is not a benchmark of this fork or deployment.
