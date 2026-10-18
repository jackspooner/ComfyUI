# Krea 2 editing efficiency

Use the updated `comfyui-krea2edit` source-patch and grounded-encode nodes with
native Krea 2, its compatible Qwen3-VL encoder, VAE and identity-edit LoRA.
Keep the same prompt, references, steps and guidance. Reference boost 1 requires
no attention bias; boosted attention is unchanged by this optimization.

The model exposes `preprocess_text_embeds(context, transformer_options={})` for
the edit adapter to run text fusion and projection after model loading. The
adapter puts the result into normal per-run conditioning through its patched
`extra_conds`; it does not retain a model-owned tensor cache. Each scheduled
conditioning is prepared separately. Weight hooks, attention patches/overrides
and dynamic sampling extensions keep the original per-step computation.

Native attention receives compact K/V heads and `enable_gqa`; the existing backend
dispatcher owns fallback expansion. No backend is forced. The output head sees
only target tokens. Reference tokens still participate in every transformer
block; caching their K/V across steps would change this editor's computation.

Preparation: `COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py`.
Run `"$COMFYUI_PYTHON" -m pytest -q tests-unit/comfy_test/test_krea2_efficiency.py`
and the node repository's tests with this checkout on `PYTHONPATH`.
CPU numerical checks cover GQA, output slicing and the node's conditioning path.
They do not establish full-checkpoint speed, memory use or cross-device quality.
Compare fixed-seed warm runs at the same dimensions and report encoding and
sampling times separately. Floating-point results may differ between attention
kernels. An already-running ComfyUI process needs a separately scheduled reload
to use changed Python code; source integration alone does not activate it.
