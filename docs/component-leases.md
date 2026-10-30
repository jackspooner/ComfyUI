# Native component leases

This fork adds opt-in, bounded CLIP owner reuse for sequential native execution.
It avoids repeated checkpoint loading when an explicit caller owns the lifetime.
Ordinary loaders retain their existing behavior outside an activated pool.

```python
from comfy.component_leases import ComponentLeasePool
from comfy.sd import CLIPType, load_clip

pool = ComponentLeasePool(max_entries=2, max_feature_entries=16,
                          max_feature_bytes=256 * 1024 * 1024)
try:
    with pool.activate():
        krea = load_clip([checkpoint], clip_type=CLIPType.KREA2)
        fibo = load_clip([checkpoint], clip_type=CLIPType.FIBO2)
    # Run sequentially through the ordinary CLIP and ModelPatcher APIs.
    # Existing `with clip.vision_encode_scope():` uses the bounded pool cache.
finally:
    pool.close()
print(pool.snapshot())
```

`checkpoint` is an already provisioned local full Qwen3-VL 4B package. There are
no downloads. The caller must activate the pool **in the actual loader execution
context**. A ContextVar set in a sibling async task does not propagate to another
task. Prompt registration, graph compilation, ordering, cancellation and pool
lifetime belong to the caller, not this inference module. `activate()` restores
the previous context on exceptions; the caller must still `close()` in its own
final cleanup. A pool may span sequential groups within that bounded lifetime.

## Compatibility and ownership

Exact same-wrapper loads of native CLIP families reuse the existing native clone
contract. Cross-wrapper reuse is explicitly admitted only for full Qwen3-VL:

| Backbone | Audited wrappers |
| --- | --- |
| Qwen3-VL 4B | generic `STABLE_DIFFUSION`, KREA2, FIBO2, FLUX3, MAGE |
| Qwen3-VL 8B | generic `STABLE_DIFFUSION`, IDEOGRAM4, BOOGU, QWEN_IMAGE |

The identity includes resolved local paths, device/inode, size, modification and
change timestamps, embedding-directory options, runtime/model options, effective
load/offload devices, precision and dynamic-offload selection. Distinct files are
not aliased by filename or guessed model family. Opaque/custom-operation options
bypass reuse. A checkpoint changing while loading is not retained. Runtime or
artifact identity differences create separate entries. File identity is local
metadata, not a cross-file content-hash deduplication service.

Cross-wrapper adapters require completely loaded parameters, the full configured
layer count and final norm, and the separate language head or the 4B tied head.
The constructor verifies architecture/configuration, dtype and operation identity
again before sharing `.model` and `.visual`. The wrapper, tokenizer, vision-cache
pointer, layer taps, padding/position behavior, normalization and output layout
remain separate. FLUX/FLUX2 Klein, partial exports, 32B H3 conditioning encoders,
JoyImage and other unaudited cross-wrapper combinations split explicitly.
Native Qwen3-VL Generate Text rejects incomplete or truncated weights, including
outside a pool. It does not promote conditioning-only packages into generators.

Adapters clone the existing ModelPatcher and keep its original model owner and
parameter namespaces. Native memory management still loads, offloads and switches
weight patches. The pool does not pin weights, reserve GPU memory or suppress
eviction. Native LoRA/weight patches stay on consumer clones; patched vision
features are not reused. Object replacement, injections and custom hooks on a
shared adapter fail with an actionable error: load outside the lease for those
operations. `CLIP.load_sd` cannot replace leased weights in place.

Call options are restored after encode/generate, including failure and
cancellation. Model histories, KV caches, RNG generators and structured-output
constraints remain per call; no generated text or conditioning output is cached.
Calls are sequential, not a concurrent inference API.

## Bounds and counters

The pool holds at most `max_entries` native owners (1–16, default 2), evicting its
least recently used reference. Existing graph outputs and ComfyUI's own cache
can still reference an evicted owner; eviction and `close()` are not claims of
immediate GPU unloading. They never globally unload unrelated models.

Explicit `vision_encode_scope()` calls on compatible leased CLIPs can share
processed image features across nodes, including positive/negative conditioning.
The cache is bounded by aggregate entry count and tensor bytes, including stored
processed pixels; exact pixel bytes, geometry, dtype, visual identity and native
patcher identity must match. Cache outputs are cloned before consumption.
Different model owners, patch generations or unsupported patches invalidate it.
Weights remain normally offloadable; cached features can retain up to the stated
byte bound on their producing device until eviction/close. Narrower per-call
feature limits create a local scope instead. Closing clears and disables the
pool cache, including for previously returned CLIPs.

`snapshot()` returns `constructions`, `exact_reuses`, `backbone_reuses`,
`evictions`, `splits` (reason-to-count), `retained`, `max_entries`, `closed`, and
`vision_features: {hits, encodes, retained, bytes}`. Construction counts successful
native loads, reuse counts returned native leases, and vision counters count
actual feature-cache work. They are not generated-result reuse or GPU performance
measurements. Counters remain available after close; retained counts/bytes clear.

## Verification and activation

Use the repository's normal `scripts/setup_worktree.py` readiness command with
the explicitly prepared interpreter, then run:

```bash
python -m pytest tests-unit/comfy_test/component_leases_test.py tests-unit/comfy_test/vision_features_test.py tests-unit/comfy_test/fibo2_test.py tests-unit/comfy_test/test_krea2_efficiency.py tests-unit/comfy_test/image_history_test.py tests-unit/comfy_test/text_history_test.py tests-unit/comfy_test/text_generation_length_test.py tests-unit/comfy_test/text_structured_output_test.py tests-unit/comfy_test/text_json_native_test.py -q
```

CPU fake-model checks verify one construction across compatible adapters,
incompatible/partial splits, private options, cancellation, feature reuse and
bounded cleanup. Real-weight output parity, cold/warm latency, VRAM/RAM behavior
and serving activation remain separate acceptance gates. No model inference or
service restart is performed by setup/tests. Deployment requires both this core
and a matching opt-in lifecycle owner; source delivery alone does not enable it.
