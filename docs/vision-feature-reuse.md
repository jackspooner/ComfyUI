# Scoped Qwen3-VL image features

This fork adds an opt-in `CLIP.vision_encode_scope(max_entries=16,
max_bytes=256 * 1024 * 1024)` context manager for a single conditioning
preparation. Qwen-Image 2.1, Krea 2, Mage and other encoders using
`Qwen3VLClipModel` can reuse identical processed image features while running
each positive and negative text pass independently. Existing calls outside the
scope keep their original behavior. Deployment of the serving checkout and
model acceptance are separate operational steps.

```python
with clip.vision_encode_scope():
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt, images=images))
```

Enter the scope in the node that owns both encodes. A scope does not span graph
nodes, prompts, workers or durable checkpoints, and `clip.clone()` does not
inherit an active scope. Older installations lack this method; optional callers
can use their ordinary encode path when the method is unavailable. Do not hold
the scope across sampling or separate requests.

Reuse requires exact processed image bytes, geometry, vision model, target
device and model/input dtype. Merged tokens and every DeepStack tensor have
private copies for each consumer; image positions and text conditioning are
assembled afresh. The byte cap accounts for retained image snapshots and feature
tensors. Reaching either cap executes later images normally. Success, exceptions
and cancellation all clear retained features when the scope exits; no model is
pinned or loaded by the scope.

The initial implementation supports inference on CPU and CUDA. Gradient-enabled
execution, training-mode vision modules, module forward hooks, CLIP schedules,
hooks, weight/custom object patches, injected code, callbacks, wrappers and custom model
options use ordinary encoding. Changes to the CLIP/patcher model identity,
patch generation, load device or native manual-cast dtype invalidate the bounded store. A forward that
advances CPU or target CUDA RNG state is never stored, so later calls retain its
normal random-number consumption. Unsupported backends and other encoder
families keep their existing behavior.

The owner must not mutate model weights or precision outside native patcher
facilities during a scope. This API does not share complete text conditioning,
VAE latents or sampling results and does not make separate Krea graph nodes
share a scope automatically.

After [worktree setup](../AGENTS.md#worktree-development), run:

```sh
python -m pytest tests-unit/comfy_test/vision_features_test.py -q
```

CPU fixtures execute native image preprocessing and CLIP option propagation with
recording vision/text stubs. They check numerical parity, call counts, private
DeepStack/grid metadata, changed content/geometry/precision, state invalidation,
RNG parity, limits and exception cleanup. They do not establish real-model GPU
parity, peak-memory savings or a throughput percentage.
