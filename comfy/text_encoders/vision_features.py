"""Bounded image-only features for one explicitly scoped conditioning preparation."""
import torch


def clone_features(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [clone_features(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_features(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_features(item) for key, item in value.items()}
    return value


def feature_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(feature_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(feature_bytes(item) for item in value.values())
    return 0


class VisionFeatures:
    def __init__(self, max_entries, max_bytes):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Vision feature limits must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = []
        self.bytes = 0
        self.identity = None
        self.hits = 0
        self.encodes = 0
        self.closed = False

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def close(self):
        self.clear()
        self.closed = True

    def accepts(self, clip):
        if self.closed:
            return False
        patcher = clip.patcher
        if (clip.use_clip_schedule or patcher.forced_hooks is not None or patcher.current_hooks is not None
                or patcher.patches or set(patcher.object_patches) - {"manual_cast_dtype"} or patcher.weight_wrapper_patches
                or patcher.injections or patcher.model_options != {"transformer_options": {}}
                or any(any(groups.values()) for groups in patcher.callbacks.values())
                or any(any(groups.values()) for groups in patcher.wrappers.values())):
            self.clear()
            return False
        identity = (id(patcher.model), patcher.patches_uuid, str(patcher.load_device),
                    patcher.object_patches.get("manual_cast_dtype"))
        if self.identity != identity:
            self.clear()
            self.identity = identity
        return True

    def encode(self, visual, image, grid, device, dtype):
        device = torch.device(device)
        # Other accelerators and training paths keep their existing execution.
        if (self.closed or torch.is_grad_enabled() or visual.training or device.type not in ("cpu", "cuda")
                or any(module._forward_hooks or module._forward_pre_hooks for module in visual.modules())):
            self.encodes += 1
            return visual(image.to(device, dtype=torch.float32), grid)
        grid_identity = (grid.dtype, tuple(grid.shape), tuple(grid.flatten().tolist())) if isinstance(grid, torch.Tensor) else tuple(grid)
        identity = (id(visual), grid_identity, str(device), dtype, image.dtype, str(image.device), tuple(image.shape))
        for key, pixels, features in self.entries:
            if key == identity and torch.equal(pixels.contiguous().view(torch.uint8), image.contiguous().view(torch.uint8)):
                self.hits += 1
                return clone_features(features)
        cpu_rng = torch.get_rng_state()
        device_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        self.encodes += 1
        result = visual(image.to(device, dtype=torch.float32), grid)
        deterministic = torch.equal(cpu_rng, torch.get_rng_state())
        if device_rng is not None:
            deterministic = deterministic and torch.equal(device_rng, torch.cuda.get_rng_state(device))
        size = feature_bytes(result) + feature_bytes(image)
        if deterministic and not self.closed and len(self.entries) < self.max_entries and self.bytes + size <= self.max_bytes:
            self.entries.append((identity, image.clone(), clone_features(result)))
            self.bytes += size
        return result
