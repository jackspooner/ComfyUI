"""Opt-in, bounded references to native CLIP owners for sequential execution."""
from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import os
import threading

import torch
from comfy.text_encoders.vision_features import VisionFeatures


active_pool = ContextVar("native_component_lease_pool", default=None)


def _option_key(value):
    if value is None or isinstance(value, (str, int, float, bool, torch.dtype, torch.device)):
        return (type(value).__name__, str(value))
    if isinstance(value, (tuple, list)):
        return tuple(_option_key(item) for item in value)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return tuple((key, _option_key(item)) for key, item in sorted(value.items()))
    raise ValueError("opaque_runtime_options")


def _file_key(path):
    path = os.path.realpath(path)
    stat = os.stat(path)
    return (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class ComponentLeasePool:
    """The caller owns lifetime; model_management still owns load/offload decisions.

    This retains at most max_entries native owners, not weight residency. Calls
    must be sequential. Only the explicit bounded image-feature cache retains
    tensors; no sessions, generation KV caches, text outputs or RNGs are retained.
    """
    def __init__(self, max_entries=2, max_feature_entries=16, max_feature_bytes=256 * 1024 * 1024):
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or not 1 <= max_entries <= 16:
            raise ValueError("Component lease max_entries must be an integer from 1 to 16")
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.counters = Counter(constructions=0, exact_reuses=0, backbone_reuses=0, evictions=0)
        self.splits = Counter()
        self.lock = threading.RLock()
        self.closed = False
        self.vision_features = VisionFeatures(max_feature_entries, max_feature_bytes)

    @contextmanager
    def activate(self):
        """Route ordinary load_clip calls in this context through this owner."""
        with self.lock:
            if self.closed:
                raise RuntimeError("Component lease pool is closed")
        token = active_pool.set(self)
        try:
            yield self
        finally:
            active_pool.reset(token)

    def close(self):
        with self.lock:
            self.entries.clear()
            self.vision_features.close()
            self.closed = True

    def snapshot(self):
        with self.lock:
            return {**self.counters, "splits": dict(self.splits), "retained": len(self.entries),
                    "max_entries": self.max_entries, "closed": self.closed,
                    "vision_features": {"hits": self.vision_features.hits, "encodes": self.vision_features.encodes,
                                        "retained": len(self.vision_features.entries), "bytes": self.vision_features.bytes}}

    def _lease(self, clip):
        clip._component_vision_features = self.vision_features
        clip._component_lease = True
        return clip

    def load_clip(self, loader, adapter, ckpt_paths, embedding_directory, clip_type, model_options, disable_dynamic, runtime_identity=()):
        """Internal loader hook. Identity is the same local artifact and runtime."""
        with self.lock:
            if self.closed:
                raise RuntimeError("Component lease pool is closed")
            try:
                key = (tuple(_file_key(path) for path in ckpt_paths), _option_key(embedding_directory),
                       _option_key(model_options), disable_dynamic, _option_key(runtime_identity))
            except ValueError as error:
                self.splits[str(error)] += 1
                result = loader(ckpt_paths, embedding_directory, clip_type, model_options, disable_dynamic)
                self.counters["constructions"] += 1
                return result
            for entry_key, (source, source_type) in reversed(self.entries.items()):
                if entry_key[0] != key:
                    continue
                result, reason = adapter(source, source_type, clip_type, embedding_directory, model_options)
                if result is not None:
                    self.entries.move_to_end(entry_key)
                    self.counters["exact_reuses" if source_type == clip_type else "backbone_reuses"] += 1
                    return self._lease(result)
                self.splits[reason] += 1
            if self.entries and not any(entry_key[0] == key for entry_key in self.entries):
                self.splits["artifact_or_runtime_identity_changed"] += 1
            source = loader(ckpt_paths, embedding_directory, clip_type, model_options, disable_dynamic)
            self.counters["constructions"] += 1
            if tuple(_file_key(path) for path in ckpt_paths) != key[0]:
                self.splits["artifact_changed_during_load"] += 1
                return source
            entry_key = (key, clip_type)
            self.entries[entry_key] = (source, clip_type)
            self.entries.move_to_end(entry_key)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
                self.vision_features.clear()
                self.counters["evictions"] += 1
            # Keep the retained patcher private: consumer patches belong to clones.
            result, _ = adapter(source, clip_type, clip_type, embedding_directory, model_options)
            return self._lease(source.clone() if result is None else result)
