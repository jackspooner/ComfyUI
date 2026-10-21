"""Optional disk cache; lifecycle is owned by ComfyUI's prompt worker."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .store import CacheError

manager = None
_node = ContextVar("model_cache_node", default=None)
_prompt = ContextVar("model_cache_prompt", default=None)
_progress = ContextVar("model_cache_progress", default=None)


def use(keys):
    node = _node.get()
    if node is not None:
        node.update(keys)
    prompt = _prompt.get()
    if prompt is not None:
        new = set(keys) - prompt["used"]
        if new and manager is not None:
            manager.touch(new)
        prompt["used"].update(keys)


def resources():
    return frozenset(_node.get() or ())


@contextmanager
def node_scope(node_id, progress=None):
    prompt = _prompt.get()
    values = prompt["nodes"].setdefault(node_id, set()) if prompt is not None else set()
    token = _node.set(values)
    progress_token = _progress.set(progress)
    try:
        yield values
    finally:
        _node.reset(token)
        _progress.reset(progress_token)


@contextmanager
def prompt_scope():
    state = {"used": set(), "nodes": {}}
    token = _prompt.set(state)
    try:
        yield
    finally:
        try:
            if manager is not None:
                manager.touch(state["used"], finished=True)
        finally:
            _prompt.reset(token)


def stage(path, progress=None):
    if manager is None:
        return path
    if Path(path).absolute().is_relative_to(manager.archive) and _prompt.get() is None:
        raise CacheError("EXECUTION_REQUIRED", "Load archive weights through the ComfyUI execution queue.")
    result, key = manager.stage(path, progress or _progress.get())
    if key is not None:
        use({key})
    return result


def resolve(path):
    # Discovery/metadata reads must not stage weights or refresh inactivity.
    return stage(path) if _node.get() is not None else str(path)
