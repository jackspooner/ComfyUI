from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from comfy.cli_args import args
args.cpu = True
import comfy.model_prefetch as mp
import comfy.ops as ops


def test_failed_prefetch_keeps_all_faulted_modules_reachable(monkeypatch):
    first = SimpleNamespace(_v='first', _v_signature='resident')
    second = SimpleNamespace(_v='second', _v_signature='old')
    root = SimpleNamespace(modules=lambda: iter([first, second]))
    queue = [None, root]
    monkeypatch.setattr(mp, 'PREFETCH_QUEUES', [queue])
    monkeypatch.setattr(mp, 'MALLOC_GRAPHS', {})
    monkeypatch.setattr(mp, 'cleanup_malloc_graph', lambda: None)
    vbar = ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(vbar, 'vbar_fault', lambda v: 'resident' if v == 'first' else 'new')
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: a == b)
    unpin = Mock()
    monkeypatch.setattr(vbar, 'vbar_unpin', unpin)
    def fail(*a):
        raise RuntimeError('prefetch preparation failed')
    monkeypatch.setattr(ops, 'materialize_meta_param', fail)
    monkeypatch.setattr(mp, 'pin_modules', lambda modules, device, dtype: ops.cast_modules_with_vbar(modules, None, device, None, False, True))
    with pytest.raises(RuntimeError, match='prefetch preparation failed'):
        mp.prefetch_queue_pop(queue, 'cpu', root)
    mp.cleanup_prefetch_queues()
    assert [c.args[0] for c in unpin.call_args_list] == ['first', 'second']
    assert not hasattr(first, '_prefetch')
    assert not hasattr(second, '_prefetch')
    mp.cleanup_prefetch_queues()
    assert unpin.call_count == 2


def test_successful_resident_prefetch_is_cleaned_once(monkeypatch):
    child = SimpleNamespace(_v='child', _v_signature='same')
    root = SimpleNamespace(modules=lambda: iter([child]))
    queue = [None, root, None]
    monkeypatch.setattr(mp, 'PREFETCH_QUEUES', [queue])
    monkeypatch.setattr(mp, 'MALLOC_GRAPHS', {})
    monkeypatch.setattr(mp, 'cleanup_malloc_graph', lambda: None)
    vbar = ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(vbar, 'vbar_fault', lambda v: 'same')
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: a == b)
    unpin = Mock()
    monkeypatch.setattr(vbar, 'vbar_unpin', unpin)
    monkeypatch.setattr(mp, 'pin_modules', lambda modules, device, dtype: ops.cast_modules_with_vbar(modules, None, device, None, False, True))
    core = Mock()
    mp.prefetch_queue_pop(queue, 'cpu', root, core=core)
    core.assert_called_once()
    assert child._prefetch['resident']
    unpin.assert_not_called()
    mp.prefetch_queue_pop(queue, 'cpu', None)
    unpin.assert_called_once_with('child')
    mp.cleanup_prefetch_queues()
    unpin.assert_called_once()


def test_direct_cast_failure_does_not_publish_incomplete_prefetch(monkeypatch):
    child = SimpleNamespace(_v='child', _v_signature='old')
    vbar = ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(vbar, 'vbar_fault', lambda v: 'new')
    monkeypatch.setattr(vbar, 'vbar_signature_compare', lambda a, b: False)
    def fail(*a):
        raise RuntimeError('preparation failed')
    monkeypatch.setattr(ops, 'materialize_meta_param', fail)
    with pytest.raises(RuntimeError, match='preparation failed'):
        ops.cast_modules_with_vbar([child], None, 'cpu', None, False)
    assert not hasattr(child, '_prefetch')
