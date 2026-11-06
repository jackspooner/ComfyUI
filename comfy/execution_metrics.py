"""Optional process-local scalar observations at native inference boundaries.

The application owns collection. Observers cannot change an inference result;
this module performs no persistence, device queries or synchronization.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

_observer = None
_stage = ContextVar('execution_metrics_stage', default=None)
_evaluation = ContextVar('execution_metrics_evaluation', default=None)
_vision = ContextVar('execution_metrics_vision', default=None)


def set_observer(observer):
    global _observer
    _observer = observer


def enabled():
    return _observer is not None


def current_stage():
    return _stage.get()


def observe_shape(event, tensor, **facts):
    if enabled():
        try:
            emit(event, shape=tuple(tensor.shape), **facts)
        except Exception:
            emit('tensor_shape_unavailable', operation=event)


def observe_shapes(event, tensors, streams=None, **facts):
    """Observe a bounded set of named native tensor shapes without retaining tensors."""
    if enabled():
        try:
            if len(tensors) > 8:
                raise ValueError('Too many tensor shapes')
            shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
            if any(len(shape) > 8 for shape in shapes.values()):
                raise ValueError('Too many tensor axes')
            if streams is not None:
                facts['input_streams'] = len(streams)
            emit(event, shapes=shapes, precisions={name: str(tensor.dtype) for name, tensor in tensors.items()},
                 devices={name: str(tensor.device) for name, tensor in tensors.items()}, **facts)
        except Exception:
            emit('tensor_shape_unavailable', operation=event)


@contextmanager
def evaluation_scope():
    """Temporary identity for one native noise evaluation, including CFG batches."""
    if not enabled():
        yield
        return
    token = _evaluation.set(object())
    try:
        yield
    finally:
        _evaluation.reset(token)


def observe_control(event, binding, tensor=None, **facts):
    if enabled():
        facts.update(binding=binding, evaluation=_evaluation.get())
        if tensor is None:
            emit(event, **facts)
        else:
            observe_shape(event, tensor, **facts)


@contextmanager
def vision_scope():
    """Correlate one preprocessing/cache/forward call without retaining a model."""
    if not enabled():
        yield
        return
    token = _vision.set(object())
    try:
        yield
    finally:
        _vision.reset(token)


def observe_vision(event, tensors=None, **facts):
    binding = _vision.get()
    if enabled() and binding is not None:
        try:
            if tensors is None:
                emit(event, binding=binding, **facts)
            else:
                observe_shapes(event, tensors, binding=binding, **facts)
        except Exception:
            emit('h3_vision_metadata_unavailable', binding=binding)


@contextmanager
def measure_vision(event):
    binding = _vision.get()
    if binding is None:
        yield
    else:
        with measure(event, binding=binding):
            yield


def observe_embedding_row(positions, active, embeddings):
    """Read existing CPU expansion metadata, never token IDs or tensor values."""
    if not enabled():
        return
    try:
        counts = {'image': 0, 'audio': 0, 'unknown': 0}
        for entry in embeddings:
            size = entry['size']
            if type(size) is not int or size < 0:
                raise ValueError('Unknown embedding expansion')
            role = entry['type']
            counts[role if role in ('image', 'audio') else 'unknown'] += size
        text = active - sum(counts.values())
        if type(positions) is not int or type(active) is not int or not 0 <= text <= active <= positions:
            raise ValueError('Unknown embedding positions')
        emit('text_embedding_row', positions=positions, text=text, image=counts['image'],
             audio=counts['audio'], unknown=counts['unknown'], padding=positions - active)
    except Exception:
        emit('text_embedding_unavailable')


@contextmanager
def measure(event, **facts):
    """Observe a CPU monotonic boundary without synchronizing devices."""
    if not enabled():
        yield
        return
    started = monotonic()
    try:
        yield
    except BaseException:
        emit(event, seconds=max(0, monotonic() - started), succeeded=False, **facts)
        raise
    else:
        emit(event, seconds=max(0, monotonic() - started), succeeded=True, **facts)


@contextmanager
def stage_scope(stage_id, class_type, inputs=None, reused=False):
    """Observe an owner's nested call without replacing its execution context."""
    token = _stage.set(stage_id)
    try:
        emit('stage_reused' if reused else 'stage_started', class_type=class_type)
        if not reused and inputs is not None:
            emit('stage_inputs', class_type=class_type, inputs=inputs)
        try:
            yield
        except BaseException:
            emit('stage_failed', class_type=class_type)
            raise
        else:
            if not reused:
                emit('stage_finished', class_type=class_type)
    finally:
        _stage.reset(token)


def emit(event, **facts):
    if _observer is not None:
        try:
            _observer(event, facts)
        except Exception:
            # Observability must not become an inference failure or retry trigger.
            pass
