"""Optional process-local scalar observations at native inference boundaries.

The application owns collection. Observers cannot change an inference result;
this module performs no persistence, device queries or synchronization.
"""
_observer = None


def set_observer(observer):
    global _observer
    _observer = observer


def enabled():
    return _observer is not None


def emit(event, **facts):
    if _observer is not None:
        try:
            _observer(event, facts)
        except Exception:
            # Observability must not become an inference failure or retry trigger.
            pass
