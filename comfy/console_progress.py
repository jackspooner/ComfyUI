"""Opt-in console progress for inference without an existing console indicator."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import sys
import threading
import time
from weakref import WeakSet

from tqdm import tqdm


_current = ContextVar('console_progress', default=None)
_roots = WeakSet()
_rendering = ContextVar('rendering_console_progress', default=False)


@contextmanager
def _console_output():
    token = _rendering.set(True)
    try:
        yield
    finally:
        _rendering.reset(token)


class ConsoleProgress:
    def __init__(self, model, phase='processing', *, total=None, unit=None, delay=0, enabled=True):
        self.model = model
        self.phase = phase
        self.total = total
        self.unit = unit
        self.completed = 0
        self.delay = delay
        self.enabled = enabled
        self._bar = None
        self._stop = threading.Event()
        self._lock = tqdm.get_lock()
        self._thread = None
        self._existing_indicator = False

    def __enter__(self):
        self.parent = _current.get()
        self.root = self.parent.root if self.parent else self
        self._token = _current.set(self)
        if self.parent:
            with self.root._lock:
                self.root._active = self
                self.root.delay = min(self.root.delay, self.delay)
        else:
            with self._lock:
                _roots.add(self)
            self._active = self
            self._started = self._last_draw = time.monotonic()
            self._last_draw -= 0.25
            self._redraw()
            if self.enabled:
                self._thread = threading.Thread(target=self._heartbeat, name='inference-progress', daemon=True)
                self._thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(1):
            with self._lock:
                self._redraw()

    def _redraw(self, final=None):
        now = time.monotonic()
        if not self.enabled or self._existing_indicator:
            return
        # A node-level phase display yields permanently to a model's own bar.
        with tqdm.get_lock(), _console_output():
            others = [bar for bar in list(tqdm._instances) if bar is not self._bar and not bar.disable]
            if others:
                if self._bar is not None:
                    self._bar.clear()
                    self._bar.disable = True
                    for bar in others:
                        bar.refresh()
                self._existing_indicator = True
                return
            if final is None and (now - self._started < self.delay or now - self._last_draw < 0.25):
                return
            if self._bar is None:
                if final and now - self._started < self.delay:
                    return
                self._bar = tqdm(total=None, file=sys.stderr, bar_format='{desc}', mininterval=0.25, leave=True)
                # This row yields to standard bars. Do not reserve a tqdm row,
                # which would make a later sampler allocate a second line.
                tqdm._instances.discard(self._bar)
            active = (self if self.unit is not None else getattr(self, '_summary', self)) if final else self._active
            count = f'{active.completed}/{active.total}' if active.total is not None else str(active.completed)
            work = f' | {count} {active.unit}' if active.unit is not None else ''
            status = final or active.phase
            self._bar.set_description_str(f'{active.model} | {status}{work} | {now - self._started:.1f}s', refresh=False)
            if final is None:
                self._bar.refresh()
            self._last_draw = now

    def update(self, completed=None, *, advance=0, phase=None, total=None):
        with self.root._lock:
            self.completed = self.completed + advance if completed is None else completed
            if (advance or completed is not None) and self.unit is None:
                self.unit = 'iterations'
            if phase is not None:
                self.phase = phase
            if total is not None:
                self.total = total
            self.root._redraw()

    def __exit__(self, exc_type, exc, traceback):
        _current.reset(self._token)
        if self.parent:
            with self.root._lock:
                self.root._summary = self
                self.root._active = self.parent
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        cancelled = exc_type is not None and (issubclass(exc_type, (asyncio.CancelledError, KeyboardInterrupt, InterruptedError)) or exc_type.__name__ == 'InterruptProcessingException')
        status = 'cancelled' if cancelled else 'failed' if exc_type else 'complete'
        with self._lock, _console_output():
            self._redraw(final=status)
            if self._bar is not None:
                self._bar.close()
                self._bar.fp.flush()
            _roots.discard(self)
        return False


def write_log(message, stream):
    """Let tqdm clear/restore our unreserved row around ordinary log records."""
    with tqdm.get_lock(), _console_output():
        bars = [root._bar for root in list(_roots) if root._bar is not None and not root._bar.disable]
        for bar in bars:
            tqdm._instances.add(bar)
        try:
            tqdm.write(message, file=stream)
        finally:
            for bar in bars:
                tqdm._instances.discard(bar)


def observe_progress(data):
    """Yield before an existing indicator writes through the raw-log stream.

    Observing writes also catches bars that start and finish between heartbeats.
    """
    if _rendering.get() or not data.startswith('\r'):
        return
    with tqdm.get_lock(), _console_output():
        for root in list(_roots):
            if not root._existing_indicator:
                if root._bar is not None:
                    root._bar.clear()
                    root._bar.disable = True
                root._existing_indicator = True


def advance(amount=1):
    progress = _current.get()
    if progress is not None:
        progress.update(advance=amount)


def phase(name, *, total=None, unit=None):
    progress = _current.get()
    if progress is not None:
        with progress.root._lock:
            if unit is not None:
                progress.unit = unit
                progress.total = total
                progress.completed = 0
            progress.update(phase=name)


def worker_progress(event):
    """Consume an already correlated IPC progress frame in the parent process."""
    progress = _current.get()
    if progress is not None:
        progress.update(phase=event['phase'], completed=event.get('completed'))


def report_progress(model, phase='processing', **options):
    """Scope a sync/async inference entrypoint; nested calls share one row."""
    def decorate(function):
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                with ConsoleProgress(model, phase, **options):
                    return await function(*args, **kwargs)
            return asynchronous

        @wraps(function)
        def synchronous(*args, **kwargs):
            with ConsoleProgress(model, phase, **options):
                return function(*args, **kwargs)
        return synchronous
    return decorate
