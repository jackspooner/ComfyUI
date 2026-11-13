"""CPU-only console/worker-facing progress contracts, with no model or server."""
import asyncio
from collections import deque
import io
import logging
from unittest.mock import patch

import pytest
from tqdm import tqdm

from app import logger
from comfy.console_progress import ConsoleProgress, advance, phase, report_progress, worker_progress, _current
from comfy_execution.console_progress import instrument_custom_node


@pytest.fixture
def clock():
    with patch('comfy.console_progress.time.monotonic') as now:
        now.return_value = 0
        yield now


def test_counts_known_and_unknown_totals_and_throttling(clock):
    output = io.StringIO()
    with patch('sys.stderr', output), ConsoleProgress('speech', unit='tokens') as progress:
        for index in range(100):
            clock.return_value = index / 1000
            advance()
        assert progress.completed == 100
        assert output.getvalue().count('speech') <= 2
        clock.return_value = 0.25
        advance()
        assert '101 tokens' in output.getvalue()
    assert output.getvalue().endswith('\n')
    assert output.getvalue().count('\n') == 1
    assert 'complete | 101 tokens' in output.getvalue()
    assert '%' not in output.getvalue() and 'ETA' not in output.getvalue()
    with patch('sys.stderr', output), ConsoleProgress('images', total=3, unit='images') as progress:
        progress.update(3)
    assert '3/3 images' in output.getvalue()


def test_nested_phases_share_bar_and_restore_parent(clock):
    output = io.StringIO()
    with patch('sys.stderr', output), ConsoleProgress('ASR', 'preparing') as outer:
        with ConsoleProgress('ASR', 'decode', unit='iterations') as inner:
            advance(3)
            clock.return_value = 0.3
            phase('alignment')
            assert inner.root is outer
            assert outer._bar is not None
            assert inner._bar is None
        assert _current.get() is outer
        assert outer._active is outer
    assert _current.get() is None
    assert 'complete | 3 iterations' in output.getvalue()
    assert output.getvalue().count('\n') == 1


@pytest.mark.parametrize('exception,status', [(RuntimeError, 'failed'), (asyncio.CancelledError, 'cancelled'), (KeyboardInterrupt, 'cancelled'), (InterruptedError, 'cancelled')])
def test_cleanup_on_error_or_cancellation(exception, status):
    output = io.StringIO()
    before = set(tqdm._instances)
    with patch('sys.stderr', output), pytest.raises(exception):
        with ConsoleProgress('model') as progress:
            raise exception()
    assert status in output.getvalue()
    assert _current.get() is None
    assert progress._stop.is_set()
    assert not progress._thread.is_alive()
    assert set(tqdm._instances) == before
    assert output.getvalue().count('\n') == 1


def test_opaque_phase_heartbeat_and_fast_node_silence(clock):
    output = io.StringIO()
    with patch('sys.stderr', output), ConsoleProgress('worker', delay=1) as progress:
        assert output.getvalue() == ''
        clock.return_value = 1
        progress._redraw()
        assert 'worker | processing | 1.0s' in output.getvalue()
        assert 'iterations' not in output.getvalue()
        clock.return_value = 2
        progress._redraw()
        assert '2.0s' in output.getvalue()
    output = io.StringIO()
    with patch('sys.stderr', output), ConsoleProgress('fast', delay=1):
        pass
    assert output.getvalue() == ''


def test_fallback_yields_to_existing_indicator(clock):
    output = io.StringIO()
    with patch('sys.stderr', output), ConsoleProgress('fallback', delay=1) as progress:
        with tqdm(total=2, file=output, desc='sampler'):
            clock.return_value = 1
            progress._redraw()
    assert 'fallback' not in output.getvalue()
    assert 'sampler' in output.getvalue()


def test_later_sampler_uses_same_row_and_owns_final_summary(clock):
    output = io.StringIO()
    with patch('sys.stderr', output), ConsoleProgress('preparing') as progress:
        assert progress._bar is not None
        with tqdm(total=2, file=output, desc='sampler') as sampler:
            assert sampler.pos == 0
            clock.return_value = 1
            progress._redraw()
            sampler.update(2)
    assert output.getvalue().count('\n') == 1
    assert 'preparing | complete' not in output.getvalue()


def test_raw_log_observer_detects_short_existing_bar_between_heartbeats():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding='utf-8')
    with patch.object(logger, 'logs', deque(maxlen=300)):
        interceptor = logger.LogInterceptor(stream)
        with patch('sys.stderr', interceptor), ConsoleProgress('custom fallback'):
            with tqdm(total=1, file=interceptor, desc='existing sampler') as sampler:
                assert sampler.pos == 0
                sampler.update(1)
        data = raw.getvalue().decode()
        assert 'custom fallback | complete' not in data
        assert data.count('\n') == 1
        assert 'existing sampler' in data
        interceptor.detach()
    stream.detach()


def test_logs_are_readable_between_progress_redraws(clock):
    output = io.StringIO()
    handler = logger.ProgressStreamHandler(output)
    with patch('sys.stderr', output), ConsoleProgress('model', unit='tokens'):
        advance(2)
        handler.emit(logging.LogRecord('test', logging.WARNING, '', 1, 'ordinary warning', (), None))
        clock.return_value = 0.3
        advance()
    assert 'ordinary warning\n' in output.getvalue()
    assert 'complete | 3 tokens' in output.getvalue()


def test_disabled_and_worker_forwarding(clock):
    output = io.StringIO()
    with patch('sys.stderr', output), ConsoleProgress('worker', enabled=False) as progress:
        worker_progress({'phase': 'encoding', 'completed': 4})
        assert progress.completed == 4 and progress.phase == 'encoding'
        assert progress._thread is None
    assert output.getvalue() == ''


def test_custom_node_descriptors_signatures_and_async_cleanup():
    class Node:
        FUNCTION = 'run'
        @classmethod
        def run(cls, value=2):
            return cls, value
    instrument_custom_node(Node, 'custom')
    method = Node.run.__func__
    instrument_custom_node(Node, 'custom again')
    assert method is Node.run.__func__
    assert Node.run(7) == (Node, 7)
    class Static:
        FUNCTION = 'execute'
        @staticmethod
        async def execute(value):
            assert _current.get() is not None
            await asyncio.sleep(0)
            return value
    instrument_custom_node(Static, 'async')
    assert asyncio.run(Static.execute(3)) == 3
    assert _current.get() is None
    assert isinstance(Static.__dict__['execute'], staticmethod)


def test_registration_preserves_v1_and_v3_execution(tmp_path, monkeypatch):
    import nodes
    monkeypatch.setattr(nodes, 'NODE_CLASS_MAPPINGS', {})
    monkeypatch.setattr(nodes, 'NODE_DISPLAY_NAME_MAPPINGS', {})
    v1 = tmp_path / 'custom_v1.py'
    v1.write_text("class Example:\n FUNCTION='run'\n def run(self, value): return (value,)\nNODE_CLASS_MAPPINGS={'Example':Example}\n")
    assert asyncio.run(nodes.load_custom_node(str(v1)))
    assert nodes.NODE_CLASS_MAPPINGS['Example']().run(2) == (2,)
    v3 = tmp_path / 'custom_v3.py'
    v3.write_text("from comfy_api.latest import io, ComfyExtension\nclass Example(io.ComfyNode):\n @classmethod\n def define_schema(cls): return io.Schema(node_id='V3Example', inputs=[], outputs=[])\n @classmethod\n def execute(cls): return io.NodeOutput()\nclass Extension(ComfyExtension):\n async def get_node_list(self): return [Example]\nasync def comfy_entrypoint(): return Extension()\n")
    assert asyncio.run(nodes.load_custom_node(str(v3)))
    cls = nodes.NODE_CLASS_MAPPINGS['V3Example']
    assert isinstance(cls.EXECUTE_NORMALIZED(), type(cls.execute()))
    assert _current.get() is None
    # Registration historically permits incomplete classes; validation owns them.
    instrument_custom_node(type('Incomplete', (), {}), 'incomplete')


@pytest.mark.parametrize('interleave', [False, True])
def test_raw_log_snapshot_and_stream_reconnect(clock, interleave):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding='utf-8')
    entries = []
    with patch.object(logger, 'logs', deque(maxlen=300)):
        interceptor = logger.LogInterceptor(stream)
        interceptor.on_flush(lambda chunks: entries.extend(chunks))
        # The first output may begin with CR in an empty log buffer.
        interceptor.write('\rfirst')
        interceptor.flush()
        with patch('sys.stderr', interceptor), ConsoleProgress('ASR', unit='iterations'):
            for index in range(1, 5):
                clock.return_value = index * .3
                advance()
                if interleave and index == 2:
                    logger.ProgressStreamHandler(interceptor).emit(logging.LogRecord('fixture', logging.WARNING, '', 1, 'ordinary warning', (), None))
        snapshot = ''.join(entry['m'] for entry in logger.get_logs())
        live = ''.join(entry['m'] for entry in entries)
        def rows(value):
            # Same CR/newline semantics used by the panel's xterm raw transport.
            result, row = [], ''
            for character in value:
                if character == '\r':
                    row = ''
                elif character == '\n':
                    result.append(row.rstrip())
                    row = ''
                else:
                    row += character
            return result, row
        assert rows(live) == rows(snapshot)
        assert len(rows(snapshot)[0]) == (2 if interleave else 1)
        assert 'complete | 4 iterations' in snapshot
        interceptor.detach()
    stream.detach()
