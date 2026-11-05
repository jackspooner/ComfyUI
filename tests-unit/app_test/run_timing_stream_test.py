import asyncio
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import jsonschema

from app.run_timing import RunTimingStore
from app.run_timing_http import add_routes
from app.run_timing_stream import TimingStreamHub, MAX_EVENT_BYTES, MAX_PENDING, MAX_SUBSCRIBERS


def decode(frame):
    return json.loads(frame.removeprefix('data: ').strip())


def item():
    return (1, 'prompt', {'node': {'class_type': 'PrimitiveInt', 'inputs': {'value': 1}}}, {}, ['node'])


def validate(value):
    schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(value, schema['$defs']['TimingObservationStreamEvent'])


def test_live_baseline_committed_snapshots_and_monotonic_elapsed_are_not_catchup(tmp_path):
    async def scenario():
        clock = [0.0]
        store = RunTimingStore(str(tmp_path / 'timing.sqlite'), host_id='fixture', now=lambda: clock[0])
        store.event('accepted', item())
        sub = store.streams.subscribe(asyncio.get_running_loop(), 1)
        ready = decode(await sub.next())
        validate(ready)
        assert ready['type'] == 'ready' and ready['cursor'] == '1' and not sub.pending
        clock[0] = 5
        await asyncio.to_thread(store.event, 'started', item())
        started = decode(await sub.next())
        validate(started)
        assert started['cursor'] == '2' and started['observation']['phase'] == 'running'
        clock[0] = 8
        await asyncio.to_thread(store.safe_stage, 'prompt', 'node', 'started')
        stage = decode(await sub.next())
        validate(stage)
        assert stage['observation']['durations']['executionSeconds'] == 3
        assert stage['observation']['durations']['queueSeconds'] == 5
        persisted = store.db.execute('SELECT record FROM timing_events WHERE seq=?', (int(stage['cursor']),)).fetchone()[0]
        assert json.loads(persisted) == stage['observation']
        clock[0] = 10
        await asyncio.to_thread(store.safe_stage, 'prompt', 'node', 'finished')
        finished_stage = decode(await sub.next())
        assert finished_stage['observation']['durations']['executionSeconds'] == 5
        assert finished_stage['observation']['stages'][0]['durationSeconds'] == 2
        store.streams.remove(sub)
        restarted = store.streams.subscribe(asyncio.get_running_loop(), 4)
        assert decode(await restarted.next())['type'] == 'ready'
        assert not restarted.pending  # Existing journal entries are never replayed live.
        store.streams.remove(restarted)
        store.db.close()
    asyncio.run(scenario())


def test_overflow_bounds_both_frames_and_event_loop_notifications():
    class Loop:
        calls = 0
        def call_soon_threadsafe(self, _):
            self.calls += 1
    loop = Loop()
    hub = TimingStreamHub('source', {'hostId': 'host', 'gpuUuid': None, 'laneId': 'comfyui', 'instanceId': 'boot'})
    sub = hub.subscribe(loop, 0)
    for index in range(1000):
        hub.publish(index + 1, '{}')
        assert len(sub.pending) <= MAX_PENDING
    assert sub.closed and loop.calls == 1
    gap = decode(sub.pending[0])
    validate(gap)
    assert gap['type'] == 'gap' and gap['reasons'] == ['timing_stream_overflow']
    hub.remove(sub)


def test_oversized_events_close_with_gap_and_subscriber_count_is_bounded():
    class Loop:
        def call_soon_threadsafe(self, _):
            pass
    hub = TimingStreamHub('source', {'hostId': 'host', 'gpuUuid': None, 'laneId': 'comfyui', 'instanceId': 'boot'})
    subscribers = [hub.subscribe(Loop(), 0) for _ in range(MAX_SUBSCRIBERS)]
    assert hub.subscribe(Loop(), 0) is None
    hub.publish(1, '"' + 'x' * MAX_EVENT_BYTES + '"')
    for sub in subscribers:
        assert sub.closed
        gap = decode(sub.pending[-1])
        validate(gap)
        assert gap['reasons'] == ['timing_stream_event_too_large']
        hub.remove(sub)
    assert not hub.subscribers


def test_stream_failure_does_not_change_durable_acceptance_or_execution(tmp_path, monkeypatch):
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    def failed(*_):
        raise OSError('telemetry observer failed')
    monkeypatch.setattr(store.streams, 'publish', failed)
    store.safe_event('accepted', item())
    assert store.issue == 'timing_stream_unavailable'
    assert json.loads(store.db.execute('SELECT record FROM timing_records').fetchone()[0])['phase'] == 'queued'
    store.safe_event('started', item())
    assert json.loads(store.db.execute('SELECT record FROM timing_records').fetchone()[0])['phase'] == 'running'
    store.db.close()


def test_http_stream_is_loopback_only_read_only_and_delivers_an_immutable_snapshot(tmp_path):
    async def scenario():
        store = RunTimingStore(str(tmp_path / 'timing.sqlite'), host_id='fixture')
        queue = SimpleNamespace(timing=store, mutex=RLock(), get_current_queue_volatile=lambda: ([], []))
        app, routes = web.Application(), web.RouteTableDef()
        add_routes(routes, queue)
        app.add_routes(routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            changes = store.db.total_changes
            denied = await client.get('/timing/observations/stream', headers={'Origin': 'http://browser.example'})
            assert denied.status == 403
            assert not store.streams.subscribers
            response = await client.get('/timing/observations/stream')
            assert response.status == 200 and response.content_type == 'text/event-stream'
            ready = decode((await asyncio.wait_for(response.content.readline(), 1)).decode())
            validate(ready)
            assert ready['type'] == 'ready' and store.db.total_changes == changes
            await response.content.readline()
            await asyncio.to_thread(store.safe_event, 'accepted', item())
            observed = decode((await asyncio.wait_for(response.content.readline(), 1)).decode())
            validate(observed)
            assert observed['observation']['phase'] == 'queued' and observed['cursor'] == '1'
            response.close()
        finally:
            for sub in tuple(store.streams.subscribers):
                store.streams.remove(sub)
            await client.close()
            store.db.close()
    asyncio.run(scenario())


def test_failed_durable_commit_never_emits_a_live_transition(tmp_path):
    class Loop:
        def call_soon_threadsafe(self, _):
            pass
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    store.safe_event('accepted', item())
    sub = store.streams.subscribe(Loop(), 1)
    sub.pending.clear()
    store.db.execute("CREATE TRIGGER timing_fixture_fail BEFORE INSERT ON timing_events BEGIN SELECT RAISE(ABORT,'fixture'); END")
    store.db.commit()
    store.safe_event('started', item())
    assert store.issue == 'timing_store_unavailable' and not sub.pending
    assert json.loads(store.db.execute('SELECT record FROM timing_records').fetchone()[0])['phase'] == 'queued'
    store.streams.remove(sub)
    store.db.close()
