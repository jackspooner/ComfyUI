from concurrent.futures import ThreadPoolExecutor
import asyncio
from threading import Barrier
from types import SimpleNamespace

import pytest

from app.queue_control import QueueControlError, QueueControlMixin
from execution import PromptQueue
from app.maintenance_store import MaintenanceStore
from app.run_timing_queue import TimedMaintenanceQueue
from app.queue_control_http import add_routes
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


class ControlledQueue(QueueControlMixin, PromptQueue):
    pass


def queue(tmp_path):
    result = ControlledQueue(SimpleNamespace(queue_updated=lambda: None))
    result.configure_queue_control(str(tmp_path / 'control.sqlite'))
    return result


def item(prompt_id):
    return (1, prompt_id, {'node': {'class_type': 'Fixture', 'inputs': {'seed': 42}}}, {}, ['node'])


def control(q, action, ids, operation='op'):
    return q.control_pending({'action': action, 'promptIds': ids, 'operationId': operation,
                              'expectedRevision': q.queue_control_snapshot()['revision']})


def test_hold_skips_work_and_resume_preserves_identity(tmp_path):
    q = queue(tmp_path)
    for key in ('a', 'b', 'c'):
        q.put(item(key))
    paused = control(q, 'pause', ['b'])
    assert paused['state'] == 'paused'
    assert [q.get(0)[0][1], q.get(0)[0][1]] == ['a', 'c']
    assert q.get(0) is None
    with pytest.raises(QueueControlError, match='identity_retained'):
        q.put(item('b'))
    control(q, 'resume', ['b'], 'resume')
    assert q.get(0)[0] == item('b')


def test_whole_selection_rejected_before_any_hold(tmp_path):
    q = queue(tmp_path)
    q.put(item('a'))
    q.put(item('b'))
    q.get(0)
    with pytest.raises(QueueControlError) as error:
        control(q, 'pause', ['a', 'b'])
    assert error.value.blocked == [{'promptId': 'a', 'reason': 'running'}]
    assert q.get(0)[0][1] == 'b'
    assert q.queue_control_snapshot()['revision'] == 0


def test_restart_restores_hold_without_automatic_execution(tmp_path):
    q = queue(tmp_path)
    q.put(item('a'))
    control(q, 'pause', ['a'])
    q.queue_control_store.close()
    q = queue(tmp_path)
    assert q.get(0) is None
    control(q, 'resume', ['a'], 'resume')
    assert q.get(0)[0] == item('a')


def test_uncertain_resume_after_restart_is_never_replayed(tmp_path):
    q = queue(tmp_path)
    q.put(item('a'))
    control(q, 'pause', ['a'])
    q.queue_control_store.db.execute("UPDATE held_prompts SET state='resuming'")
    q.queue_control_store.db.commit()
    q.queue_control_store.close()
    q = queue(tmp_path)
    with pytest.raises(QueueControlError) as error:
        control(q, 'resume', ['a'], 'resume')
    assert error.value.blocked[0]['reason'] == 'unknown'
    assert q.get(0) is None


def test_confirmed_resume_retry_requires_retained_native_evidence(tmp_path):
    q = queue(tmp_path)
    q.put(item('a'))
    control(q, 'pause', ['a'])
    request = {'action': 'resume', 'promptIds': ['a'], 'operationId': 'resume',
               'expectedRevision': q.queue_control_snapshot()['revision']}
    receipt = q.control_pending(request)
    assert q.control_pending(request) == receipt
    q.queue_control_store.close()

    # Restart loses the in-memory pending prompt. Its old release receipt must
    # not stand in for current queue/history evidence or authorize replay.
    q = queue(tmp_path)
    with pytest.raises(QueueControlError, match='queue_control_uncertain'):
        q.control_pending(request)
    assert q.get(0) is None
    assert q.queue_control_snapshot()['prompts'] == [{'promptId': 'a', 'state': 'released'}]
    q.queue_control_store.close()


def test_confirmed_resume_retry_accepts_running_and_terminal_evidence(tmp_path):
    q = queue(tmp_path)
    q.put(item('a'))
    control(q, 'pause', ['a'])
    request = {'action': 'resume', 'promptIds': ['a'], 'operationId': 'resume',
               'expectedRevision': q.queue_control_snapshot()['revision']}
    receipt = q.control_pending(request)
    _, index = q.get(0)
    assert q.control_pending(request) == receipt
    q.task_done(index, {'outputs': {}}, PromptQueue.ExecutionStatus('success', True, []))
    assert q.control_pending(request) == receipt
    q.queue_control_store.close()


def test_retry_and_stale_revision_cannot_undo_newer_control(tmp_path):
    q = queue(tmp_path)
    q.put(item('a'))
    request = {'action': 'pause', 'promptIds': ['a'], 'operationId': 'pause', 'expectedRevision': 0}
    first = q.control_pending(request)
    assert q.control_pending(request) == first
    control(q, 'resume', ['a'], 'resume')
    with pytest.raises(QueueControlError, match='revision_conflict'):
        q.control_pending(request)
    assert control(q, 'pause', ['a'], 'pause2')['state'] == 'paused'


def test_held_maintenance_receipts_survive_restart_and_allow_other_admission(tmp_path):
    directory = tmp_path / 'maintenance'
    store = MaintenanceStore(str(directory))
    q = TimedMaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), store)
    q.timing = None
    q.configure_queue_control(str(tmp_path / 'control.sqlite'))
    q.put(item('a'))
    control(q, 'pause', ['a'])
    assert 'queue_receipt_mismatch' not in q.maintenance_status()['blockers']
    q.put(item('b'))
    started, index = q.get(0)
    assert started[1] == 'b'
    q.task_done(index, {'outputs': {}}, PromptQueue.ExecutionStatus('success', True, []))
    q.queue_control_store.close()
    store.close()
    store = MaintenanceStore(str(directory))
    q = TimedMaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), store)
    q.timing = None
    q.configure_queue_control(str(tmp_path / 'control.sqlite'))
    assert store.receipt('a')['state'] == 'accepted'
    assert 'queue_receipt_mismatch' not in q.maintenance_status()['blockers']
    control(q, 'resume', ['a'], 'resume')
    assert q.get(0)[0] == item('a')
    q.queue_control_store.close()
    store.close()


def test_pause_and_dequeue_race_has_exactly_one_winner(tmp_path):
    for index in range(40):
        q = queue(tmp_path / str(index))
        q.put(item('a'))
        gate = Barrier(2)
        def dequeue():
            gate.wait()
            return q.get(0)
        def pause():
            gate.wait()
            try:
                return control(q, 'pause', ['a'])
            except QueueControlError as error:
                return error
        with ThreadPoolExecutor(2) as workers:
            native = workers.submit(dequeue)
            receipt = workers.submit(pause)
            started, result = native.result(), receipt.result()
        if isinstance(result, QueueControlError):
            assert result.blocked == [{'promptId': 'a', 'reason': 'running'}]
            assert started[0][1] == 'a'
        else:
            assert result['state'] == 'paused'
            assert started is None
        q.queue_control_store.close()


def test_http_controls_and_browser_rejection(tmp_path):
    q = queue(tmp_path)
    q.put(item('a'))
    async def check():
        routes = web.RouteTableDef()
        add_routes(routes, q)
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.get('/queue/control')
            assert response.status == 200
            assert (await response.json())['revision'] == 0
            request = {'action': 'pause', 'promptIds': ['a'], 'operationId': 'pause', 'expectedRevision': 0}
            response = await client.post('/queue/control', json=request)
            assert response.status == 200
            first = await response.json()
            assert first['state'] == 'paused' and first['receiptId']
            response = await client.post('/queue/control', json=request)
            assert await response.json() == first
            response = await client.post('/queue/control', json=request, headers={'Origin': 'http://localhost'})
            assert response.status == 403
            response = await client.post('/queue/control', json={**request, 'promptIds': []})
            assert response.status == 400
    asyncio.run(check())
    q.queue_control_store.close()
