import asyncio
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from app.maintenance_http import add_routes, middleware, recovery_request
from app.maintenance_queue import MaintenanceQueue
from app.maintenance_store import MaintenanceError, MaintenanceStore, PromptReplay


TOKEN = "recovery-fixture-token-never-used-with-serving-state"


def item(prompt_id="old", value=1):
    return (value, prompt_id, {"out": {"class_type": "Fixture", "inputs": {"value": value}}}, {}, ["out"], {})


@pytest.fixture
def queue(tmp_path):
    directory = tmp_path / "store"
    store = MaintenanceStore(directory)
    native = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), store)
    native.put(item())
    native.put(item("other"))
    store.close()
    reopened = MaintenanceStore(directory)
    result = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), reopened)
    yield result
    result.store.close()


def start(queue):
    status = queue.maintenance_status()
    return queue.maintenance_command("start", {"operationId": "recovery-operation", "ownerId": "fixture",
                                              "targetRevision": "fixture-target", "expectedRevision": status["revision"]})


def request(queue):
    status = queue.maintenance_status()
    return {"operationId": "recovery-operation", "ownerId": "fixture", "expectedRevision": status["revision"],
            "expectedGeneration": status["generation"], "previousInvocationId": "ended-prior-invocation",
            "physicalWorkEnded": True, "evidence": "Fixture prior cgroup terminated; independent resource owner confirms effects ended; outcome unknown."}


def test_confirm_ended_preserves_outcome_digest_replay_and_restart(queue, tmp_path):
    original = queue.store.receipt("old")
    assert original["state"] == "interrupted_unknown"
    start(queue)
    envelope = request(queue)
    result = queue.confirm_ended("old", envelope)
    assert {key: value for key, value in result.items() if key != "recovery"} == original
    assert result["terminal"] is None and result["history"] is None
    assert result["recovery"]["physicalWorkEnded"] is True
    assert queue.maintenance_status()["revision"] == envelope["expectedRevision"] + 1
    assert queue.confirm_ended("old", envelope) == result
    assert queue.maintenance_status()["counts"]["interruptedUnknown"] == 1
    assert not queue.maintenance_status()["safeToStop"]
    with pytest.raises(MaintenanceError, match="receipt_recovery_conflict"):
        queue.confirm_ended("old", {**envelope, "evidence": "different assertion"})
    with pytest.raises(PromptReplay) as replay:
        queue.put(item())
    assert replay.value.receipt == result
    with pytest.raises(MaintenanceError, match="prompt_id_conflict"):
        queue.put(item(value=2))
    queue.store.close()
    queue.store = MaintenanceStore(tmp_path / "store")
    assert queue.store.receipt("old") == result
    assert queue.confirm_ended("old", envelope) == result
    queue.confirm_ended("other", request(queue))
    status = queue.maintenance_status()
    assert status["safeToStop"] and not status["accepting"]
    assert status["counts"]["interruptedUnknown"] == 0
    assert set(status["counts"]) == {"pending", "running", "validating", "finishing", "publishing", "accepted", "interruptedUnknown"}
    queue.maintenance_command("resume", {"operationId": "recovery-operation", "ownerId": "fixture",
                                         "expectedRevision": status["revision"], "expectedGeneration": status["generation"]})
    assert queue.maintenance_status()["accepting"]
    queue.put(item("new"))
    assert queue.get_tasks_remaining() == 1
    assert queue.store.receipt("new")["state"] == "accepted"
    assert queue.store.receipt("old")["state"] == "interrupted_unknown"


@pytest.mark.parametrize("change,code", [
    ({"operationId": "foreign"}, "operation_conflict"),
    ({"ownerId": "foreign"}, "operation_conflict"),
    ({"expectedRevision": 0}, "revision_conflict"),
    ({"expectedGeneration": "old-generation"}, "revision_conflict"),
    ({"physicalWorkEnded": False}, "native_work_unsettled"),
])
def test_recovery_rejects_wrong_owner_stale_or_unconfirmed(queue, change, code):
    start(queue)
    with pytest.raises(MaintenanceError, match=code):
        queue.confirm_ended("old", {**request(queue), **change})
    assert "recovery" not in queue.store.receipt("old")
    assert queue.maintenance_status()["counts"]["interruptedUnknown"] == 2


@pytest.mark.parametrize("state", ["accepted", "terminal"])
def test_recovery_requires_fence_existing_interrupted_receipt_and_ended_invocation(queue, state):
    with pytest.raises(MaintenanceError, match="operation_conflict"):
        queue.confirm_ended("old", request(queue))
    start(queue)
    with pytest.raises(MaintenanceError, match="receipt_not_found"):
        queue.confirm_ended("missing", request(queue))
    with pytest.raises(MaintenanceError, match="invocation_still_current"):
        queue.confirm_ended("old", {**request(queue), "previousInvocationId": queue.store.invocation})
    queue.store.db.execute("UPDATE receipts SET state=? WHERE id='other'", (state,))
    queue.store.db.commit()
    with pytest.raises(MaintenanceError, match="receipt_not_interrupted"):
        queue.confirm_ended("other", request(queue))


@pytest.mark.parametrize("work", ["pending", "running", "validating", "finishing", "publishing", "storage"])
def test_recovery_rejects_current_work_or_storage_failure(queue, work):
    start(queue)
    if work == "pending":
        queue.queue.append(item("current"))
    elif work == "running":
        queue.currently_running[1] = item("current")
    elif work in ("validating", "finishing"):
        setattr(queue, work, 1)
    elif work == "publishing":
        queue.publication_pending[1] = {"prompt": [1, "current"]}
    else:
        queue.store.failure = "receipt_store_failure"
    with pytest.raises(MaintenanceError, match="native_work_unsettled"):
        queue.confirm_ended("old", request(queue))
    assert "recovery" not in queue.store.receipt("old")


@pytest.mark.parametrize("change", [{"physicalWorkEnded": 1}, {"evidence": " "}, {"evidence": "x" * 2049},
                                     {"evidence": "line\nbreak"}, {"previousInvocationId": "invalid id"}, {"extra": True}])
def test_recovery_request_is_strict(queue, change):
    with pytest.raises(MaintenanceError, match="invalid_maintenance_request"):
        recovery_request({**request(queue), **change})


def test_recovery_http_authentication_and_receipt_readback(queue):
    start(queue)
    envelope = request(queue)

    async def run():
        app = web.Application(middlewares=[middleware(queue, TOKEN)])
        routes = web.RouteTableDef()
        add_routes(routes, queue)
        aliases = web.RouteTableDef()
        for route in routes:
            aliases.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
        app.add_routes(aliases)
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            path = "/maintenance/receipts/old/confirm-ended"
            assert (await client.post(path, json=envelope)).status == 401
            headers = {"Authorization": "Bearer " + TOKEN}
            assert (await client.post(path, json=envelope, headers={**headers, "Origin": "null"})).status == 403
            assert (await client.post(path, json={**envelope, "physicalWorkEnded": False}, headers=headers)).status == 400
            response = await client.post(path, json=envelope, headers=headers)
            assert response.status == 200
            result = await response.json()
            assert "digest" not in result and result["state"] == "interrupted_unknown"
            assert result["terminal"] is None and result["history"] is None
            assert (await client.get("/maintenance/receipts/old", headers=headers)).status == 200
            assert await (await client.post(path, json=envelope, headers=headers)).json() == result
            assert await (await client.post("/api" + path, json=envelope, headers=headers)).json() == result
    asyncio.run(run())
