import asyncio
import json
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from app.maintenance_http import add_routes, authorize, command_request, configured_store, middleware
from app.maintenance_queue import MaintenanceQueue
from app.maintenance_store import MaintenanceError, MaintenanceStore, PromptReplay
from execution import PromptQueue


TOKEN = "test-operator-token-not-used-outside-fixtures"


@pytest.fixture
def store(tmp_path):
    result = MaintenanceStore(tmp_path / "state")
    yield result
    result.close()


@pytest.fixture
def queue(store):
    return MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), store)


def item(prompt_id="first", value=1):
    return (value, prompt_id, {"out": {"class_type": "Fixture", "inputs": {"value": value}}},
            {"private_metadata": "not persisted", "create_time": 1}, ["out"], {"token": "secret"})


def start(queue, operation="op-1"):
    return queue.maintenance_command("start", {"operationId": operation, "ownerId": "fixture",
                                              "targetRevision": "test-revision", "expectedRevision": queue.maintenance_status()["revision"]})


def release(queue, action="resume", operation="op-1"):
    state = queue.maintenance_status()
    request = {"operationId": operation, "ownerId": "fixture", "expectedRevision": state["revision"],
               "expectedGeneration": state["generation"]}
    return queue.maintenance_command(action, request), request


def done(queue, index):
    queue.task_done(index, {"outputs": {"out": {"text": ["complete"], "images": [{"filename": "fixture.png", "subfolder": "", "type": "output"}]}}},
                    PromptQueue.ExecutionStatus("success", True, [["execution_error", {"private": "secret"}]]))


def test_validation_fence_final_put_race(queue):
    entered = threading.Event()
    proceed = threading.Event()
    result = []

    def validating():
        with queue.admission():
            entered.set()
            assert proceed.wait(2)
            try:
                queue.put(item())
            except MaintenanceError as exc:
                result.append((exc.code, exc.accepted))

    worker = threading.Thread(target=validating)
    worker.start()
    assert entered.wait(2)
    state = start(queue)
    assert state["counts"]["validating"] == 1
    assert not state["safeToStop"]
    proceed.set()
    worker.join(2)
    assert not worker.is_alive()
    assert result == [("maintenance_fenced", False)]
    assert queue.get_tasks_remaining() == 0
    assert queue.maintenance_status()["safeToStop"]
    with pytest.raises(MaintenanceError, match="maintenance_fenced"):
        queue.put(item("direct-producer"))


def test_running_pending_drain_and_worker_publication_boundary(queue):
    queue.put(item("a"))
    queue.put(item("b", 2))
    _, index = queue.get()
    assert start(queue)["counts"]["pending"] == 1
    done(queue, index)
    assert not queue.maintenance_status()["safeToStop"]
    _, index = queue.get()
    done(queue, index)
    assert queue.maintenance_status()["counts"]["finishing"] == 1
    assert not queue.maintenance_status()["safeToStop"]
    # A worker entering its next idle dequeue has completed the entire cycle.
    waiting = threading.Event()
    original_wait = queue.not_empty.wait

    def observed_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    queue.not_empty.wait = observed_wait
    worker = threading.Thread(target=lambda: queue.get(timeout=0.01))
    worker.start()
    assert waiting.wait(2)
    state = queue.maintenance_status()
    assert state["safeToStop"] and state["counts"]["accepted"] == 0
    release(queue)
    worker.join(2)
    assert not worker.is_alive()


def test_publication_failure_blocks_and_retains_running(queue, monkeypatch):
    queue.put(item())
    queue.put(item("second", 2))
    _, index = queue.get()
    start(queue)
    complete = queue.store.complete

    def fail(prompt_id, history):
        with queue.store.transaction():
            raise sqlite3.OperationalError("fixture failure, not a live database")

    monkeypatch.setattr(queue.store, "complete", fail)
    done(queue, index)
    state = queue.maintenance_status()
    assert not state["safeToStop"]
    assert state["counts"]["publishing"] == 1
    assert "receipt_store_failure" in state["blockers"]
    with pytest.raises(MaintenanceError, match="maintenance_not_ready"):
        release(queue)
    # The same native worker is free to drain the second accepted prompt.
    _, second = queue.get()
    monkeypatch.setattr(queue.store, "complete", complete)
    done(queue, second)
    state = queue.maintenance_status()
    queue.maintenance_command("retry-publication", {"operationId": "op-1", "ownerId": "fixture",
                                                    "expectedRevision": state["revision"], "expectedGeneration": state["generation"]})
    assert queue.store.receipt("first")["state"] == "terminal"
    assert queue.maintenance_status()["counts"]["publishing"] == 0
    assert queue.maintenance_status()["blockers"] == []


def test_operation_conflicts_replay_revision_and_own_release(queue):
    state = start(queue)
    assert start(queue) == state
    with pytest.raises(MaintenanceError, match="operation_conflict"):
        start(queue, "foreign")
    request = {"operationId": "op-1", "ownerId": "fixture", "expectedRevision": state["revision"] - 1,
               "expectedGeneration": state["generation"]}
    with pytest.raises(MaintenanceError, match="revision_conflict"):
        queue.maintenance_command("resume", request)
    result, request = release(queue)
    assert result["accepting"] and not result["safeToStop"]
    assert queue.maintenance_command("resume", request) == result
    with pytest.raises(MaintenanceError, match="operation_already_released"):
        start(queue)
    start(queue, "new")
    # Old response-loss retries must not release a newer operation.
    assert queue.maintenance_command("resume", request)["operationId"] == "new"


def test_restart_retains_gate_sanitized_history_and_requires_new_generation(tmp_path):
    directory = tmp_path / "store"
    store = MaintenanceStore(directory)
    q = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), store)
    q.put(item())
    _, index = q.get()
    done(q, index)
    state = start(q)
    store.close()
    reopened = MaintenanceStore(directory)
    try:
        q2 = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), reopened)
        status = q2.maintenance_status()
        assert status["operationId"] == "op-1" and status["safeToStop"]
        assert status["generation"] != state["generation"]
        history = q2.get_history("first")["first"]
        assert history["outputs"]["out"]["text"] == ["complete"]
        assert "secret" not in json.dumps(history)
        assert "private_metadata" not in json.dumps(history)
        assert "class_type" not in json.dumps(history)
        with pytest.raises(MaintenanceError, match="maintenance_fenced"):
            q2.put(item("new"))
        with pytest.raises(MaintenanceError, match="abort_after_restart_forbidden"):
            release(q2, "abort")
        release(q2)
    finally:
        reopened.close()


def test_crash_accepted_is_unknown_never_replayed_or_ready(tmp_path):
    directory = tmp_path / "store"
    store = MaintenanceStore(directory)
    q = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), store)
    q.put(item())
    start(q)
    store.close()
    reopened = MaintenanceStore(directory)
    try:
        q2 = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), reopened)
        state = q2.maintenance_status()
        assert state["counts"]["interruptedUnknown"] == 1
        assert not state["safeToStop"] and not state["accepting"]
        assert q2.get_tasks_remaining() == 0
        with pytest.raises(PromptReplay) as exc:
            q2.put(item())
        assert exc.value.receipt["state"] == "interrupted_unknown"
        with pytest.raises(MaintenanceError, match="maintenance_not_ready"):
            release(q2)
    finally:
        reopened.close()


def test_prompt_identity_collision_and_replay(queue):
    queue.put(item())
    duplicate = list(item())
    duplicate[0] = 42
    duplicate[3] = {**duplicate[3], "create_time": 9000, "client_id": "changed-client"}
    with pytest.raises(PromptReplay):
        queue.put(tuple(duplicate))
    assert queue.get_tasks_remaining() == 1
    with pytest.raises(MaintenanceError, match="prompt_id_conflict"):
        queue.put(item(value=2))


def test_cancel_pending_receipts_and_protected_drain(queue):
    queue.put(item())
    queue.put(item("second", 2))
    queue.delete_queue_item(lambda x: x[1] == "first")
    assert queue.store.receipt("first")["state"] == "terminal"
    start(queue)
    with pytest.raises(MaintenanceError, match="maintenance_drain_protected"):
        queue.wipe_queue()
    assert queue.get_tasks_remaining() == 1


def test_config_opt_in_and_permissions(tmp_path, monkeypatch):
    monkeypatch.delenv("COMFYUI_MAINTENANCE_DIR", raising=False)
    monkeypatch.delenv("COMFYUI_MAINTENANCE_TOKEN_FILE", raising=False)
    assert configured_store() == (None, None)
    monkeypatch.setenv("COMFYUI_MAINTENANCE_DIR", str(tmp_path / "state"))
    with pytest.raises(ValueError, match="both"):
        configured_store()
    token = tmp_path / "token"
    token.write_text(TOKEN)
    token.chmod(0o644)
    monkeypatch.setenv("COMFYUI_MAINTENANCE_TOKEN_FILE", str(token))
    with pytest.raises(ValueError, match="owner-only"):
        configured_store()
    token.chmod(0o600)
    store, secret = configured_store()
    try:
        assert secret == TOKEN
        with pytest.raises(BlockingIOError):
            MaintenanceStore(tmp_path / "state")
    finally:
        store.close()


@pytest.mark.parametrize("remote,host,headers,code", [
    ("10.0.0.1", "localhost", {}, "operator_local_only"),
    ("127.0.0.1", "evil.example", {}, "operator_local_only"),
    ("127.0.0.1", "localhost", {"Origin": "http://localhost"}, "operator_origin_forbidden"),
    ("127.0.0.1", "localhost", {"X-Forwarded-For": "127.0.0.1"}, "operator_origin_forbidden"),
    ("127.0.0.1", "localhost", {"Authorization": "Bearer wrong"}, "operator_unauthorized"),
])
def test_auth_and_csrf(remote, host, headers, code):
    request = SimpleNamespace(remote=remote, host=host, headers={"Authorization": "Bearer " + TOKEN, **headers})
    with pytest.raises(MaintenanceError, match=code):
        authorize(request, TOKEN)


@pytest.mark.parametrize("remote,host", [("127.0.0.1", "localhost:8188"), ("::1", "[::1]:8188"), ("::ffff:127.0.0.1", "localhost:8188")])
def test_local_authorization(remote, host):
    authorize(SimpleNamespace(remote=remote, host=host, headers={"Authorization": "Bearer " + TOKEN}), TOKEN)


def test_http_contract_and_definitive_rejection(queue):
    async def run():
        app = web.Application(middlewares=[middleware(queue, TOKEN)])
        routes = web.RouteTableDef()
        add_routes(routes, queue)

        @routes.post("/prompt")
        async def submit(request):
            queue.put(item())
            return web.json_response({"prompt_id": "first", "number": 1, "node_errors": {}})

        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/maintenance")).status == 401
            headers = {"Authorization": "Bearer " + TOKEN}
            response = await client.get("/maintenance", headers=headers)
            state = await response.json()
            assert state["accepting"]
            response = await client.post("/maintenance/start", headers=headers, json={"operationId": "http", "ownerId": "fixture", "targetRevision": "test", "expectedRevision": state["revision"]})
            assert response.status == 200
            assert (await response.json())["safeToStop"]
            response = await client.post("/prompt", json={})
            assert response.status == 503
            body = await response.json()
            assert body["accepted"] is False and body["error"]["type"] == "maintenance_fenced"
            assert queue.get_tasks_remaining() == 0
            assert (await client.get("/maintenance", headers={**headers, "Origin": "http://localhost"})).status == 403
            state = await (await client.get("/maintenance", headers=headers)).json()
            command = {"operationId": "http", "ownerId": "fixture", "expectedRevision": state["revision"],
                       "expectedGeneration": state["generation"]}
            response = await client.post("/maintenance/retry-publication", headers=headers, json=command)
            assert response.status == 200 and (await response.json())["publicationErrors"] == []
            response = await client.post("/maintenance/abort", headers=headers, json=command)
            assert response.status == 200 and (await response.json())["accepting"]
            response = await client.post("/maintenance/abort", headers=headers, json=command)
            assert response.status == 200
    asyncio.run(run())


def test_strict_command_schema():
    with pytest.raises(MaintenanceError, match="invalid_maintenance_request"):
        command_request("start", {"operationId": "a"})


def test_unknown_queue_command_never_releases(queue):
    start(queue)
    with pytest.raises(MaintenanceError, match="invalid_maintenance_request"):
        queue.maintenance_command("typo", {})
    assert queue.maintenance_status()["operationId"] == "op-1"


def test_ordinary_queue_remains_unchanged():
    queue = PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    queue.put(item())
    _, index = queue.get()
    done(queue, index)
    assert queue.get_tasks_remaining() == 0
    assert queue.get_history("first")["first"]["outputs"]["out"]["text"] == ["complete"]


def test_history_deletion_stays_hidden_on_restart(tmp_path):
    directory = tmp_path / "state"
    store = MaintenanceStore(directory)
    q = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), store)
    q.put(item())
    _, index = q.get()
    done(q, index)
    q.delete_history_item("first")
    store.close()
    reopened = MaintenanceStore(directory)
    try:
        q2 = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), reopened)
        assert q2.get_history("first") == {}
        assert reopened.receipt("first")["state"] == "terminal"
    finally:
        reopened.close()


def test_receipt_capacity_and_retention(queue, monkeypatch):
    monkeypatch.setattr(queue.store, "max_receipts", 1)
    queue.put(item())
    with pytest.raises(MaintenanceError, match="receipt_capacity"):
        queue.put(item("second", 2))
    _, index = queue.get()
    done(queue, index)
    queue.store.db.execute("UPDATE receipts SET terminal=0")
    queue.store.db.commit()
    queue.put(item("second", 2))
    assert queue.store.receipt("first") is None


def test_capacity_increase_preserves_receipts_replay_and_fence(tmp_path):
    directory = tmp_path / "state"
    original = MaintenanceStore(directory, max_receipts=1)
    q = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), original)
    q.put(item())
    _, index = q.get()
    done(q, index)
    retained = original.receipt("first")
    with pytest.raises(MaintenanceError, match="receipt_capacity"):
        q.put(item("second", 2))
    start(q)
    original.close()
    reopened = MaintenanceStore(directory, max_receipts=2)
    try:
        assert reopened.receipt("first") == retained
        with pytest.raises(PromptReplay):
            reopened.accept(item())
        with pytest.raises(MaintenanceError, match="maintenance_fenced"):
            reopened.accept(item("second", 2))
        q2 = MaintenanceQueue(SimpleNamespace(queue_updated=lambda: None), reopened)
        release(q2)
        q2.put(item("second", 2))
        assert reopened.receipt("first") == retained
        assert reopened.receipt("second")["state"] == "accepted"
        with pytest.raises(MaintenanceError, match="receipt_capacity"):
            q2.put(item("third", 3))
    finally:
        reopened.close()


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "2"])
def test_invalid_capacity_does_not_create_store(tmp_path, capacity):
    directory = tmp_path / "state"
    with pytest.raises(ValueError, match="positive integer"):
        MaintenanceStore(directory, max_receipts=capacity)
    assert not directory.exists()


@pytest.mark.parametrize("capacity", ["", "0", "-1", "1.5", " 2", "１２", "1000000000"])
def test_invalid_capacity_environment(tmp_path, monkeypatch, capacity):
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN)
    token_file.chmod(0o600)
    monkeypatch.setenv("COMFYUI_MAINTENANCE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COMFYUI_MAINTENANCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("COMFYUI_MAINTENANCE_MAX_RECEIPTS", capacity)
    with pytest.raises(ValueError, match="COMFYUI_MAINTENANCE_MAX_RECEIPTS"):
        configured_store()
    assert not (tmp_path / "state").exists()


def test_capacity_environment_reaches_store(tmp_path, monkeypatch):
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN)
    token_file.chmod(0o600)
    monkeypatch.setenv("COMFYUI_MAINTENANCE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COMFYUI_MAINTENANCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("COMFYUI_MAINTENANCE_MAX_RECEIPTS", "100000")
    configured, _ = configured_store()
    try:
        assert configured.max_receipts == 100000
    finally:
        configured.close()


def test_serialization_failure_is_not_lost_or_reexecuted(queue):
    queue.put(item())
    _, index = queue.get()
    queue.task_done(index, {"outputs": {"invalid": float("nan")}}, PromptQueue.ExecutionStatus("success", True, []))
    assert queue.maintenance_status()["counts"]["publishing"] == 1
    assert not queue.maintenance_status()["accepting"]
    assert queue.store.receipt("first")["state"] == "accepted"


def test_retry_repairs_other_entries_after_unserializable_result(queue, monkeypatch):
    queue.put(item())
    queue.put(item("second", 2))
    _, first = queue.get()
    queue.task_done(first, {"outputs": {"invalid": float("nan")}}, PromptQueue.ExecutionStatus("success", True, []))
    _, second = queue.get()
    original = queue.store.complete
    monkeypatch.setattr(queue.store, "complete", Mock(side_effect=MaintenanceError("receipt_store_failure", 503)))
    done(queue, second)
    monkeypatch.setattr(queue.store, "complete", original)
    state = start(queue)
    result = queue.maintenance_command("retry-publication", {"operationId": "op-1", "ownerId": "fixture",
                                       "expectedRevision": state["revision"], "expectedGeneration": state["generation"]})
    assert result["publicationErrors"] == [{"promptId": "first", "code": "receipt_serialization_failure"}]
    assert result["counts"]["publishing"] == 1
    assert result["counts"]["running"] == 0
    assert not result["safeToStop"]
    assert queue.store.receipt("second")["state"] == "terminal"


def test_retry_after_native_publication_already_popped_running(queue):
    queue.put(item())
    _, index = queue.get()
    queue.server.queue_updated = Mock(side_effect=RuntimeError("fixture notification failure"))
    done(queue, index)
    state = start(queue)
    assert state["counts"]["running"] == 0
    assert state["blockers"] == ["completion_publication_failure"]
    assert queue.store.receipt("first")["state"] == "terminal"
    queue.delete_history_item("first")
    queue.server.queue_updated = lambda: None
    result = queue.maintenance_command("retry-publication", {"operationId": "op-1", "ownerId": "fixture",
                                       "expectedRevision": state["revision"], "expectedGeneration": state["generation"]})
    assert result["publicationErrors"] == []
    assert result["counts"]["publishing"] == 0
    assert result["blockers"] == []
    assert queue.get_history("first") == {}


def test_worker_store_read_failure_still_drains_accepted(queue, monkeypatch):
    queue.put(item())
    with monkeypatch.context() as patcher:
        patcher.setattr(queue.store, "state", Mock(side_effect=sqlite3.OperationalError("fixture read failure")))
        _, index = queue.get()
    done(queue, index)
    assert queue.store.receipt("first")["state"] == "terminal"
    assert queue.maintenance_status()["blockers"] == ["receipt_store_failure"]


@pytest.mark.parametrize("version", [0, 1])
def test_seven_column_receipt_store_upgrade_and_unknown_version(tmp_path, version):
    directory = tmp_path / "state"
    store = MaintenanceStore(directory)
    store.close()
    db = sqlite3.connect(directory / "maintenance.sqlite3")
    db.execute("ALTER TABLE receipts DROP COLUMN visible")
    db.execute(f"PRAGMA user_version={version}")
    db.execute("INSERT INTO receipts VALUES ('old','digest',1,'terminal',1,2,?)", (json.dumps({"outputs": {}}),))
    db.commit()
    db.close()
    store = MaintenanceStore(directory)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == 2
    assert store.receipt("old")["visible"] == 1
    store.db.execute("PRAGMA user_version=999")
    store.close()
    with pytest.raises(ValueError, match="unsupported maintenance store version"):
        MaintenanceStore(directory)


def test_store_read_failure_is_domain_error_and_closed(queue):
    queue.store.db.close()
    with pytest.raises(MaintenanceError) as exc:
        queue.maintenance_status()
    assert exc.value.code == "receipt_store_failure"
    assert exc.value.accepted is None
    with pytest.raises(MaintenanceError, match="receipt_store_failure"):
        with queue.admission():
            pytest.fail("admission after receipt read failure")


def test_native_server_handler_and_api_aliases(tmp_path, monkeypatch):
    import server as server_module
    monkeypatch.setattr(server_module, "dependencies_available", lambda: False)
    monkeypatch.setattr(server_module.FrontendManager, "init_frontend", lambda version: str(tmp_path))
    monkeypatch.setattr(server_module.folder_paths, "user_directory", str(tmp_path / "users"))
    token = tmp_path / "token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    monkeypatch.setenv("COMFYUI_MAINTENANCE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COMFYUI_MAINTENANCE_TOKEN_FILE", str(token))

    async def run():
        assets = Mock(enabled=False)
        instance = server_module.PromptServer(asyncio.get_running_loop(), assets)
        aliases = web.RouteTableDef()
        for route in instance.routes:
            aliases.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
        instance.app.add_routes(aliases)
        instance.app.add_routes(instance.routes)
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def validate(prompt_id, prompt, targets):
            entered.set()
            await proceed.wait()
            return True, None, ["out"], {}

        monkeypatch.setattr(server_module.execution, "validate_prompt", validate)
        instance.node_replace_manager.apply_replacements = lambda prompt: None
        try:
            async with TestClient(TestServer(instance.app)) as client:
                submitting = asyncio.create_task(client.post("/api/prompt", json={"prompt": item()[2]}))
                await asyncio.wait_for(entered.wait(), 2)
                headers = {"Authorization": "Bearer " + TOKEN}
                status = await (await client.get("/api/maintenance", headers=headers)).json()
                response = await client.post("/api/maintenance/start", headers=headers,
                                             json={"operationId": "http", "ownerId": "test", "targetRevision": "test",
                                                   "expectedRevision": status["revision"]})
                assert response.status == 200
                assert not (await response.json())["safeToStop"]
                proceed.set()
                response = await submitting
                assert response.status == 503 and (await response.json())["accepted"] is False
                status = await (await client.get("/api/maintenance", headers=headers)).json()
                assert status["safeToStop"]
                assert (await client.get("/api/maintenance", headers={**headers, "Origin": "null"})).status == 403
                response = await client.post("/api/maintenance/resume", headers=headers,
                                             json={"operationId": "http", "ownerId": "test", "expectedRevision": status["revision"],
                                                   "expectedGeneration": status["generation"]})
                assert response.status == 200
                response = await client.post("/prompt", json={"prompt": item()[2]})
                assert response.status == 200
                prompt_id = (await response.json())["prompt_id"]
                _, index = instance.prompt_queue.get()
                done(instance.prompt_queue, index)
                response = await client.get("/maintenance/receipts/" + prompt_id, headers=headers)
                assert response.status == 200 and (await response.json())["state"] == "terminal"
                response = await client.post("/history", json={"delete": [prompt_id]})
                assert response.status == 200
                assert instance.prompt_queue.get_history(prompt_id) == {}
                response = await client.get("/maintenance/receipts/" + prompt_id, headers=headers)
                assert (await response.json())["visible"] == 0
        finally:
            instance.prompt_queue.store.close()
    asyncio.run(run())
