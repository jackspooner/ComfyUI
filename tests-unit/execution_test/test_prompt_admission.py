import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import comfy.model_management as mm
import execution
import nodes
from execution import CacheType, PromptExecutor


@pytest.fixture
def executor(monkeypatch):
    server = SimpleNamespace(client_id=None, send_sync=Mock(), last_node_id=None)
    executor = PromptExecutor(server, cache_type=CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0})
    monkeypatch.setattr(executor.prompt_model_tracker, "start", Mock())
    monkeypatch.setattr(executor.prompt_model_tracker, "end", Mock())
    monkeypatch.setattr(executor, "_notify_prompt_lifecycle", Mock())
    yield executor
    nodes.interrupt_processing(False)


def test_admission_precedes_cache_evaluation_and_runs_on_cached_prompts(executor, monkeypatch):
    calls = []

    class Output:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT",)}}

        RETURN_TYPES = ()
        FUNCTION = "run"
        OUTPUT_NODE = True

        @classmethod
        def IS_CHANGED(cls, value):
            calls.append("cache")
            return value

        def run(self, value):
            calls.append("node")
            return {"ui": {"value": [value]}}

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "AdmissionOutput", Output)

    def admit():
        assert not mm.processing_interrupted()
        assert executor.status_messages[0][0] == "execution_start"
        assert executor.server.client_id == "client"
        calls.append("admit")

    monkeypatch.setattr(executor, "before_prompt_execution", admit)
    prompt = {"out": {"class_type": "AdmissionOutput", "inputs": {"value": 7}}}
    for prompt_id in ("first", "cached"):
        nodes.interrupt_processing(True)  # Left over from an earlier prompt.
        calls.clear()
        executor.execute(copy.deepcopy(prompt), prompt_id, {"client_id": "client"}, ["out"])
        assert executor.success
        assert calls[0] == "admit"
        assert "cache" in calls
        assert ("node" in calls) == (prompt_id == "first")
        assert executor.history_result["outputs"]["out"]["value"] == [7]


@pytest.mark.parametrize("error", [RuntimeError("admission unavailable"), mm.InterruptProcessingException()])
def test_admission_failure_records_empty_history_cleans_up_and_next_prompt_runs(executor, monkeypatch, error):
    executor.history_result = {"outputs": {"old": "must not leak"}}
    executor.server.last_node_id = "old"
    state = Mock()
    monkeypatch.setattr(execution.comfy.memory_management, "set_ram_cache_release_state", state)
    monkeypatch.setattr(executor, "before_prompt_execution", Mock(side_effect=error))
    executor.execute({}, "failed", {"client_id": "client"})
    assert not executor.success
    assert executor.history_result == {"outputs": {}, "meta": {}}
    assert executor.server.last_node_id is None
    event, payload = executor.status_messages[-1]
    assert event == ("execution_interrupted" if isinstance(error, mm.InterruptProcessingException) else "execution_error")
    assert payload["prompt_id"] == "failed"
    assert payload["node_id"] is None
    assert payload["node_type"] is None
    assert payload["executed"] == []
    executor.prompt_model_tracker.end.assert_called_once_with()
    executor._notify_prompt_lifecycle.assert_called_with("end", "failed")
    state.assert_called_with(None, 0)

    monkeypatch.setattr(executor, "before_prompt_execution", Mock())
    executor.execute({}, "next")
    assert executor.success
    assert [event for event, _ in executor.status_messages] == ["execution_start", "execution_cached", "execution_success"]
    assert executor.prompt_model_tracker.end.call_count == 2
