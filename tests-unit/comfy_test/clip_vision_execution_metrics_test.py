import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import jsonschema

from app import run_timing
from app.run_timing_metrics import attach_runtime_metrics
from comfy import clip_vision, execution_metrics
from comfy_execution.utils import CurrentNodeContext


@pytest.fixture(autouse=True)
def reset_observer(monkeypatch):
    execution_metrics.set_observer(None)
    monkeypatch.setattr(clip_vision.comfy.model_management, 'load_model_gpu', lambda *_: None)
    monkeypatch.setattr(clip_vision.comfy.model_management, 'intermediate_device', lambda: 'cpu')
    yield
    execution_metrics.set_observer(None)


def encoder(model_type='clip_vision_model', all_hidden=False):
    value = clip_vision.ClipVisionModel.__new__(clip_vision.ClipVisionModel)
    value.model_type, value.return_all_hidden_states = model_type, all_hidden
    value.load_device, value.dtype = 'cpu', torch.float32
    value.patcher = SimpleNamespace(offload_device='cpu', is_dynamic=lambda: False)
    value.image_size = 0 if model_type == 'siglip2_vision_model' else 8
    value.image_mean, value.image_std = [0.5] * 3, [0.5] * 3
    value.config = {'patch_size': 2, 'num_patches': 12}
    def forward(pixel_values, intermediate_output):
        last = pixel_values.flatten(2).transpose(1, 2)
        hidden = torch.stack((last, last + 1), dim=1) if all_hidden else last + 1
        return last, hidden, last.mean(1), None
    value.model = forward
    return value


@pytest.mark.parametrize('model_type', ['clip_vision_model', 'siglip_vision_model', 'siglip2_vision_model', 'dinov2', 'dinov3'])
@pytest.mark.parametrize('crop', [False, True])
def test_native_processed_shapes_positions_and_observer_failures_preserve_outputs(model_type, crop):
    value = encoder(model_type, all_hidden=model_type == 'siglip_vision_model')
    pixels = torch.arange(2 * 10 * 18 * 4, dtype=torch.float32).reshape(2, 10, 18, 4) / 1440
    expected = value.encode_image(pixels, crop=crop)
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = value.encode_image(pixels, crop=crop)
    for key in ('last_hidden_state', 'penultimate_hidden_states', 'image_embeds'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual.image_sizes == expected.image_sizes
    facts = dict(events)
    assert facts['clip_vision_source'] == {'shape': (2, 10, 18, 4), 'crop': crop}
    assert facts['clip_vision_plan']['shape'] == tuple((2, *actual.image_sizes[0]))
    assert facts['clip_vision_plan']['model_type'] == model_type
    assert facts['clip_vision_plan']['memory_strategy'] == 'standard'
    assert facts['clip_vision_forward']['succeeded'] and facts['clip_vision_preprocess']['succeeded']
    assert facts['clip_vision_output']['shape'] == tuple(actual.last_hidden_state.shape)
    assert len(events) == 5
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer offline')))
    torch.testing.assert_close(value.encode_image(pixels, crop=crop).last_hidden_state, expected.last_hidden_state)


def graph():
    return {'image': {'class_type': 'LoadImage', 'inputs': {'image': '/PRIVATE/ref.png'}},
            'load': {'class_type': 'CLIPVisionLoader', 'inputs': {'clip_name': 'vision-weight'}},
            'encode': {'class_type': 'CLIPVisionEncode', 'inputs': {
                'clip_vision': ['load', 0], 'image': ['image', 0], 'crop': 'center'}}}


def store(tmp_path):
    clock = [0.0]
    value = run_timing.RunTimingStore(str(tmp_path / 'timing.sqlite'), now=lambda: clock[0])
    item = (1, 'prompt', graph(), {}, ['encode'])
    selections = []
    def identity(key, selected):
        selections.append((key, selected))
        return 'weight'
    with patch.object(run_timing, 'selected_file_identity', identity):
        value.event('accepted', item)
        value.event('started', item)
    assert selections == [('clip_vision_name', 'vision-weight')] * 2
    attach_runtime_metrics(SimpleNamespace(timing=value, mutex=Lock(), currently_running={0: item}))
    return value, item, clock


def stage(value):
    return next(stage for stage in value.page('0', 100, [], [])['observations'][-1]['stages'] if stage['id'] == 'encode')


def test_native_records_resolve_only_vision_work_and_keep_outcomes_separate(tmp_path):
    value, item, clock = store(tmp_path)
    assert stage(value)['kind'] == 'encoding'
    assert stage(value)['workload']['profile'] == 'vision'
    assert stage(value)['workload']['reasons'] == ['clip_vision_workload_unmeasured']
    value.safe_stage('prompt', 'encode', 'started')
    with CurrentNodeContext('prompt', 'encode'):
        result = encoder().encode_image(torch.zeros(2, 10, 18, 3))
    live = stage(value)
    assert live['workload']['complete']
    assert live['workload']['resolved']['visionEncode.processed.axis.2']['value'] == 8
    assert live['workload']['resolved']['visionEncode.processed.axis.2']['availableAt'] == 'live'
    assert live['workload']['resolved']['executionPrecision']['value'] == 'torch.float32'
    assert 'visionEncodedPositions' not in live['workload']['measurements']
    clock[0] = 3
    value.safe_stage('prompt', 'encode', 'finished')
    finished = stage(value)
    assert finished['workload']['complete'] and finished['durationSeconds'] == 3
    output = finished['workload']['measurements']['visionEncodedPositions']
    assert output['value'] == result.last_hidden_state.shape[0] * result.last_hidden_state.shape[1]
    assert output['availableAt'] == 'completion'
    reference = finished['workload']['components'][0]
    assert reference['role'] == 'encode:image' and reference['features']['sourceStage']['value'] == 'image'
    assert reference['features']['encodedPositions']['availableAt'] == 'completion'
    assert '/PRIVATE' not in json.dumps(value.page('0', 100, [], []))
    assert not value.page('0', 100, [], [])['observations'][-1]['workload']['evidence']['complete']
    schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(value.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])


def test_repeated_calls_immediately_invalidate_live_comparability_with_bounded_writes(tmp_path):
    value, _, _ = store(tmp_path)
    value.safe_stage('prompt', 'encode', 'started')
    with CurrentNodeContext('prompt', 'encode'):
        encoder().encode_image(torch.zeros(1, 10, 18, 3))
        assert stage(value)['workload']['complete']
        execution_metrics.observe_shape('clip_vision_source', torch.zeros(1, 10, 18, 3), crop=True)
        assert not stage(value)['workload']['complete']
        assert 'multiple_clip_vision_invocations' in stage(value)['workload']['reasons']
        before = value.db.total_changes
        for _ in range(20):
            execution_metrics.observe_shape('clip_vision_source', torch.zeros(1, 10, 18, 3), crop=True)
        assert value.db.total_changes == before


def test_native_forward_failure_keeps_original_error_and_known_input_work(tmp_path):
    value, _, _ = store(tmp_path)
    value.safe_stage('prompt', 'encode', 'started')
    native = encoder()
    def failed(**_):
        raise RuntimeError('fixture native forward failure')
    native.model = failed
    with CurrentNodeContext('prompt', 'encode'), pytest.raises(RuntimeError, match='fixture native forward failure'):
        native.encode_image(torch.zeros(1, 10, 18, 3))
    value.safe_stage('prompt', 'encode', 'failed')
    current = stage(value)
    assert current['phase'] == 'failed'
    assert current['workload']['complete']
    assert current['workload']['measurements']['visionEncodeForwardFailed']['value'] == 1
    assert 'visionEncodedPositions' not in current['workload']['measurements']


def test_cached_encoding_does_not_invent_prior_reference_shapes(tmp_path):
    value, _, _ = store(tmp_path)
    value.safe_cache('prompt', ['encode'])
    assert stage(value)['reused'] is True
    assert not stage(value)['workload']['complete']


def test_failed_metadata_probe_preserves_inference_and_an_explicit_gap(tmp_path):
    value, _, _ = store(tmp_path)
    native = encoder()
    def unavailable():
        raise RuntimeError('fixture metadata unavailable')
    native.patcher.is_dynamic = unavailable
    value.safe_stage('prompt', 'encode', 'started')
    with CurrentNodeContext('prompt', 'encode'):
        result = native.encode_image(torch.zeros(1, 10, 18, 3))
    assert result.last_hidden_state.shape == (1, 64, 3)
    value.safe_stage('prompt', 'encode', 'finished')
    assert stage(value)['phase'] == 'succeeded'
    assert 'clip_vision_metadata_unresolved' in stage(value)['workload']['reasons']


@pytest.mark.parametrize('change', [dict(shape=(1, 8, 8)), dict(shape=(1, 4, 8, 8)), dict(precision=None),
                                   dict(model_type='unknown'), dict(crop='center'), dict(shape=(2, 3, 8, 8)),
                                   dict(crop=False)])
def test_invalid_or_inconsistent_plans_keep_workload_unknown(tmp_path, change):
    value, _, _ = store(tmp_path)
    value.safe_stage('prompt', 'encode', 'started')
    value.safe_metric('prompt', 'encode', 'clip_vision_source', {'shape': (1, 10, 18, 3), 'crop': True})
    plan = dict(shape=(1, 3, 8, 8), crop=True, model_type='clip_vision_model', precision='torch.float32',
                load_device='cpu', offload_device='cpu', memory_strategy='standard')
    plan.update(change)
    value.safe_metric('prompt', 'encode', 'clip_vision_plan', plan)
    assert not stage(value)['workload']['complete']


def test_success_without_observed_forward_and_output_cannot_qualify_as_measured_work(tmp_path):
    value, _, _ = store(tmp_path)
    value.safe_stage('prompt', 'encode', 'started')
    value.safe_metric('prompt', 'encode', 'clip_vision_source', {'shape': (1, 10, 18, 3), 'crop': True})
    value.safe_metric('prompt', 'encode', 'clip_vision_plan', dict(shape=(1, 3, 8, 8), crop=True,
        model_type='clip_vision_model', precision='torch.float32', load_device='cpu', offload_device='cpu', memory_strategy='standard'))
    assert stage(value)['workload']['complete']
    value.safe_stage('prompt', 'encode', 'finished')
    assert 'clip_vision_execution_unresolved' in stage(value)['workload']['reasons']


def test_vision_artifact_lookup_uses_the_native_vision_category(tmp_path, monkeypatch):
    import folder_paths
    weight = tmp_path / 'weight.fixture'
    weight.write_bytes(b'weight')
    calls = []
    def selected(category, name):
        calls.append((category, name))
        return str(weight)
    monkeypatch.setattr(folder_paths, 'get_full_path', selected)
    assert run_timing.selected_file_identity('clip_vision_name', 'fixture')
    assert calls == [('clip_vision', 'fixture')]
