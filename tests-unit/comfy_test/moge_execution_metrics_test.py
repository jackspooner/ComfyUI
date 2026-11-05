from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import numpy as np

from comfy import execution_metrics
from comfy.ldm.moge import model, modules
from comfy_extras import nodes_moge


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


@pytest.mark.parametrize('version,requested,effective', [('v1', 8, 0), ('v2', 8, 0), ('v3', 2, 2)])
def test_infer_observes_actual_token_precision_and_load_boundary_without_retry(monkeypatch, version, requested, effective):
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    instance = model.MoGeModel.__new__(model.MoGeModel)
    instance.load_device = torch.device('cpu')
    instance.dtype = torch.bfloat16  # The real path chooses fp16 compute.
    instance.version = version
    instance.num_tokens_range = (1200, 3600)
    instance.patcher = SimpleNamespace(offload_device=torch.device('cpu'), is_dynamic=lambda: False)
    failure = RuntimeError('original-forward-failure')
    instance.model = SimpleNamespace(forward=Mock(side_effect=failure))
    load = Mock()
    monkeypatch.setattr(model.comfy.model_management, 'load_model_gpu', load)
    image = torch.zeros(1, 3, 4, 6)
    with pytest.raises(RuntimeError, match='original-forward-failure') as caught:
        instance.infer(image, resolution_level=6, refine_steps=requested)
    assert caught.value is failure
    assert instance.model.forward.call_count == load.call_count == 1
    event = next(facts for event, facts in events if event == 'moge_model_plan')
    assert event['shape'] == (1, 3, 4, 6) and event['num_tokens'] == 2800
    assert event['precision'] == 'torch.float16' and event['refine_steps'] == effective
    actual = instance.model.forward.call_args.args[0]
    assert str(actual.dtype) == event['activation_precision']
    assert image.dtype == torch.float32
    assert next(facts for event, facts in events if event == 'moge_load')['succeeded'] is True
    assert next(facts for event, facts in events if event == 'moge_forward')['succeeded'] is False
    assert not any(isinstance(value, torch.Tensor) for _, facts in events for value in facts.values())


@pytest.mark.parametrize('variant', ['v1', 'v2plus'])
def test_processed_shapes_are_read_from_the_tensor_actually_sent_to_the_backbone(variant):
    events, shapes = [], []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    def stop(image, *args, **kwargs):
        shapes.append(tuple(image.shape))
        raise RuntimeError('stop-before-model')
    cls = model.MoGeModelV1 if variant == 'v1' else modules.DINOv2Encoder
    instance = cls.__new__(cls)
    torch.nn.Module.__init__(instance)
    instance.image_mean = torch.zeros(1, 3, 1, 1)
    instance.image_std = torch.ones(1, 3, 1, 1)
    instance.intermediate_layers = 1
    instance.backbone = SimpleNamespace(encoder=SimpleNamespace(layer=[None]), get_intermediate_layers=stop)
    with pytest.raises(RuntimeError, match='stop-before-model'):
        if variant == 'v1':
            instance.forward(torch.zeros(1, 3, 8, 12), num_tokens=4)
        else:
            instance.forward(torch.zeros(1, 3, 8, 12), 2, 3)
    facts = next(facts for event, facts in events if event == 'moge_processed_input')
    assert facts['shape'] == shapes[0]
    assert facts['shape'] == ((1, 3, 14, 28) if variant == 'v1' else (1, 3, 28, 42))
    if variant == 'v1':
        assert facts['head_shape'] == (1, 3, 22, 34)


def test_refiner_uses_existing_spatial_metadata_without_reading_coordinates_into_telemetry():
    events, calls = [], []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    instance = model.MoGeModelV3.__new__(model.MoGeModelV3)
    torch.nn.Module.__init__(instance)
    def refine(features, coords, spatial, conditioning):
        calls.append((tuple(coords.shape), spatial))
        return torch.zeros(features.shape[0])
    instance.refiner = refine
    result = instance._refine_logz(torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 2, 3))
    assert result.shape == (1, 2, 3) and calls == [((6, 4), (2, 3, 1))]
    facts = next(facts for event, facts in events if event == 'moge_refinement_plan')
    assert facts == {'points': 6, 'spatial': (2, 3, 1), 'conditioning_shape': (1, 2, 2, 3)}


def test_node_plan_preserves_chunking_and_original_results_when_observer_fails():
    events, batches = [], []
    def observe(event, facts):
        events.append((event, facts))
        raise OSError('telemetry-write-failure')
    execution_metrics.set_observer(observe)
    def infer(image, **kwargs):
        batches.append(image.shape[0])
        return {'depth': torch.ones(image.shape[0], 4, 6), 'mask': torch.ones(image.shape[0], 4, 6, dtype=torch.bool)}
    output = nodes_moge.MoGeInference.execute(SimpleNamespace(infer=infer), torch.zeros(5, 4, 6, 4),
        resolution_level=6, fov_x_degrees=0, batch_size=2, force_projection=True, apply_mask=True, refine_steps=3)
    assert batches == [2, 2, 1] and output.result[0]['depth'].shape == (5, 4, 6)
    facts = next(facts for event, facts in events if event == 'moge_node_plan')
    assert facts == {'shape': (5, 3, 4, 6), 'operation': 'perspective', 'views': 5, 'batch_size': 2}


def test_panorama_plan_uses_actual_splits_and_effective_merge_canvas(monkeypatch):
    events, batches = [], []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    monkeypatch.setattr(nodes_moge.comfy.model_management, 'load_model_gpu', lambda _: None)
    monkeypatch.setattr(nodes_moge, 'get_panorama_cameras', lambda: ([np.eye(4)] * 7, [np.eye(3)] * 7))
    monkeypatch.setattr(nodes_moge, 'split_panorama_image', lambda *args: torch.zeros(7, 3, 2, 2))
    monkeypatch.setattr(nodes_moge, 'merge_panorama_depth', lambda w, h, *args, **kwargs:
                        (np.ones((h, w), dtype=np.float32), np.ones((h, w), dtype=bool)))
    def infer(image, **kwargs):
        batches.append(image.shape[0])
        return {'points': torch.ones(image.shape[0], 2, 2, 3), 'mask': torch.ones(image.shape[0], 2, 2, dtype=torch.bool)}
    instance = SimpleNamespace(infer=infer, patcher=object(), load_device=torch.device('cpu'), dtype=torch.float32)
    output = nodes_moge.MoGePanoramaInference.execute(instance, torch.zeros(1, 4, 8, 3), 6, 2, 8, 3, 3)
    assert batches == [3, 3, 1] and output.result[0]['depth'].shape == (1, 4, 8)
    facts = next(facts for event, facts in events if event == 'moge_node_plan')
    assert facts == {'shape': (1, 3, 4, 8), 'operation': 'panorama', 'views': 7, 'batch_size': 3, 'merge_shape': (32, 32)}
