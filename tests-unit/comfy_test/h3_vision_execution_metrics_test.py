import pytest
import torch

import comfy.ops
from comfy import execution_metrics
from comfy.text_encoders.minimax import MiniMaxQwen3VL, PH3ImageQwen3VL
from comfy.text_encoders.qwen3vl import Qwen3VL, Qwen3VLVisionModel
from comfy.text_encoders.vision_features import VisionFeatures


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def vision_backend(model_class=MiniMaxQwen3VL, cache=False, depth=1):
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    config = dict(hidden_size=8, intermediate_size=12, depth=depth, num_heads=1, patch_size=16,
                  temporal_patch_size=2, in_channels=3, spatial_merge_size=2, num_position_embeddings=16,
                  out_hidden_size=4, deepstack_visual_indexes=[depth - 1])
    model.visual = Qwen3VLVisionModel(config, device='cpu', dtype=torch.float32, ops=comfy.ops.manual_cast).eval()
    for parameter in model.visual.parameters():
        torch.nn.init.constant_(parameter, .01)
    model.dtype, model.num_layers = torch.float32, 2
    model.vision_feature_cache = VisionFeatures(16, 1024 ** 2) if cache else None
    return model


def equal_outputs(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1]['grid'], expected[1]['grid'], rtol=0, atol=0)
    for left, right in zip(actual[1]['deepstack'], expected[1]['deepstack']):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('video,cache', [(False, False), (True, False), (False, True)])
@torch.inference_mode()
def test_real_native_vision_geometry_calls_and_outputs_preserve_execution(monkeypatch, video, cache):
    model = vision_backend(cache=cache)
    image = torch.linspace(0, 1, (2 if video else 1) * 32 * 32 * 3).reshape(2 if video else 1, 32, 32, 3)
    embed = {'type': 'image', 'data': image, 'minimax_video_block': video}
    queries, events = [], []
    original_forward = model.visual.forward
    original_rng, original_equal, original_list = torch.get_rng_state, torch.equal, torch.Tensor.tolist

    def forward(*args, **kwargs):
        queries.append(('forward', tuple(args[0].shape), tuple(args[1].shape)))
        return original_forward(*args, **kwargs)

    def rng():
        queries.append(('rng',))
        return original_rng()

    def equality(*args):
        queries.append(('equal',))
        return original_equal(*args)

    def tolist(value):
        queries.append(('grid_list', tuple(value.shape)))
        return original_list(value)

    monkeypatch.setattr(model.visual, 'forward', forward)
    monkeypatch.setattr(torch, 'get_rng_state', rng)
    monkeypatch.setattr(torch, 'equal', equality)
    monkeypatch.setattr(torch.Tensor, 'tolist', tolist)
    baseline = [model.preprocess_embed(embed, 'cpu') for _ in range(2)]
    baseline_queries = list(queries)
    if cache:
        model.vision_feature_cache.clear()
    queries.clear()
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    observed = [model.preprocess_embed(embed, 'cpu') for _ in range(2)]
    assert queries == baseline_queries
    monkeypatch.undo()
    for actual, expected in zip(observed, baseline):
        equal_outputs(actual, expected)
    geometry = [facts for event, facts in events if event == 'h3_vision_geometry']
    assert len(geometry) == 2
    assert all(facts['grid_h'] == facts['grid_w'] == 4 and facts['height'] == facts['width'] == 64 for facts in geometry)
    assert all(facts['source_frames'] == (2 if video else 1) and facts['shapes']['patches'] == (16, 1536) for facts in geometry)
    decisions = [facts for event, facts in events if event == 'h3_vision_execution']
    assert [facts['reused'] for facts in decisions] == [False, cache]
    assert [facts['cache_eligible'] for facts in decisions] == [cache, cache]
    bindings = [facts['binding'] for event, facts in events if event == 'h3_vision_begin']
    assert len(bindings) == 2 and bindings[0] is not bindings[1] and all(type(value) is object for value in bindings)
    assert not any(isinstance(value, torch.Tensor) for _, facts in events for value in facts.values())
    if cache:
        assert model.vision_feature_cache.entries
        model.vision_feature_cache.clear()
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('telemetry offline')))
    equal_outputs(model.preprocess_embed(embed, 'cpu'), baseline[0])


@torch.inference_mode()
def test_cache_capacity_and_ineligible_decisions_are_observed_at_the_owner():
    model, events = vision_backend(cache=True), []
    model.vision_feature_cache.max_entries = 1
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    for value in (0., .5, .5):
        model.preprocess_embed({'type': 'image', 'data': torch.full((1, 32, 32, 3), value)}, 'cpu')
    decisions = [facts for event, facts in events if event == 'h3_vision_execution']
    assert [facts['reused'] for facts in decisions] == [False, False, False]
    assert model.vision_feature_cache.encodes == 3 and model.vision_feature_cache.hits == 0
    with torch.enable_grad():
        model.preprocess_embed({'type': 'image', 'data': torch.zeros(1, 32, 32, 3)}, 'cpu')
    assert [facts for event, facts in events if event == 'h3_vision_execution'][-1]['cache_eligible'] is False


@torch.inference_mode()
def test_forward_failure_and_scope_cleanup_preserve_native_error(monkeypatch):
    model, events = vision_backend(), []
    failure = RuntimeError('original vision failure')
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(model.visual, 'forward', fail)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(RuntimeError) as raised:
        model.preprocess_embed({'type': 'image', 'data': torch.zeros(1, 32, 32, 3)}, 'cpu')
    assert raised.value is failure
    assert events[-1][0] == 'h3_vision_forward' and events[-1][1]['succeeded'] is False
    monkeypatch.undo()
    events.clear()
    generic = vision_backend(model_class=Qwen3VL)
    generic.preprocess_embed({'type': 'image', 'data': torch.zeros(1, 32, 32, 3)}, 'cpu')
    assert not any(event.startswith('h3_vision_') for event, _ in events)


@torch.inference_mode()
def test_metadata_failure_and_custom_variant_never_change_outputs(monkeypatch):
    model, events = vision_backend(), []
    image = {'type': 'image', 'data': torch.zeros(1, 32, 32, 3)}
    baseline = model.preprocess_embed(image, 'cpu')
    monkeypatch.setattr(execution_metrics, 'observe_shapes', lambda *_a, **_kw: (_ for _ in ()).throw(OSError('probe failed')))
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    equal_outputs(model.preprocess_embed(image, 'cpu'), baseline)
    assert any(event == 'h3_vision_metadata_unavailable' for event, _ in events)
    monkeypatch.undo()
    events.clear()
    variant = vision_backend(model_class=PH3ImageQwen3VL)
    variant.preprocess_embed(image, 'cpu')
    assert [facts for event, facts in events if event == 'h3_vision_begin'][0]['native'] is False


def test_nested_scope_restores_only_scalar_operation_bindings():
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append(facts))
    with execution_metrics.vision_scope():
        execution_metrics.observe_vision('outer')
        with execution_metrics.vision_scope():
            execution_metrics.observe_vision('inner')
        execution_metrics.observe_vision('outer_again')
    execution_metrics.observe_vision('outside')
    assert len(events) == 3 and events[0]['binding'] is events[2]['binding'] and events[0]['binding'] is not events[1]['binding']
