import pytest
import torch

from comfy import execution_metrics
from comfy_extras import nodes_minimax_h3 as h3


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


@pytest.mark.parametrize('length,frames,video_t,audio_t', [(1, 5, 2, 8), (5, 5, 2, 8), (6, 22, 7, 37),
                                                         (22, 22, 7, 37), (23, 39, 12, 65), (124, 124, 37, 207)])
def test_native_temporal_snapping_and_both_allocations_remain_unchanged(monkeypatch, length, frames, video_t, audio_t):
    calls, events = [], []
    zeros, temporal = torch.zeros, h3.temporal_shape
    def device():
        calls.append(('device',))
        return 'cpu'
    def allocate(shape, **kwargs):
        calls.append(('zeros', tuple(shape), kwargs))
        return zeros(shape, **kwargs)
    def schedule(value):
        calls.append(('temporal', value))
        return temporal(value)
    monkeypatch.setattr(h3.comfy.model_management, 'intermediate_device', device)
    monkeypatch.setattr(h3, 'temporal_shape', schedule)
    monkeypatch.setattr(h3.torch, 'zeros', allocate)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    output = h3.EmptyMiniMaxH3LatentAV.execute(65, 47, length).result[0]
    video, audio = output['samples'].unbind()
    assert tuple(video.shape) == (1, 24, video_t, 2, 4)
    assert tuple(audio.shape) == (1, 32, 2, audio_t)
    assert torch.count_nonzero(video) == torch.count_nonzero(audio) == 0
    assert calls == [('temporal', length), ('device',), ('zeros', tuple(video.shape), {'device': 'cpu'}),
                     ('device',), ('zeros', tuple(audio.shape), {'device': 'cpu'})]
    assert [name for name, _ in events] == ['h3_av_temporal_plan', 'h3_av_allocation_plan', 'h3_av_allocate', 'h3_av_allocation_output']
    assert events[0][1] == dict(frames=frames, video_latents=video_t, audio_latents=audio_t, frame_rate=24, audio_latent_rate=40)
    assert events[1][1] == dict(video_shape=tuple(video.shape), audio_shape=tuple(audio.shape))
    assert events[2][1]['succeeded'] and events[2][1]['seconds'] >= 0
    assert events[3][1]['precisions'] == {'video': 'torch.float32', 'audio': 'torch.float32'}
    assert events[3][1]['devices'] == {'video': 'cpu', 'audio': 'cpu'}
    calls.clear()
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('telemetry offline')))
    second = h3.EmptyMiniMaxH3LatentAV.execute(65, 47, length).result[0]
    assert len(calls) == 5
    for actual, expected in zip(second['samples'].unbind(), (video, audio)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('failing_call', [1, 2])
def test_allocation_failure_is_not_retried_or_replaced_by_telemetry(monkeypatch, failing_call):
    calls, events = [], []
    zeros = torch.zeros
    monkeypatch.setattr(h3.comfy.model_management, 'intermediate_device', lambda: 'cpu')
    def allocate(shape, **kwargs):
        calls.append(tuple(shape))
        if len(calls) == failing_call:
            raise RuntimeError('original allocation failure')
        return zeros(shape, **kwargs)
    monkeypatch.setattr(h3.torch, 'zeros', allocate)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(RuntimeError, match='original allocation failure'):
        h3.EmptyMiniMaxH3LatentAV.execute(32, 32, 5)
    assert len(calls) == failing_call
    assert events[-1][0] == 'h3_av_allocate' and events[-1][1]['succeeded'] is False
    assert not any(name == 'h3_av_allocation_output' for name, _ in events)


def test_shared_single_frame_allocator_does_not_invent_a_video_frame_schedule(monkeypatch):
    events = []
    monkeypatch.setattr(h3.comfy.model_management, 'intermediate_device', lambda: 'cpu')
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    output = h3._av_latent(32, 32, 1, 2, batch_size=2)
    assert tuple(output['samples'].unbind()[0].shape) == (2, 24, 1, 2, 2)
    assert not any(name == 'h3_av_temporal_plan' for name, _ in events)
    assert len(events) == 3


def test_output_metadata_failure_preserves_allocated_objects(monkeypatch):
    events = []
    monkeypatch.setattr(h3.comfy.model_management, 'intermediate_device', lambda: 'cpu')
    class TensorWithoutMetadata:
        @property
        def shape(self):
            raise OSError('shape unavailable')
    allocated = [TensorWithoutMetadata(), TensorWithoutMetadata()]
    iterator = iter(allocated)
    monkeypatch.setattr(h3.torch, 'zeros', lambda *_args, **_kwargs: next(iterator))
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    output = h3.EmptyMiniMaxH3LatentAV.execute(32, 32, 5).result[0]
    assert output['samples'].unbind() == allocated
    assert events[-1] == ('tensor_shape_unavailable', {'operation': 'h3_av_allocation_output'})
