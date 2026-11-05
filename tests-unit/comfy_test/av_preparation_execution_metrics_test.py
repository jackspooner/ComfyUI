from types import SimpleNamespace

import pytest
import torch

from comfy import execution_metrics
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


@pytest.mark.parametrize('layout,axis', [('bctf', 2), ('bcat', 3)])
@pytest.mark.parametrize('length', [2, 4, 6])
@pytest.mark.parametrize('mask_role', [None, 'video', 'audio'])
def test_concat_fitting_and_masks_preserve_outputs_with_observers(layout, axis, length, mask_role):
    video = torch.zeros(1, 4, 2, 2, 2)
    audio_shape = [1, 2, 4, 2] if layout == 'bctf' else [1, 2, 2, 4]
    audio_shape[axis] = length
    source_audio = torch.arange(torch.tensor(audio_shape).prod()).reshape(audio_shape).float()
    target = torch.zeros((1, 2, 4, 2) if layout == 'bctf' else (1, 2, 2, 4))
    video_input = {'samples': NestedTensor((video, target)), 'video_metadata': 'unchanged'}
    audio_input = {'samples': source_audio, 'audio_metadata': 'unchanged'}
    if mask_role == 'video':
        video_input['noise_mask'] = NestedTensor((torch.ones_like(video), torch.ones_like(target)))
    elif mask_role == 'audio':
        audio_input['noise_mask'] = torch.ones_like(source_audio)
    baseline = LTXVConcatAVLatent.execute(video_input, audio_input).result[0]
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    observed = LTXVConcatAVLatent.execute(video_input, audio_input).result[0]
    assert observed['samples'].unbind()[0] is video
    assert tuple(observed['samples'].unbind()[1].shape) == tuple(target.shape)
    assert len(events) == 3 and [name for name, _ in events] == ['av_concat_plan', 'av_concat_fit', 'av_concat_output']
    assert events[0][1]['shapes']['audio'] == tuple(source_audio.shape)
    assert events[0][1]['shapes']['target_audio'] == tuple(target.shape)
    assert events[0][1]['precisions']['audio'] == 'torch.float32'
    assert events[0][1]['devices']['audio'] == 'cpu'
    assert events[1][1]['succeeded'] and events[1][1]['seconds'] >= 0
    for key in ('samples', 'noise_mask') if mask_role else ('samples',):
        for expected, actual in zip(baseline[key].unbind(), observed[key].unbind()):
            torch.testing.assert_close(actual, expected)
    assert observed['video_metadata'] == observed['audio_metadata'] == 'unchanged'
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer unavailable')))
    failed_observer = LTXVConcatAVLatent.execute(video_input, audio_input).result[0]
    for expected, actual in zip(baseline['samples'].unbind(), failed_observer['samples'].unbind()):
        torch.testing.assert_close(actual, expected)


def test_plain_concat_and_separate_keep_tensor_identity_and_original_metadata():
    video, audio = torch.zeros(1, 4, 1, 2, 2), torch.zeros(1, 8, 3, 2)
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    joined = LTXVConcatAVLatent.execute({'samples': video, 'kind': 'video'}, {'samples': audio, 'kind': 'audio'}).result[0]
    assert events[0][1]['input_streams'] == 1 and 'target_audio' not in events[0][1]['shapes']
    assert [name for name, _ in events] == ['av_concat_plan', 'av_concat_output']
    events.clear()
    separated = LTXVSeparateAVLatent.execute(joined).result
    assert separated[0]['samples'] is video and separated[1]['samples'] is audio
    assert separated[0]['kind'] == separated[1]['kind'] == 'audio'
    assert [name for name, _ in events] == ['av_separate_plan', 'av_separate_output']
    assert events[0][1]['input_streams'] == 2
    assert 'samples' in joined and joined['samples'].is_nested


def test_native_fitting_failure_stays_the_same_and_telemetry_does_not_retry():
    video, target, audio = torch.zeros(1, 4, 1, 2, 2), torch.zeros(1, 8, 4, 2), torch.zeros(1, 8, 6, 3)
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(ValueError, match='cannot be fitted'):
        LTXVConcatAVLatent.execute({'samples': NestedTensor((video, target))}, {'samples': audio})
    assert [name for name, _ in events] == ['av_concat_plan', 'av_concat_fit']
    assert events[1][1]['succeeded'] is False
    assert target.shape == (1, 8, 4, 2) and audio.shape == (1, 8, 6, 3)


def test_shape_probe_failure_and_disabled_telemetry_preserve_node_behavior():
    class UnavailableShape:
        is_nested = False
        @property
        def shape(self):
            raise RuntimeError('metadata unavailable')
    video, audio = UnavailableShape(), UnavailableShape()
    plain = LTXVConcatAVLatent.execute({'samples': video}, {'samples': audio}).result[0]
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    joined = LTXVConcatAVLatent.execute({'samples': video}, {'samples': audio}).result[0]
    assert joined['samples'].unbind() == plain['samples'].unbind() == [video, audio]
    assert events == [('tensor_shape_unavailable', {'operation': 'av_concat_plan'}),
                      ('tensor_shape_unavailable', {'operation': 'av_concat_output'})]
    events.clear()
    execution_metrics.observe_shapes('av_concat_plan', {str(index): SimpleNamespace(shape=()) for index in range(9)})
    assert events == [('tensor_shape_unavailable', {'operation': 'av_concat_plan'})]


def test_stream_count_probe_failure_preserves_indexable_native_inputs():
    video, audio = torch.zeros(1, 4, 1, 2, 2), torch.zeros(1, 8, 3, 2)
    class IndexableStreams:
        def __getitem__(self, index):
            return (video, audio)[index]
        def __len__(self):
            raise OSError('count unavailable')
    joint = SimpleNamespace(is_nested=True, unbind=lambda: IndexableStreams())
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    joined = LTXVConcatAVLatent.execute({'samples': joint}, {'samples': audio}).result[0]
    assert joined['samples'].unbind()[0] is video
    assert joined['samples'].unbind()[1] is audio
    assert events[0] == ('tensor_shape_unavailable', {'operation': 'av_concat_plan'})
    events.clear()
    separated = LTXVSeparateAVLatent.execute({'samples': joint}).result
    assert separated[0]['samples'] is video and separated[1]['samples'] is audio
    assert events[0] == ('tensor_shape_unavailable', {'operation': 'av_separate_plan'})
