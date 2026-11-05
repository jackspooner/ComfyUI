import pytest
import torch

from comfy import execution_metrics
from comfy.ldm.minimax.model import PackedLayout


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def layout(cls=PackedLayout):
    keyframes = [{'resolved_frame_index': 0, 'latent': torch.zeros(1, 24, 2, 4, 6),
                  'audio_latent': torch.zeros(1, 32, 2, 3)}]
    refs = [{'kind': 'image', 'latent_h': 4, 'latent_w': 6},
            {'kind': 'video_audio', 'latent_t': 2, 'latent_h': 4, 'latent_w': 6, 'ref_audio_t': 3},
            {'kind': 'audio', 'ref_audio_t': 4}]
    return cls(4, 3, 4, 6, 5, keyframes=keyframes, refs=refs)


def test_actual_packed_roles_and_rows_preserve_native_layout_and_observer_failure():
    execution_metrics.set_observer(None)
    expected = layout()
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = layout()
    event, facts = events[0]
    assert event == 'h3_packed_layout' and facts['native'] is True
    assert facts['signature'] == (4, 3, 4, 6, 5)
    assert facts['segments'] == (('text', 4), ('cond', 12), ('cond_audio', 6), ('ref_img', 6),
                                 ('ref_audio', 6), ('ref_img', 12), ('ref_audio', 8), ('audio', 10), ('video', 18))
    assert facts['tokens'] == actual.seq_len == 82 and facts['segment_count'] == 9
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('offline')))
    failed_observer = layout()
    for value in (actual, failed_observer):
        assert value.signature == expected.signature and value.segments == expected.segments
        for key in ('position_ids', 'img_pos', 'img_update', 'audio_pos', 'audio_update'):
            torch.testing.assert_close(getattr(value, key), getattr(expected, key), rtol=0, atol=0)
    assert not any(isinstance(value, torch.Tensor) for value in facts.values())


def test_layout_subclasses_and_bounded_overflow_are_explicit():
    class CustomLayout(PackedLayout):
        pass
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append(facts))
    layout(CustomLayout)
    assert events[-1]['native'] is False
    refs = [{'kind': 'audio', 'ref_audio_t': 1} for _ in range(130)]
    value = PackedLayout(1, 1, 2, 2, 1, refs=refs)
    facts = events[-1]
    assert facts['segment_count'] == 133 and len(facts['segments']) == 128
    assert value.seq_len == 264
