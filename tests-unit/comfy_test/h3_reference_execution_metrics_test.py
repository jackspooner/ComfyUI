import pytest
import torch

from comfy import execution_metrics
from comfy_extras import nodes_minimax_h3 as h3


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def backends(monkeypatch, calls, failure=None):
    resize = h3._resize
    monkeypatch.setattr(h3.comfy.model_management, 'intermediate_device', lambda: 'cpu')

    def observed_resize(image, width, height, crop):
        calls.append(('resize', tuple(image.shape), width, height, crop))
        return resize(image, width, height, crop)

    def resample(waveform, source, target):
        calls.append(('resample', tuple(waveform.shape), source, target))
        return waveform.repeat_interleave(target // source, dim=-1)

    class Clip:
        def tokenize(self, prompt, **kwargs):
            calls.append(('tokenize', prompt, tuple((item['type'], tuple(item['data'].shape) if 'data' in item else None,
                         tuple(item.get('timestamps', []))) for item in kwargs['minimax_ref_items'])))
            return 'opaque-token-binding'

        def encode_from_tokens_scheduled(self, tokens):
            calls.append(('text_encode', tokens))
            return [[torch.zeros(1, 3, 4), {}]]

    class VAE:
        def encode(self, pixels):
            calls.append(('visual_encode', tuple(pixels.shape)))
            if failure is not None:
                raise failure
            return torch.zeros(1, 24, h3.video_latent_t(pixels.shape[0]), pixels.shape[1] // 16, pixels.shape[2] // 16)

    class AudioVAE:
        audio_sample_rate = 32000

        def encode(self, waveform):
            calls.append(('audio_encode', tuple(waveform.shape)))
            return torch.zeros(1, 32, 2, 4)

    monkeypatch.setattr(h3, '_resize', observed_resize)
    monkeypatch.setattr(h3.comfy.audio, 'resample', resample)
    return Clip(), VAE(), AudioVAE()


def execute(clip, vae, audio_vae):
    audio = {'waveform': torch.zeros(2, 2, 16), 'sample_rate': 8000}
    return h3.MiniMaxH3ReferenceToVideo.execute(clip, 'private prompt', 32, 32, 6, vae=vae, audio_vae=audio_vae,
        ref_images={'ref_image_0': None, 'ref_image_1': torch.zeros(3, 64, 32, 4)},
        ref_videos={'ref_video_2': torch.zeros(23, 32, 32, 3)}, ref_video_audios={'ref_video_audio_2': audio},
        ref_audios={'ref_audio_1': audio}).result


@pytest.mark.parametrize('visual,audio', [(True, True), (True, False), (False, True), (False, False)])
def test_reference_work_order_selection_and_outputs_preserve_execution(monkeypatch, visual, audio):
    calls, events = [], []
    clip, vae, audio_vae = backends(monkeypatch, calls)
    baseline = execute(clip, vae if visual else None, audio_vae if audio else None)
    baseline_calls = list(calls)
    calls.clear()
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    observed = execute(clip, vae if visual else None, audio_vae if audio else None)
    assert calls == baseline_calls
    assert [call[0] for call in calls] == (['resize'] + (['visual_encode'] if visual else [])
        + ['resize'] + (['visual_encode'] if visual else []) + (['resample', 'audio_encode'] if visual and audio else [])
        + (['resample', 'audio_encode'] if audio else []) + ['tokenize', 'text_encode'])
    for actual, expected in zip(observed[1]['samples'].unbind(), baseline[1]['samples'].unbind()):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(observed[0][0][0], baseline[0][0][0])
    blocks = observed[0][0][1].get('minimax_refs', [])
    assert [block['kind'] for block in blocks] == (['image', 'video_audio' if audio else 'video'] if visual else []) + (['audio'] if audio else [])
    sources = [facts for event, facts in events if event == 'h3_reference_source']
    assert [(facts['index'], facts['role'], facts['slot']) for facts in sources] == [
        (0, 'image', 'ref_images.ref_image_1'), (1, 'video', 'ref_videos.ref_video_2'), (2, 'audio', 'ref_audios.ref_audio_1')]
    assert sources[0]['shapes']['source'] == (3, 64, 32, 4)
    assert sources[1]['shapes']['source'][0] == 23 and sources[1]['frame_limit'] == 22
    assert sources[1]['audio_encode'] is (visual and audio)
    processed = [facts for event, facts in events if event == 'h3_reference_processed']
    assert processed[0]['shapes'] == {'visual': (1, 32, 32, 3), 'tokenizer': (1, 32, 32, 3)}
    assert processed[1]['shapes'] == {'visual': (22, 32, 32, 3), 'tokenizer': (2, 32, 32, 3)}
    summary = next(facts for event, facts in events if event == 'h3_reference_prepared')
    assert summary == {'references': 3, 'tokenizer_items': 4, 'blocks': 2 * visual + audio}
    assert 'private prompt' not in repr(events) and 'opaque-token-binding' not in repr(events)
    assert not any(isinstance(value, torch.Tensor) for _, facts in events for value in facts.values())
    calls.clear()
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('timing unavailable')))
    execute(clip, vae if visual else None, audio_vae if audio else None)
    assert calls == baseline_calls


def test_failed_encoding_keeps_original_exception_and_does_not_retry(monkeypatch):
    failure = RuntimeError('native VAE failure')
    calls, events = [], []
    clip, vae, audio_vae = backends(monkeypatch, calls, failure)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(RuntimeError) as raised:
        execute(clip, vae, audio_vae)
    assert raised.value is failure
    assert [call[0] for call in calls] == ['resize', 'visual_encode']
    assert events[-1][0] == 'h3_reference_visual_encode' and events[-1][1]['succeeded'] is False
    assert not any(event in ('h3_reference_output', 'h3_reference_prepared') for event, _ in events)


def test_short_video_retains_native_error_after_full_resize(monkeypatch):
    calls, events = [], []
    clip, vae, _ = backends(monkeypatch, calls)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(ValueError, match='reference videos need at least 5 frames'):
        h3.MiniMaxH3ReferenceToVideo.execute(clip, '', 32, 32, 6, vae=vae,
            ref_videos={'ref_video_0': torch.zeros(4, 32, 32, 3)})
    assert calls == [('resize', (4, 32, 32, 3), 32, 32, 'disabled')]
    assert not any(event in ('h3_reference_processed', 'h3_reference_prepared') for event, _ in events)


def test_unconsumed_audio_and_nonstandard_image_keys_cannot_break_execution(monkeypatch):
    calls, events = [], []
    clip, _, _ = backends(monkeypatch, calls)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = h3.MiniMaxH3ReferenceToVideo.execute(clip, '', 32, 32, 6,
        ref_images={42: torch.zeros(1, 32, 32, 3)}, ref_audios={'ref_audio_0': {}})
    assert result.result[0] and calls[-1] == ('text_encode', 'opaque-token-binding')
    assert sum(event == 'h3_reference_metadata_unavailable' for event, _ in events) == 3


def test_metadata_probe_failure_preserves_reference_results(monkeypatch):
    calls, events = [], []
    clip, vae, audio_vae = backends(monkeypatch, calls)
    original = execution_metrics.observe_shapes
    def broken(event, *args, **kwargs):
        if event.startswith('h3_reference_'):
            raise OSError('metadata unavailable')
        return original(event, *args, **kwargs)
    monkeypatch.setattr(execution_metrics, 'observe_shapes', broken)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = execute(clip, vae, audio_vae)
    assert len(result[0][0][1]['minimax_refs']) == 3
    assert any(event == 'h3_reference_metadata_unavailable' for event, _ in events)
