from types import SimpleNamespace

import pytest
import torch

from comfy import execution_metrics, sd
from comfy.nested_tensor import NestedTensor
from comfy_extras import nodes_audio, nodes_minimax_h3


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


def test_encode_observes_processed_rate_and_shape_with_one_resample_and_vae_call(monkeypatch):
    events, calls = [], []
    source = torch.zeros(2, 2, 12)
    processed = torch.ones(2, 2, 24)
    latent = torch.zeros(2, 64, 3)
    def resample(waveform, source_rate, target_rate):
        calls.append(('resample', source_rate, target_rate))
        assert waveform is source
        return processed
    def encode(waveform):
        calls.append(('encode',))
        torch.testing.assert_close(waveform, processed.movedim(1, -1))
        return latent
    monkeypatch.setattr(nodes_audio.comfy.audio, 'resample', resample)
    vae = SimpleNamespace(audio_sample_rate=24000, encode=encode)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = nodes_audio.VAEEncodeAudio.execute(vae, {'waveform': source, 'sample_rate': 12000})
    assert result.result[0]['samples'] is latent
    assert calls == [('resample', 12000, 24000), ('encode',)]
    assert events == [('audio_vae_encode_input', {'shape': (2, 2, 24), 'sample_rate': 24000, 'source_sample_rate': 12000})]


@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
def test_decode_preserves_normalization_rate_override_and_tiled_arguments(tiled, normalize):
    events, calls = [], []
    latent = torch.ones(1, 64, 3)
    raw = torch.tensor([[[-.8, .8], [.8, -.8]]])
    def decode(samples, **kwargs):
        assert samples is latent
        calls.append(kwargs)
        return raw.clone()
    vae = SimpleNamespace(decode=decode, decode_tiled=decode, audio_sample_rate=24000, audio_sample_rate_output=44100)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = nodes_audio.vae_decode_audio(vae, {'samples': latent, 'sample_rate': 16000},
                                           tile=8 if tiled else None, overlap=2 if tiled else None, normalize=normalize)
    expected = raw.movedim(-1, 1)
    if normalize:
        expected = expected / (torch.std(expected, dim=[1, 2], keepdim=True) * 5).clamp(min=1)
    torch.testing.assert_close(result['waveform'], expected)
    assert result['sample_rate'] == 16000
    assert calls == [{'tile_x': 8, 'tile_y': 8, 'overlap': 2} if tiled else {}]
    assert events == [
        ('audio_vae_decode_input', {'shape': (1, 64, 3), 'tiled': tiled, 'tile_size': 8 if tiled else None,
                                   'overlap': 2 if tiled else None, 'normalize': normalize}),
        ('audio_decoded', {'shape': (1, 2, 2), 'sample_rate': 16000})]


def test_decode_observes_only_the_selected_audio_stream():
    audio = torch.ones(1, 32, 2, 3)
    events = []
    def decode(samples):
        assert samples is audio
        return torch.zeros(1, 12, 2)
    vae = SimpleNamespace(decode=decode, audio_sample_rate=32000)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = nodes_audio.vae_decode_audio(vae, {'samples': NestedTensor((torch.zeros(1, 24, 1, 1, 1), audio))})
    assert result['waveform'].shape == (1, 2, 12)
    assert events[0][1]['shape'] == (1, 32, 2, 3)


def test_h3_reference_audio_observes_only_the_encoded_first_batch_item():
    calls, events = [], []
    latent = torch.ones(1, 32, 2, 3)
    def encode(waveform):
        calls.append(waveform.shape)
        return latent
    vae = SimpleNamespace(encode=encode, audio_sample_rate=32000)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result, length = nodes_minimax_h3._encode_ref_audio(vae, {'waveform': torch.zeros(2, 2, 12), 'sample_rate': 32000})
    assert result is latent and length == 3
    assert calls == [torch.Size([1, 12, 2])]
    assert events == [('audio_vae_encode_input', {'shape': (1, 2, 12), 'sample_rate': 32000, 'source_sample_rate': 32000})]


def test_loader_metadata_excludes_filename_and_waveform_values(monkeypatch):
    events = []
    waveform = torch.ones(2, 12)
    monkeypatch.setattr(nodes_audio.folder_paths, 'get_annotated_filepath', lambda _: '/PRIVATE/source.wav')
    monkeypatch.setattr(nodes_audio, 'load', lambda _: (waveform, 24000))
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = nodes_audio.LoadAudio.execute('private filename').result[0]
    torch.testing.assert_close(result['waveform'], waveform.unsqueeze(0))
    assert events == [('audio_loaded', {'shape': (1, 2, 12), 'sample_rate': 24000})]


def test_observer_failure_preserves_native_value_and_exception():
    calls = []
    def broken(*_):
        raise OSError('telemetry unavailable')
    execution_metrics.set_observer(broken)
    audio = {'waveform': torch.zeros(1, 1, 12), 'sample_rate': 24000}
    vae = SimpleNamespace(audio_sample_rate=24000, encode=lambda waveform: calls.append(1) or waveform)
    result = nodes_audio.VAEEncodeAudio.execute(vae, audio).result[0]['samples']
    torch.testing.assert_close(result, audio['waveform'].movedim(1, -1))
    assert calls == [1]
    def failed(_):
        raise ValueError('native failure')
    vae.encode = failed
    with pytest.raises(ValueError, match='native failure'):
        nodes_audio.VAEEncodeAudio.execute(vae, audio)


def test_native_vae_audio_oom_fallback_keeps_existing_call_order_and_result(monkeypatch):
    calls, events = [], []
    vae = sd.VAE.__new__(sd.VAE)
    def fail(samples, **kwargs):
        calls.append('regular')
        raise torch.cuda.OutOfMemoryError('fixture OOM')
    vae.first_stage_model = SimpleNamespace(decode=fail)
    vae.throw_exception_if_invalid = lambda: None
    vae.device = vae.output_device = torch.device('cpu')
    vae.vae_dtype = torch.float32
    vae.vae_output_dtype = lambda: torch.float32
    vae.memory_used_decode = lambda *_: 1
    vae.patcher = SimpleNamespace(get_free_memory=lambda *_: 100)
    vae.disable_offload, vae.latent_dim, vae.extra_1d_channel = False, 1, None
    vae.decode_tiled_1d = lambda samples: calls.append('tiled') or torch.ones(1, 2, 12)
    monkeypatch.setattr(sd.model_management, 'load_models_gpu', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sd.model_management, 'soft_empty_cache', lambda: calls.append('empty_cache'))
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    result = vae.decode(torch.zeros(1, 64, 3))
    torch.testing.assert_close(result, torch.ones(1, 12, 2))
    assert calls == ['regular', 'empty_cache', 'tiled']
    assert events == [('vae_decode_tiled_fallback', {})]
