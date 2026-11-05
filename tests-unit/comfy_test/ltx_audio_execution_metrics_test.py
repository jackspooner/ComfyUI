import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import torch
import jsonschema

from app.run_timing import RunTimingStore
from app.run_timing_metrics import attach_runtime_metrics
from comfy import execution_metrics
from comfy.ldm.lightricks.vae.audio_vae import AudioVAE, AudioPreprocessor, CausalityAxis
from comfy_execution.utils import CurrentNodeContext


@pytest.fixture(autouse=True)
def reset_observer():
    execution_metrics.set_observer(None)
    yield
    execution_metrics.set_observer(None)


def encoder():
    value = AudioVAE.__new__(AudioVAE)
    torch.nn.Module.__init__(value)
    value.preprocessor = AudioPreprocessor(8000, 4, 2, 8)
    value.autoencoder = SimpleNamespace(encoder=SimpleNamespace(in_channels=2),
        encode=lambda mel: torch.cat((mel, torch.zeros_like(mel)), dim=1))
    value.normalizer = SimpleNamespace(normalize=lambda latent: latent)
    return value


def decoder(mismatch=False):
    value = AudioVAE.__new__(AudioVAE)
    torch.nn.Module.__init__(value)
    def decode(latent, target_shape):
        shape = (*target_shape[:2], target_shape[2] - int(mismatch), target_shape[3])
        return torch.arange(torch.tensor(shape).prod().item(), dtype=latent.dtype).reshape(shape)
    value.autoencoder = SimpleNamespace(decoder=SimpleNamespace(out_ch=2), mel_bins=16,
        causality_axis=CausalityAxis.NONE, decode=decode)
    value.normalizer = SimpleNamespace(denormalize=lambda latent: latent)
    value.vocoder = lambda mel: mel.mean(dim=2)
    return value


@pytest.mark.parametrize('rate', [4000, 8000])
def test_real_mel_preprocessing_channel_expansion_and_stub_encode_preserve_outputs(rate):
    native = encoder()
    source = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32) / 32
    expected = native.encode(source, sample_rate=rate)
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = native.encode(source, sample_rate=rate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    facts = dict(events)
    resampled = facts['ltx_audio_resampled']
    assert resampled['shape'] == (1, 2, 64 if rate == 4000 else 32)
    assert resampled['sample_rate'] == 8000 and resampled['source_sample_rate'] == rate
    assert (resampled['mel_bins'], resampled['hop_length'], resampled['n_fft']) == (4, 2, 8)
    assert facts['ltx_audio_mel']['shape'] == tuple(actual.shape)
    assert facts['ltx_audio_encode_forward']['succeeded']
    assert facts['ltx_audio_latent_output']['shape'] == tuple(actual.shape)
    assert all(not isinstance(value, torch.Tensor) for _, values in events for value in values.values())
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer offline')))
    torch.testing.assert_close(native.encode(source, sample_rate=rate), expected, rtol=0, atol=0)


def test_decoder_plan_vocoder_input_and_stub_forwards_preserve_output_and_failures():
    native = decoder()
    latent = torch.zeros(1, 8, 3, 4)
    expected = native.decode(latent)
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    torch.testing.assert_close(native.decode(latent), expected, rtol=0, atol=0)
    facts = dict(events)
    assert facts['ltx_audio_decode_plan']['target_shape'] == (1, 2, 12, 16)
    assert facts['ltx_audio_vocoder_input']['shape'] == (1, 2, 16, 12)
    assert facts['ltx_audio_decode_forward']['succeeded'] and facts['ltx_audio_vocoder_forward']['succeeded']
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer offline')))
    torch.testing.assert_close(native.decode(latent), expected, rtol=0, atol=0)
    def failed(*_):
        raise RuntimeError('original vocoder failure')
    native.vocoder = failed
    with pytest.raises(RuntimeError, match='original vocoder failure'):
        native.decode(latent)


def store(tmp_path, kind, audio_source=False):
    value = RunTimingStore(str(tmp_path / 'timing.sqlite'))
    graph = {'stage': {'class_type': kind, 'inputs': {'audio': ['source', 0]} if audio_source else {}}}
    if audio_source:
        graph['source'] = {'class_type': 'LoadAudio', 'inputs': {'audio': 'fixture.wav'}}
    item = (1, 'prompt', graph, {}, ['stage'])
    value.event('accepted', item)
    value.event('started', item)
    attach_runtime_metrics(SimpleNamespace(timing=value, mutex=Lock(), currently_running={0: item}))
    if audio_source:
        value.safe_stage('prompt', 'source', 'started')
        value.safe_metric('prompt', 'source', 'audio_loaded', dict(shape=(1, 1, 32), sample_rate=8000))
        value.safe_stage('prompt', 'source', 'finished')
    value.safe_stage('prompt', 'stage', 'started')
    return value


def stage(value):
    return next(stage for stage in value.page('0', 100, [], [])['observations'][-1]['stages'] if stage['id'] == 'stage')


def test_ltx_encode_collection_covers_actual_mel_work_before_forward_and_retains_reference_role(tmp_path):
    from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode
    value = store(tmp_path, 'LTXVAudioVAEEncode', audio_source=True)
    native = encoder()
    calls = []
    def encode(mel):
        calls.append(tuple(mel.shape))
        current = stage(value)['workload']
        assert current['complete']
        assert current['resolved']['ltxAudio.resampled.channels']['value'] == 2
        assert current['resolved']['ltxAudio.mel.axis.3']['value'] == 4
        assert not current['measurements']
        return torch.cat((mel, torch.zeros_like(mel)), dim=1)
    native.autoencoder.encode = encode
    vae = SimpleNamespace(audio_sample_rate=8000, encode=lambda waveform: native.encode(waveform.movedim(-1, 1), sample_rate=8000))
    with CurrentNodeContext('prompt', 'stage'):
        result = LTXVAudioVAEEncode.execute({'waveform': torch.zeros(1, 1, 32), 'sample_rate': 8000}, vae).result[0]
    value.safe_stage('prompt', 'stage', 'metrics')
    live = stage(value)['workload']
    assert live['measurements']['ltx_audio_latent_output.axis.2']['availableAt'] == 'completion'
    assert 'ltx_audio_latent_output.axis.2' not in live['resolved']
    value.safe_stage('prompt', 'stage', 'finished')
    current = stage(value)['workload']
    assert current['complete'] and len(calls) == 1
    reference = current['components'][0]
    assert reference['role'] == 'stage:audio' and reference['features']['sourceStage']['value'] == 'source'
    assert reference['features']['ltxAudio.resampled.sampleRate']['value'] == 8000
    assert tuple(result['samples'].shape) == calls[0]
    schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(value.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])


@pytest.mark.parametrize('mismatch', [False, True])
def test_collector_verifies_planned_vocoder_work_and_keeps_future_shapes_as_outcomes(tmp_path, mismatch):
    value = store(tmp_path, 'LTXVAudioVAEDecode')
    latent = torch.zeros(1, 8, 3, 4)
    value.safe_metric('prompt', 'stage', 'audio_vae_decode_input',
        dict(shape=tuple(latent.shape), layout='bctf', tiled=False, normalize=False))
    assert not stage(value)['workload']['complete']
    with CurrentNodeContext('prompt', 'stage'):
        decoder(mismatch).decode(latent)
    value.safe_stage('prompt', 'stage', 'finished')
    current = stage(value)
    assert current['workload']['complete'] is (not mismatch)
    assert current['workload']['resolved']['ltxAudio.decode.target.axis.2']['value'] == 12
    assert current['workload']['measurements']['ltx_audio_vocoder_input.axis.3']['availableAt'] == 'completion'
    assert 'ltx_audio_vocoder_input.axis.3' not in current['workload']['resolved']
    if mismatch:
        assert 'ltx_audio_vocoder_plan_mismatch' in current['workload']['reasons']
    else:
        assert 'ltx_audio_decode_workload_unmeasured' not in current['workload']['reasons']
    assert 'audio_output_unmeasured' in current['reasons']


def test_missing_internal_ltx_preprocessing_stays_incomplete(tmp_path):
    value = store(tmp_path, 'LTXVAudioVAEEncode')
    value.safe_metric('prompt', 'stage', 'audio_vae_encode_input',
        dict(shape=(1, 2, 32), sample_rate=44100, source_sample_rate=8000))
    assert 'audio_processed_workload_unmeasured' not in stage(value)['workload']['reasons']
    assert 'ltx_audio_encode_workload_unmeasured' in stage(value)['workload']['reasons']
    assert not stage(value)['workload']['complete']
