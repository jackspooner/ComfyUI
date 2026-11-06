from types import SimpleNamespace

import pytest
import torch

import comfy.ops
from comfy import execution_metrics, model_management
from comfy.sd import VAE
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE, Encoder, AttnProjection
from comfy_extras import nodes_minimax_h3 as h3


@pytest.fixture(autouse=True)
def cleanup():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    execution_metrics.set_observer(None)
    torch.set_num_threads(threads)


def audio_backend():
    model = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    torch.nn.Module.__init__(model)
    model.sample_rate, model.hop_length = 32000, 800
    model.encoder = Encoder(2, (2, 4, 4, 5, 5), 32)
    model.pre_block = AttnProjection(32, 32, 8)
    model.mean_proj = comfy.ops.disable_weight_init.Conv1d(32, 32, 1)
    model.register_buffer('latents_mean', torch.zeros(32))
    model.register_buffer('latents_std', torch.ones(32))
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, .01)
    model.pre_block.attn.zero_k_bias.zero_()
    return model.eval()


def wrapper(crop=True):
    result = VAE.__new__(VAE)
    result.first_stage_model = audio_backend()
    result.device = result.output_device = torch.device('cpu')
    result.vae_dtype, result.latent_dim, result.output_channels = torch.float32, 2, 2
    result.pad_channel_value, result.crop_input = 'replicate', crop
    result.not_video = result.disable_offload = False
    result.extra_1d_channel = result.format_encoded = None
    result.downscale_ratio = 800
    result.audio_sample_rate = 32000
    result.process_input = lambda pixels: pixels
    result.memory_used_encode = lambda shape, dtype: 1
    result.patcher = SimpleNamespace(get_free_memory=lambda device: 5000)
    return result


@pytest.mark.parametrize('channels,length,batch', [(1, 1600, 1), (2, 1601, 1), (2, 800, 2)])
@torch.inference_mode()
def test_native_padding_encoder_attention_and_mean_outputs_preserve_execution(monkeypatch, channels, length, batch):
    model, calls, events = audio_backend(), [], []
    for name in ('encoder', 'pre_block', 'mean_proj'):
        module, original = getattr(model, name), getattr(model, name).forward
        def forward(value, original=original, name=name):
            calls.append((name, tuple(value.shape)))
            return original(value)
        monkeypatch.setattr(module, 'forward', forward)
    waveform = torch.linspace(-.5, .5, batch * channels * length).reshape(batch, channels, length)
    before = model.encode(waveform)
    before_calls = list(calls)
    calls.clear()
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    after = model.encode(waveform)
    assert calls == before_calls
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    plan = next(facts for event, facts in events if event == 'h3_audio_plan')
    assert plan['right_pad'] == (-length) % 800 and plan['native'] is True
    assert next(facts for event, facts in events if event == 'h3_audio_encoder_input')['shapes']['input'] == (batch * channels, 1, length + plan['right_pad'])
    assert not any(isinstance(value, torch.Tensor) for _, facts in events for value in facts.values())
    calls.clear()
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer unavailable')))
    torch.testing.assert_close(model.encode(waveform), before, rtol=0, atol=0)
    assert calls == before_calls


@pytest.mark.parametrize('rate,crop,channels', [(16000, True, 1), (32000, True, 2), (32000, False, 2)])
@torch.inference_mode()
def test_real_resample_wrapper_crop_and_channel_work_remain_unchanged(monkeypatch, rate, crop, channels):
    native, calls, events = wrapper(crop), [], []
    monkeypatch.setattr(model_management, 'load_models_gpu', lambda *_a, **_k: None)
    original = h3.comfy.audio.resample
    def resample(value, source, target):
        calls.append((tuple(value.shape), source, target))
        return original(value, source, target)
    monkeypatch.setattr(h3.comfy.audio, 'resample', resample)
    audio = {'waveform': torch.linspace(-.5, .5, 2 * channels * 801).reshape(2, channels, 801), 'sample_rate': rate}
    before, _ = h3._encode_ref_audio(native, audio)
    before_calls = list(calls)
    calls.clear()
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with execution_metrics.h3_reference_vae_scope(3):
        after, _ = h3._encode_ref_audio(native, audio)
    assert calls == before_calls
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    source = next(facts for event, facts in events if event == 'h3_audio_source')
    plan = next(facts for event, facts in events if event == 'h3_audio_plan')
    assert source['shapes']['waveform'][0] == 2 and plan['shapes']['waveform'][0:2] == (1, 2)
    assert plan['binding'] is source['binding'] and plan['reference_index'] == 3
    assert plan['right_pad'] == (799 if not crop else 0)


@pytest.mark.parametrize('phase', ['encoder', 'pre_block', 'mean_proj'])
@torch.inference_mode()
def test_original_error_identity_and_scope_cleanup(monkeypatch, phase):
    model, events = audio_backend(), []
    failure = RuntimeError('original audio encode failure')
    def fail(*_args, **_kwargs):
        raise failure
    monkeypatch.setattr(getattr(model, phase), 'forward', fail)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(RuntimeError) as caught:
        model.encode(torch.zeros(1, 2, 800))
    assert caught.value is failure
    assert not any(event == 'h3_audio_output' for event, _ in events)
    assert events[-1][1]['succeeded'] is False
    events.clear()
    execution_metrics.observe_h3_audio('outside')
    assert not events


@torch.inference_mode()
def test_probe_failures_and_custom_encoder_stay_observable_without_affecting_outputs(monkeypatch):
    model, events = audio_backend(), []
    waveform = torch.zeros(1, 2, 800)
    before = model.encode(waveform)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    monkeypatch.setattr(execution_metrics, 'observe_shapes', lambda *_a, **_kw: (_ for _ in ()).throw(OSError('probe failed')))
    torch.testing.assert_close(model.encode(waveform), before, rtol=0, atol=0)
    assert any(event == 'h3_audio_metadata_unavailable' for event, _ in events)
    monkeypatch.undo()
    class CustomEncoder(Encoder):
        pass
    model.encoder.__class__ = CustomEncoder
    events.clear()
    torch.testing.assert_close(model.encode(waveform), before, rtol=0, atol=0)
    assert next(facts for event, facts in events if event == 'h3_audio_plan')['native'] is False
