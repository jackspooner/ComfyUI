import pytest
import torch

from comfy import execution_metrics
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, PH3ImageVAE


@pytest.fixture(autouse=True)
def reset_observer():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    execution_metrics.set_observer(None)
    torch.set_num_threads(threads)


def vae_backend(tiling=True, tile_size=48, clip_length=17, token_drop=3):
    model = MiniMaxH3VideoVAE(ch=32, ch_mult=(1, 1, 1, 1, 1, 1), num_res_blocks=1,
                            decoder=False, tiling=tiling, tile_size=tile_size, tile_overlap_min=16,
                            clip_length=clip_length, token_drop=token_drop)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, .01)
    return model.eval()


@pytest.mark.parametrize('frames,size,tiling', [(1, 32, True), (22, 32, True), (22, 64, True), (22, 32, False)])
@torch.inference_mode()
def test_actual_native_encoder_calls_values_and_input_plans_are_preserved(monkeypatch, frames, size, tiling):
    model, calls, events = vae_backend(tiling=tiling), [], []
    original_encoder, original_split = model.encoder.forward, model.split_tiles

    def encoder(value):
        calls.append(('encoder', tuple(value.shape), str(value.dtype)))
        return original_encoder(value)

    def split(value):
        calls.append(('split', value))
        return original_split(value)

    monkeypatch.setattr(model.encoder, 'forward', encoder)
    monkeypatch.setattr(model, 'split_tiles', split)
    pixels = torch.linspace(-1, 1, 3 * frames * size * size).reshape(1, 3, frames, size, size)
    before = model.encode(pixels)
    before_calls = list(calls)
    calls.clear()
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with execution_metrics.h3_reference_vae_scope(2):
        after = model.encode(pixels)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert calls == before_calls
    begin = next(facts for event, facts in events if event == 'h3_vae_begin')
    assert begin['native'] is True and begin['reference_index'] == 2
    assert begin['shapes']['pixels'] == tuple(pixels.shape)
    forwards = [facts for event, facts in events if event == 'h3_vae_forward_input']
    assert len(forwards) == sum(call[0] == 'encoder' for call in calls)
    assert len(forwards) == (1 if frames == 1 else 2) * (4 if size == 64 and tiling else 1)
    assert all(facts['binding'] is begin['binding'] for _, facts in events if 'binding' in facts)
    assert [facts for event, facts in events if event == 'h3_vae_output'][0]['shapes']['latents'] == tuple(after.shape)
    assert not any(isinstance(value, torch.Tensor) for _, facts in events for value in facts.values())
    calls.clear()
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('telemetry unavailable')))
    torch.testing.assert_close(model.encode(pixels), before, rtol=0, atol=0)
    assert calls == before_calls


@torch.inference_mode()
def test_original_encode_error_survives_and_scopes_release(monkeypatch):
    model, events = vae_backend(), []
    failure = RuntimeError('original H3 encoder failure')
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(model.encoder, 'forward', fail)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    with pytest.raises(RuntimeError) as raised, execution_metrics.h3_reference_vae_scope(0):
        model.encode(torch.zeros(1, 3, 1, 32, 32))
    assert raised.value is failure
    assert [facts['succeeded'] for event, facts in events if event in ('h3_vae_forward', 'h3_vae_encode')] == [False, False]
    assert not any(event == 'h3_vae_output' for event, _ in events)
    events.clear()
    execution_metrics.observe_h3_vae('h3_vae_clip', {'pixels': torch.zeros(1, 3, 1, 32, 32)})
    assert not events
    monkeypatch.undo()
    model.encode(torch.zeros(1, 3, 1, 32, 32))
    assert next(facts for event, facts in events if event == 'h3_vae_begin')['reference_index'] is None


@torch.inference_mode()
def test_probe_failures_and_custom_variants_do_not_change_native_result(monkeypatch):
    model, events = vae_backend(), []
    pixels = torch.zeros(1, 3, 1, 32, 32)
    before = model.encode(pixels)
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    monkeypatch.setattr(execution_metrics, 'observe_shapes', lambda *_a, **_k: (_ for _ in ()).throw(OSError('probe failure')))
    torch.testing.assert_close(model.encode(pixels), before, rtol=0, atol=0)
    assert any(event == 'h3_vae_metadata_unavailable' for event, _ in events)
    monkeypatch.undo()
    events.clear()
    model.__class__ = PH3ImageVAE
    # Exercise the inherited encoder explicitly, without PH3's sampling override.
    MiniMaxH3VideoVAE.encode(model, pixels)
    assert next(facts for event, facts in events if event == 'h3_vae_begin')['native'] is False
