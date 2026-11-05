from types import SimpleNamespace

import pytest
import torch

import nodes
from comfy import execution_metrics, sd
from comfy.text_encoders.vision_features import VisionFeatures


@pytest.fixture(autouse=True)
def reset_observer():
    yield
    execution_metrics.set_observer(None)


@pytest.mark.parametrize('internal_chunks', [False, True])
def test_vae_reports_actual_cropped_input_without_changing_samples(monkeypatch, internal_chunks):
    vae = sd.VAE.__new__(sd.VAE)
    vae.first_stage_model = SimpleNamespace(encode=lambda pixels, **_: pixels[:, :, ::4, ::4],
                                           comfy_has_chunked_io=internal_chunks)
    vae.crop_input, vae.downscale_ratio, vae.output_channels, vae.pad_channel_value = True, 4, 3, None
    vae.latent_dim, vae.vae_dtype = 2, torch.float32
    vae.device, vae.output_device = torch.device('cpu'), torch.device('cpu')
    vae.disable_offload, vae.format_encoded = False, None
    vae.memory_used_encode = lambda *_: 1
    vae.patcher = SimpleNamespace(get_free_memory=lambda *_: 100)
    vae.process_input = lambda value: value
    vae.vae_output_dtype = lambda: torch.float32
    monkeypatch.setattr(sd.model_management, 'load_models_gpu', lambda *_args, **_kwargs: None)
    pixels = torch.arange(10 * 14 * 3, dtype=torch.float32).reshape(1, 10, 14, 3)
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = vae.encode(pixels)
    expected = pixels[:, 1:9, 1:13, :].movedim(-1, 1)[:, :, ::4, ::4]
    torch.testing.assert_close(actual, expected)
    assert dict(events)['vae_processed_input'] == {'shape': (1, 3, 8, 12), 'tiled': False, 'tile_count_complete': not internal_chunks}
    assert dict(events)['vae_forward_attempt']['shape'] == (1, 3, 8, 12)
    assert dict(events)['vae_encode_plan'] == dict(shape=(1, 3, 8, 12), tiled=False, forwards=1, batch_size=1,
                                                 complete=not internal_chunks, precision='torch.float32', load_device='cpu', output_device='cpu')
    assert events[-1] == ('vae_forward_completed', {'tiled': False})


@torch.inference_mode()
def test_standard_2d_vae_counts_real_tile_forwards_and_preserves_three_pass_blend():
    vae = sd.VAE.__new__(sd.VAE)
    forwards, events = [], []
    def encode(pixels):
        forwards.append(tuple(pixels.shape))
        return torch.ones_like(pixels)
    vae.first_stage_model = SimpleNamespace(encode=encode)
    vae.process_input, vae.vae_output_dtype = lambda value: value, lambda: torch.float32
    vae.vae_dtype, vae.device, vae.output_device = torch.float32, 'cpu', 'cpu'
    vae.downscale_ratio, vae.latent_channels = 1, 3
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = vae.encode_tiled_(torch.zeros(1, 3, 8, 8), tile_x=4, tile_y=4, overlap=0)
    torch.testing.assert_close(actual, torch.ones(1, 3, 8, 8))
    assert len(forwards) == 12
    assert dict(events)['vae_encode_plan']['forwards'] == len(forwards)
    assert len([event for event, _ in events if event == 'vae_forward_attempt']) == len(forwards)
    assert len([event for event, _ in events if event == 'vae_forward_completed']) == len(forwards)
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer offline')))
    torch.testing.assert_close(vae.encode_tiled_(torch.zeros(1, 3, 8, 8), tile_x=4, tile_y=4, overlap=0), actual)


def test_loaded_image_dimensions_and_vision_reuse_exclude_contents_and_paths(monkeypatch):
    pixels = torch.zeros(2, 10, 12, 3)
    monkeypatch.setattr(nodes, 'resolve_image_path', lambda _: '/PRIVATE/image.png')
    monkeypatch.setattr(nodes.InputImpl, 'VideoFromFile', lambda _: SimpleNamespace(get_components=lambda: SimpleNamespace(images=pixels, alpha=None)))
    monkeypatch.setattr(nodes.comfy.model_management, 'intermediate_device', lambda: 'cpu')
    monkeypatch.setattr(nodes.comfy.model_management, 'intermediate_dtype', lambda: torch.float32)
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    image, mask = nodes.LoadImage().load_image('private input')
    assert image is pixels and mask.shape == (2, 64, 64)
    assert [event for event, _ in events] == ['image_source_decode', 'image_output_prepare', 'image_loaded']
    assert events[-1] == ('image_loaded', {'shape': (2, 10, 12, 3), 'pipeline_version': 1, 'loader': 'pyav',
                                        'mask_shape': (2, 64, 64), 'alpha': False, 'precision': 'torch.float32', 'device': 'cpu'})
    assert '/PRIVATE' not in str(events) and 'private input' not in str(events)

    class Visual(torch.nn.Module):
        def forward(self, image, grid):
            return image * 2
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda: False)
    visual, features = Visual().eval(), VisionFeatures(2, 1024)
    first = features.encode(visual, torch.zeros(1, 3), [1, 1, 1], 'cpu', torch.float32)
    second = features.encode(visual, torch.zeros(1, 3), [1, 1, 1], 'cpu', torch.float32)
    torch.testing.assert_close(first, second)
    assert [event for event, _ in events][-2:] == ['vision_forward', 'vision_feature_reused']
