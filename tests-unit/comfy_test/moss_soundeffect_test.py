import hashlib
import html
import importlib.util
import importlib
import json
import os
import re
from pathlib import Path
import sys
import types

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model
from safetensors import safe_open

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.ldm.audio.moss_soundeffect_vae import MossSoundEffectVAE
from comfy.ldm.audio.moss_soundeffect import MossSoundEffect
from comfy.text_encoders.moss_soundeffect import MossSoundEffectTokenizer, MossSoundEffectClipModel, ftfy
from scripts.convert_moss_soundeffect import fold_dac_weights, convert_vae, transformer_key
from comfy import model_detection


@pytest.fixture(scope='module')
def reference():
    source = os.environ.get('MOSS_SOUNDEFFECT_SOURCE')
    if not source:
        pytest.skip('Set MOSS_SOUNDEFFECT_SOURCE to the pinned moss_soundeffect_v2 source')
    path = Path(source) / 'diffsynth/models/dac_vae.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '2daf0150fd60d06b4f18c54148dcc831674fe551306482c96a746b410a9382dd'
    spec = importlib.util.spec_from_file_location('soundeffect_dac_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.is_grad_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        torch.set_grad_enabled(False)
        yield
    torch.set_grad_enabled(previous)


@pytest.mark.parametrize('length', [1, 959, 960, 1921])
def test_continuous_dac_matches_reference(reference, length):
    config = dict(encoder_dim=2, encoder_rates=[2, 3, 4, 5, 8], latent_dim=4, decoder_dim=64, decoder_rates=[8, 5, 4, 3, 2])
    original = reference.DAC(**config, sample_rate=48000, continuous=True).eval()
    native = MossSoundEffectVAE(**config, operations=manual_cast).eval()
    folded, transforms = fold_dac_weights(original.state_dict())
    assert len(transforms) == 74
    native.load_state_dict(folded, strict=True)
    audio = torch.randn(1, 1, length)
    expected = original.encode(original.preprocess(audio, 48000))[0].mode()
    latent = native.encode(audio)
    assert latent.shape == (1, 4, (length + 959) // 960)
    torch.testing.assert_close(latent, expected, atol=2e-6, rtol=2e-5)
    decoded = native.decode(latent)
    assert decoded.shape[-1] == latent.shape[-1] * 960
    torch.testing.assert_close(decoded, original.decode(expected), atol=2e-6, rtol=2e-5)


def test_dac_conversion_rejects_bad_weights_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match='dense FP32'):
        fold_dac_weights({'x': torch.ones(1, dtype=torch.float16)})
    with pytest.raises(ValueError, match='Missing weight magnitude'):
        fold_dac_weights({'x.weight_v': torch.ones(1, 1, 1)})
    destination = tmp_path / 'existing.safetensors'
    destination.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        convert_vae(tmp_path / 'missing', destination)
    assert destination.read_bytes() == b'preserve'


def test_checkpoint_dac_decode(reference):
    source = os.environ.get('MOSS_SOUNDEFFECT_WEIGHTS')
    if not source:
        pytest.skip('Set MOSS_SOUNDEFFECT_WEIGHTS to the original model bundle for checkpoint parity')
    checkpoint = torch.load(Path(source) / 'vae/vae_128d_48k.pth', map_location='cpu', weights_only=True, mmap=True)
    with torch.device('meta'):
        original = reference.DAC(**checkpoint['metadata']['kwargs'])
        native = MossSoundEffectVAE(operations=manual_cast)
    original.load_state_dict(checkpoint['state_dict'], assign=True, strict=True)
    folded, _ = fold_dac_weights(checkpoint['state_dict'])
    native.load_state_dict(folded, assign=True, strict=True)
    latent = torch.randn(1, 128, 2)
    actual = native.decode(latent)
    expected = original.decode(latent)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.fixture(scope='module')
def dit_reference():
    source = os.environ.get('MOSS_SOUNDEFFECT_SOURCE')
    if not source:
        pytest.skip('Set MOSS_SOUNDEFFECT_SOURCE to the pinned moss_soundeffect_v2 source')
    source = Path(source) / 'diffsynth/models'
    for filename, digest in [('wan_audio_dit.py', '50bcf69de26619d3526b6069be8c5b0495e0d15049e6a327bef6631fd7665fbe'),
                             ('wan_video_dit.py', '7cadd1cdef05078167f5845f033ecac018b4d850be35e4e320c83cd2e93a7dc1')]:
        assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == digest
    package = types.ModuleType('soundeffect_dit_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    video = importlib.import_module(package.__name__ + '.wan_video_dit')
    video.FLASH_ATTN_3_AVAILABLE = video.FLASH_ATTN_2_AVAILABLE = video.SAGE_ATTN_AVAILABLE = False
    return importlib.import_module(package.__name__ + '.wan_audio_dit')


@pytest.mark.parametrize('batch,frames', [(1, 3), (2, 11)])
def test_flow_model_matches_reference(dit_reference, batch, frames):
    config = dict(dim=48, in_dim=4, ffn_dim=96, out_dim=4, text_dim=12, freq_dim=16, eps=1e-6, patch_size=(1,), num_heads=4, num_layers=2)
    original = dit_reference.WanAudioModel(**config, has_image_input=False, vae_type='dac')
    native = MossSoundEffect(**config, operations=manual_cast)
    native.load_state_dict(original.state_dict(), strict=True)
    x, context = torch.randn(batch, 4, frames), torch.randn(batch, 7, 12)
    timestep = torch.linspace(250, 950, batch)
    torch.testing.assert_close(native(x, timestep, context), original(x, timestep, context), atol=2e-5, rtol=2e-5)


def test_patch_padding_crops_target():
    model = MossSoundEffect(dim=16, in_dim=4, out_dim=4, text_dim=8, ffn_dim=32, freq_dim=8, patch_size=(2,), num_heads=2, num_layers=1, operations=manual_cast)
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.05)
    assert model(torch.randn(2, 4, 7), torch.tensor([50., 90.]), torch.randn(2, 4, 8)).shape == (2, 4, 7)


def test_checkpoint_transformer_mapping_and_detection(dit_reference):
    source = os.environ.get('MOSS_SOUNDEFFECT_WEIGHTS')
    if not source:
        pytest.skip('Set MOSS_SOUNDEFFECT_WEIGHTS to the original model bundle')
    with safe_open(Path(source) / 'transformer/diffusion_pytorch_model.safetensors', framework='pt') as weights:
        shapes = {key: torch.empty(weights.get_slice(key).get_shape(), device='meta') for key in weights.keys()}
    expected, _ = dit_reference.WanAudioModel.state_dict_converter().from_diffusers(shapes)
    mapped = {transformer_key(key): value for key, value in shapes.items()}
    assert len(mapped) == len(shapes) == len(expected)
    assert all(mapped[key] is value for key, value in expected.items())
    with torch.device('meta'):
        native = MossSoundEffect(operations=manual_cast)
    native.load_state_dict(mapped, strict=True)
    assert model_detection.detect_unet_config(mapped, '') == {'audio_model': 'moss_soundeffect'}
    config = model_detection.model_config_from_unet(mapped, '')
    assert config.__class__.__name__ == 'MossSoundEffect'
    config.set_inference_dtype(torch.float32, None)
    wrapped = config.get_model(mapped, device=torch.device('meta'))
    assert wrapped.get_dtype() == torch.float32
    wrapped.diffusion_model.load_state_dict(mapped, strict=True)


@pytest.mark.parametrize('block', [0, 29, None])
def test_checkpoint_transformer_components(dit_reference, block):
    source = os.environ.get('MOSS_SOUNDEFFECT_WEIGHTS')
    if not source:
        pytest.skip('Set MOSS_SOUNDEFFECT_WEIGHTS to the original model bundle')
    with safe_open(Path(source) / 'transformer/diffusion_pytorch_model.safetensors', framework='pt') as weights:
        selected = {key.replace(f'blocks.{block}.', 'blocks.0.'): weights.get_tensor(key) for key in weights.keys()
                    if block is None or not key.startswith('blocks.') or key.startswith(f'blocks.{block}.')}
    expected_weights, _ = dit_reference.WanAudioModel.state_dict_converter().from_diffusers(selected)
    config = dict(dim=1536, in_dim=128, ffn_dim=8960, out_dim=128, text_dim=2048, freq_dim=256,
                  eps=1e-6, patch_size=(1,), num_heads=12, num_layers=30 if block is None else 1)
    with torch.device('meta'):
        original = dit_reference.WanAudioModel(**config, has_image_input=False, vae_type='dac')
        native = MossSoundEffect(**config, operations=manual_cast)
    for index, frequency in enumerate(dit_reference.precompute_freqs_cis_1d(128)):
        setattr(original, f'freqs_cis_{index}', frequency)
    original.load_state_dict(expected_weights, strict=True, assign=True)
    native.load_state_dict({transformer_key(key): value for key, value in selected.items()}, strict=True, assign=True)
    x, context, timestep = torch.randn(1, 128, 3), torch.randn(1, 7, 2048), torch.tensor([750.])
    torch.testing.assert_close(native(x, timestep, context), original(x, timestep, context), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize('text', ['', 'Thunder &amp;amp; wind\n duration: 5.0s', 'cafÃ© <|endoftext|> rain'])
def test_tokenizer_matches_pinned_assets(text):
    source = os.environ.get('MOSS_SOUNDEFFECT_WEIGHTS')
    if not source:
        pytest.skip('Set MOSS_SOUNDEFFECT_WEIGHTS to the original model bundle')
    path = Path(source) / 'tokenizer'
    raw = (path / 'tokenizer.json').read_bytes()
    tokenizer = MossSoundEffectTokenizer(tokenizer_data={'tokenizer_json': raw})
    expected = AutoTokenizer.from_pretrained(path, local_files_only=True)(re.sub(r'\s+', ' ', html.unescape(html.unescape(ftfy.fix_text(text)))).strip(), padding='max_length', max_length=512)
    ids = [token[0] for token in tokenizer.tokenize_with_weights(text)['moss_soundeffect'][0]]
    assert [151643 if token == -1 else token for token in ids] == expected.input_ids
    assert [int(token != -1) for token in ids] == expected.attention_mask
    assert len(tokenizer.tokenize_with_weights('rain ' * 600)['moss_soundeffect'][0]) > 512


def test_qwen_conditioning_preserves_literal_pad_token_and_zeroes_padding():
    config = dict(vocab_size=151644, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=40960, rope_theta=1000000.)
    original = Qwen3Model(Qwen3Config(**config, head_dim=128)).eval()
    native = MossSoundEffectClipModel(dtype=torch.float32, model_options={'_model_config': config})
    weights = {'model.' + key: value for key, value in original.state_dict().items()}
    weights['model.lm_head.weight'] = weights['model.embed_tokens.weight']
    native.load_sd(weights)
    ids = torch.tensor([[1, 2, 151643, 3]])
    expected = original(ids, attention_mask=torch.tensor([[1, 1, 1, 1]])).last_hidden_state
    actual = native([[1, 2, 151643, 3, -1, -1]])[0]
    torch.testing.assert_close(actual[:, :4], expected, atol=2e-6, rtol=2e-5)
    assert torch.count_nonzero(actual[:, 4:]) == 0
    assert torch.count_nonzero(native([[-1] * 6])[0]) == 0


def test_checkpoint_qwen_conditioning():
    source = os.environ.get('MOSS_SOUNDEFFECT_WEIGHTS')
    if not source:
        pytest.skip('Set MOSS_SOUNDEFFECT_WEIGHTS to the original model bundle')
    source = Path(source) / 'text_encoder'
    config = Qwen3Config(**json.loads((source / 'config.json').read_text()))
    with torch.device('meta'):
        original = Qwen3Model(config).eval()
        native = MossSoundEffectClipModel(dtype=torch.float32, device='meta')
    original.rotary_emb = type(original.rotary_emb)(config)
    weights = {}
    for path in sorted(source.glob('*.safetensors')):
        with safe_open(path, framework='pt') as shard:
            weights.update({key: shard.get_tensor(key).float() for key in shard.keys() if key != 'lm_head.weight'})
    original.load_state_dict({key.removeprefix('model.'): value for key, value in weights.items()}, strict=True, assign=True)
    native.can_assign_sd = True
    native.load_sd(weights)
    ids = torch.tensor([[1, 2, 151643, 3]])
    expected = original(ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    actual = native([[1, 2, 151643, 3, -1, -1]])[0]
    torch.testing.assert_close(actual[:, :4], expected, atol=2e-4, rtol=2e-4)
    assert torch.count_nonzero(actual[:, 4:]) == 0
