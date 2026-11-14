"""Native ASR artifact boundaries with tiny CPU-only package fixtures."""
import hashlib
import json
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from comfy.cli_args import args
args.cpu = True
from comfy.asr import common
import comfy.memory_management
from comfy.ops import manual_cast


def write_package(tmp_path, *, weight=None, tokenizer=None, entries=None, tokenizer_hash=None):
    if weight is None:
        weight = torch.ones(4, 3)
    encoded = Tokenizer(WordLevel({'word': 0})).to_str().encode()
    if tokenizer is None:
        tokenizer = torch.tensor(list(encoded), dtype=torch.uint8)
    metadata = {'architecture': 'whisper', 'asr_package_version': '1'}
    path = tmp_path / 'native.safetensors'
    save_file({'encoder.weight': weight, 'tokenizer_json': tokenizer}, path, metadata=metadata)
    receipt = {'metadata': metadata,
               'tensors': entries if entries is not None else {'source.weight': {
                   'destination': 'encoder.weight', 'shape': [4, 3], 'dtype': 'F32'}},
               'tokenizer_sha256': tokenizer_hash or hashlib.sha256(tokenizer.numpy().tobytes()).hexdigest()}
    path.with_suffix('.manifest.json').write_text(json.dumps(receipt))
    return path


@pytest.mark.parametrize('dtype, declared', [(torch.bfloat16, 'BF16'), (torch.float16, 'F16'), (torch.float32, 'F32')])
def test_package_preserves_supported_storage_dtype(tmp_path, dtype, declared):
    path = write_package(tmp_path, weight=torch.ones(4, 3, dtype=dtype), entries={
        'source.weight': {'destination': 'encoder.weight', 'shape': [4, 3], 'dtype': declared}})
    metadata, state = common.package(path, 'whisper')
    assert metadata['architecture'] == 'whisper'
    assert state['encoder.weight'].dtype == dtype


@pytest.mark.parametrize('weight', [torch.ones(5, 3), torch.ones(4, 3, dtype=torch.int32)])
def test_package_rejects_shape_and_dtype_mismatch_before_loading(tmp_path, weight):
    path = write_package(tmp_path, weight=weight)
    with patch.object(common.comfy.utils, 'load_torch_file') as load:
        with pytest.raises(ValueError, match='shape/dtype'):
            common.package(path, 'whisper')
    load.assert_not_called()


def test_package_rejects_matching_unsupported_integer_dtype(tmp_path):
    path = write_package(tmp_path, weight=torch.ones(4, 3, dtype=torch.int32), entries={
        'source.weight': {'destination': 'encoder.weight', 'shape': [4, 3], 'dtype': 'I32'}})
    with pytest.raises(ValueError, match='shape/dtype'):
        common.package(path, 'whisper')


@pytest.mark.parametrize('kind', ['duplicate', 'reserved', 'missing', 'extra'])
def test_package_rejects_invalid_tensor_inventory(tmp_path, kind):
    entry = {'destination': 'encoder.weight', 'shape': [4, 3], 'dtype': 'F32'}
    entries = {'source.weight': entry}
    if kind == 'duplicate':
        entries['alias.weight'] = entry.copy()
    elif kind == 'reserved':
        entry['destination'] = 'tokenizer_json'
    elif kind == 'missing':
        entry['destination'] = 'missing.weight'
    else:
        entries['extra.weight'] = {**entry, 'destination': 'extra.weight'}
    path = write_package(tmp_path, entries=entries)
    with pytest.raises(ValueError, match='destinations|inventory'):
        common.package(path, 'whisper')


@pytest.mark.parametrize('tokenizer', [torch.ones(4, dtype=torch.int32), torch.ones(2, 2, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8)])
def test_package_rejects_invalid_tokenizer_tensor(tmp_path, tokenizer):
    path = write_package(tmp_path, tokenizer=tokenizer)
    with pytest.raises(ValueError, match='U8 vector'):
        common.package(path, 'whisper')


def test_package_rejects_changed_tokenizer_hash(tmp_path):
    path = write_package(tmp_path, tokenizer_hash='0' * 64)
    with pytest.raises(ValueError, match='tokenizer differs'):
        common.package(path, 'whisper')


def test_package_rejects_invalid_tokenizer_json(tmp_path):
    path = write_package(tmp_path, tokenizer=torch.tensor(list(b'{"model":null}'), dtype=torch.uint8))
    with patch.object(common.comfy.utils, 'load_torch_file') as load:
        with pytest.raises(Exception, match='ModelUntagged'):
            common.package(path, 'whisper')
    load.assert_not_called()


def architecture(aimdo, monkeypatch):
    monkeypatch.setattr(comfy.memory_management, 'aimdo_enabled', aimdo)
    return torch.nn.Sequential(manual_cast.Linear(3, 4, dtype=torch.bfloat16),
                               manual_cast.Embedding(5, 4, dtype=torch.bfloat16))


@pytest.mark.parametrize('aimdo', [False, True])
@pytest.mark.parametrize('invalid', ['weight_shape', 'bias_shape', 'embedding_shape', 'integer', 'missing', 'extra'])
def test_patcher_checks_architecture_before_assignment(aimdo, invalid, monkeypatch):
    model = architecture(aimdo, monkeypatch)
    state = {'0.weight': torch.ones(4, 3), '0.bias': torch.ones(4), '1.weight': torch.ones(5, 4)}
    if invalid == 'weight_shape':
        state['0.weight'] = torch.ones(5, 3)
    elif invalid == 'bias_shape':
        state['0.bias'] = torch.ones(5)
    elif invalid == 'embedding_shape':
        state['1.weight'] = torch.ones(6, 4)
    elif invalid == 'integer':
        state['0.weight'] = torch.ones(4, 3, dtype=torch.int32)
    elif invalid == 'missing':
        del state['0.bias']
    else:
        state['extra'] = torch.ones(1)
    with patch.object(model, 'load_state_dict') as assign:
        with pytest.raises(ValueError, match='architecture tensor'):
            common.patcher(model, state, torch.device('cpu'))
    assign.assert_not_called()


@pytest.mark.parametrize('aimdo', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_patcher_accepts_storage_precision_independent_of_compute(aimdo, dtype, monkeypatch):
    model = architecture(aimdo, monkeypatch)
    state = {'0.weight': torch.ones(4, 3, dtype=dtype), '0.bias': torch.ones(4, dtype=torch.float32),
             '1.weight': torch.ones(5, 4, dtype=torch.float16)}
    result = common.patcher(model, state, torch.device('cpu'))
    assert result.model is model
    assert model[0].weight.dtype == dtype
    assert model[0].bias.dtype == torch.float32
    assert model[1].weight.dtype == torch.float16


@pytest.mark.parametrize('aimdo', [False, True])
def test_patcher_accepts_biasless_linear(aimdo, monkeypatch):
    monkeypatch.setattr(comfy.memory_management, 'aimdo_enabled', aimdo)
    model = manual_cast.Linear(3, 4, bias=False, dtype=torch.bfloat16)
    common.patcher(model, {'weight': torch.ones(4, 3)}, torch.device('cpu'))
    assert model.bias is None


@pytest.mark.parametrize('shape', [(2, 1, 8), (1, 8), (8,), (1, 0, 8), (1, 1, 0), (0, 1, 8)])
def test_audio_validation_rejects_invalid_single_clip_shapes(shape):
    with pytest.raises(ValueError, match='one nonempty AUDIO clip'):
        common.validate_audio({'waveform': torch.zeros(shape), 'sample_rate': 16000})


@pytest.mark.parametrize('sample', [float('nan'), float('inf'), -float('inf')])
def test_audio_validation_rejects_nonfinite_samples(sample):
    with pytest.raises(ValueError, match='non-finite'):
        common.validate_audio({'waveform': torch.tensor([[[sample]]]), 'sample_rate': 16000})


def test_audio_validation_checks_mono_compute_precision():
    with pytest.raises(ValueError, match='non-finite'):
        common.validate_audio({'waveform': torch.full((1, 1, 8), 1e308, dtype=torch.float64), 'sample_rate': 16000})


@pytest.mark.parametrize('audio', [None, {}, {'waveform': torch.ones(1, 1, 8)}])
def test_audio_validation_requires_audio_mapping(audio):
    with pytest.raises(ValueError, match='AUDIO waveform and sample_rate'):
        common.validate_audio(audio)


def test_audio_validation_requires_tensor():
    with pytest.raises(TypeError, match='waveform must be a tensor'):
        common.validate_audio({'waveform': [[[0.0]]], 'sample_rate': 16000})


@pytest.mark.parametrize('rate', [0, -1, 0.5])
def test_audio_validation_rejects_unusable_sample_rate(rate):
    with pytest.raises(ValueError, match='sample rate must be positive'):
        common.validate_audio({'waveform': torch.ones(1, 2, 8), 'sample_rate': rate})


@pytest.mark.parametrize('rate', [16000, 16000.5, '16000'])
def test_audio_validation_returns_cpu_float_mono_without_resampling(rate):
    value = torch.arange(16, dtype=torch.float64).reshape(1, 2, 8)
    with patch.object(common.comfy.audio, 'resample') as resample:
        mono = common.validate_audio({'waveform': value, 'sample_rate': rate})
    assert mono.device.type == 'cpu' and mono.dtype == torch.float32
    torch.testing.assert_close(mono, value.float().mean(1))
    resample.assert_not_called()
