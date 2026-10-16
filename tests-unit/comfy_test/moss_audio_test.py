import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file, load_file

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.text_encoders.moss_audio import MossAudioEncoder, audio_features, audio_token_ids, AUDIO_TOKEN
from scripts import convert_moss_audio as converter


def shards(tmp_path):
    weights = {'language_model.small.weight': torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
               'lm_head.weight': torch.tensor([[1, 2]], dtype=torch.bfloat16)}
    save_file(weights, tmp_path / 'model-1.safetensors')
    index = {'metadata': {'total_size': 28}, 'weight_map': {key: 'model-1.safetensors' for key in weights}}
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))
    return weights, index


def test_stream_exact_payloads_and_tokenizer(tmp_path):
    weights, _ = shards(tmp_path)
    records = converter.inspect_shards(tmp_path)
    dest = tmp_path / 'converted.safetensors'
    tokenizer = b'{"arbitrary":"utf8 \\u00e9"}'
    converter.stream_checkpoint(records, tokenizer, dest, {'format': 'pt'})
    actual = load_file(dest)
    for key, tensor in weights.items():
        assert torch.equal(actual[converter.mapped_name(key)], tensor)
    assert actual['tokenizer_json'].numpy().tobytes() == tokenizer
    assert dest.with_suffix('.manifest.json').is_file()
    with pytest.raises(FileExistsError):
        converter.stream_checkpoint(records, tokenizer, dest, {})


@pytest.mark.parametrize('fault', ['missing_shard', 'truncated', 'missing_tensor', 'duplicate', 'lfs'])
def test_invalid_shards(tmp_path, fault):
    weights, index = shards(tmp_path)
    path = tmp_path / 'model-1.safetensors'
    if fault == 'missing_shard':
        path.unlink()
    elif fault == 'truncated':
        path.write_bytes(path.read_bytes()[:-1])
    elif fault == 'missing_tensor':
        index['weight_map']['missing'] = 'model-1.safetensors'
    elif fault == 'duplicate':
        save_file(weights, tmp_path / 'model-2.safetensors')
        index['weight_map']['extra'] = 'model-2.safetensors'
    elif fault == 'lfs':
        path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 1000\n')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))
    with pytest.raises(Exception):
        converter.inspect_shards(tmp_path)


def test_duplicate_json_key():
    with pytest.raises(ValueError, match='Duplicate'):
        json.loads('{"x":1,"x":2}', object_pairs_hook=converter.unique_object)


def test_interrupted_conversion_leaves_no_published_output(tmp_path, monkeypatch):
    shards(tmp_path)
    records = converter.inspect_shards(tmp_path)
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(converter, 'copy_payload', interrupt)
    with pytest.raises(KeyboardInterrupt):
        converter.stream_checkpoint(records, b'{}', tmp_path / 'converted.safetensors', {})
    assert not list(tmp_path.glob('*converted*'))
    assert not list(tmp_path.glob('.*converted*'))


@pytest.mark.parametrize('length,tail', [(24, [AUDIO_TOKEN]), (25, [17]), (125, [16,15]), (1500,[16,17,15])])
def test_timestamp_boundaries(length, tail):
    ids = audio_token_ids(length)
    assert ids.count(AUDIO_TOKEN) == length
    assert ids[-len(tail):] == tail


@pytest.mark.parametrize('rate', [16000, 44100, 48000])
@pytest.mark.parametrize('channels', [1, 2])
def test_audio_rates_channels(rate, channels):
    features = audio_features({'waveform': torch.zeros(1, channels, rate), 'sample_rate': rate})
    assert features.shape == (128, 100)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize('samples', [1, 100, 201, 64001, 480160, 1920000])
def test_no_tail_truncation(samples):
    features = audio_features({'waveform': torch.zeros(1, 1, samples), 'sample_rate': 16000})
    assert features.shape[1] == max(samples, 201) // 160


@pytest.mark.parametrize('waveform,rate', [(torch.zeros(1,1,0),16000), (torch.zeros(2,1,160),16000), (torch.zeros(1,1,160),0), (torch.full((1,1,160), float('nan')),16000)])
def test_bad_audio(waveform, rate):
    with pytest.raises(ValueError, match='MOSS-Audio'):
        audio_features({'waveform': waveform, 'sample_rate': rate})


def test_encoder_partial_chunk_and_capture_shapes():
    encoder = MossAudioEncoder(dtype=torch.float32, operations=manual_cast, d_model=16, n_heads=2, d_ff=32,
                               n_layers=3, conv_channels=2, capture_layers=(0,1))
    for parameter in encoder.parameters():
        torch.nn.init.normal_(parameter, std=.1)
    output = encoder(torch.randn(128, 411))
    assert len(output) == 3
    assert all(t.shape == (52,16) and torch.isfinite(t).all() for t in output)


def test_original_inventory():
    shapes = converter.expected_shapes()
    assert len(shapes) == 901
    assert sum(__import__('math').prod(shape) * 2 for shape in shapes.values()) == 18104926976
