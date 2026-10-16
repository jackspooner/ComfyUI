import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from transformers import Qwen2TokenizerFast

from scripts import convert_moss_audio as writer
from scripts import convert_moss_speech as converter
from scripts import convert_moss_codec as codec_converter


def test_single_checkpoint_preserves_independent_heads_and_tokenizer(tmp_path):
    weights = {'language_model.embed_tokens.weight': torch.ones(3, 4, dtype=torch.bfloat16),
               'lm_heads.0.weight': torch.full((3, 4), 2, dtype=torch.bfloat16),
               'emb_ext.0.weight': torch.full((2, 4), 3, dtype=torch.bfloat16),
               'lm_heads.1.weight': torch.full((2, 4), 4, dtype=torch.bfloat16)}
    save_file(weights, tmp_path / 'model.safetensors')
    records = writer.inspect_shards(tmp_path, {key: list(t.shape) for key, t in weights.items()})
    output = tmp_path / 'converted.safetensors'
    tokenizer = b'{"exact": "original bytes"}'
    manifest = writer.stream_checkpoint(records, tokenizer, output, {'format': 'pt'})
    actual = load_file(output)
    assert len(manifest['tensors']) == len(weights)
    assert actual['tokenizer_json'].numpy().tobytes() == tokenizer
    for key, tensor in weights.items():
        assert torch.equal(actual[writer.mapped_name(key)], tensor)
        assert manifest['tensors'][key]['source_file'] == 'model.safetensors'
    assert not torch.equal(actual['model.embed_tokens.weight'], actual['lm_heads.0.weight'])


@pytest.mark.parametrize('fault', ['extra', 'missing', 'shape', 'dtype', 'escape', 'lfs', 'truncated'])
def test_single_checkpoint_rejects_invalid_inventory(tmp_path, fault):
    path = tmp_path / 'model.safetensors'
    weights = {'w': torch.ones(2, 3, dtype=torch.bfloat16)}
    expected = {'w': [2, 3]}
    if fault == 'extra':
        weights['extra'] = torch.zeros(1, dtype=torch.bfloat16)
    elif fault == 'missing':
        expected['missing'] = [1]
    elif fault == 'shape':
        expected['w'] = [3, 2]
    elif fault == 'dtype':
        weights['w'] = weights['w'].float()
    save_file(weights, path)
    if fault == 'escape':
        outside = tmp_path.parent / (tmp_path.name + '.safetensors')
        path.rename(outside)
        path.symlink_to(outside)
    elif fault == 'lfs':
        path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 1000\n')
    elif fault == 'truncated':
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises((ValueError, SafetensorError)):
        writer.inspect_shards(tmp_path, expected)


def test_pinned_source_mismatch_and_path_escape(tmp_path):
    path = tmp_path / 'config.json'
    data = b'{"config":1}'
    path.write_bytes(data)
    manifest = {'revision': 'fixture', 'files': {'config.json': {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}
    snapshot = converter.verify_sources(tmp_path, manifest)
    assert snapshot[path] == writer.file_identity(path)
    path.write_bytes(b'{"config":2}')
    with pytest.raises(ValueError, match='does not match pinned'):
        converter.verify_sources(tmp_path, manifest)
    manifest['files']['../escape'] = manifest['files'].pop('config.json')
    with pytest.raises(ValueError, match='directly inside'):
        converter.verify_sources(tmp_path, manifest)


@pytest.mark.parametrize('when', ['before', 'during'])
def test_changing_source_never_publishes_output(tmp_path, monkeypatch, when):
    path = tmp_path / 'model.safetensors'
    save_file({'w': torch.ones(2, dtype=torch.bfloat16)}, path)
    records = writer.inspect_shards(tmp_path)
    snapshots = {path: writer.file_identity(path)}
    output = tmp_path / 'converted.safetensors'
    def change_source():
        with path.open('ab') as stream:
            stream.write(b'x')
    if when == 'before':
        change_source()
    else:
        original = writer.copy_payload
        def copy_and_change(*args):
            result = original(*args)
            change_source()
            return result
        monkeypatch.setattr(writer, 'copy_payload', copy_and_change)
    with pytest.raises(ValueError, match='Source changed'):
        writer.stream_checkpoint(records, b'{}', output, {}, source_stats=snapshots)
    assert not output.exists()
    assert not output.with_suffix('.manifest.json').exists()
    assert not list(tmp_path.glob('.converted*'))


def test_unknown_attention_config_is_rejected():
    config = dict(converter.SPEECH, language_config=dict(converter.LANGUAGE, layer_types=['full_attention'] * 28))
    converter.validate_config(config)
    config['language_config']['rope_scaling'] = {'rope_type': 'linear', 'factor': 2}
    with pytest.raises(ValueError, match='rope_scaling'):
        converter.validate_config(config)


def test_refuse_existing_output_before_reading_source(tmp_path):
    output = tmp_path / 'converted.safetensors'
    output.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        converter.convert(tmp_path / 'absent', output)
    assert output.read_bytes() == b'preserve'


def test_codec_folding_matches_weight_normalized_convolution(tmp_path):
    convolution = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 5, 1))
    weights = {'conv.' + key: value for key, value in convolution.state_dict().items()}
    save_file(weights, tmp_path / 'model.safetensors')
    records = writer.inspect_shards(tmp_path)
    folded, transforms = codec_converter.fold_convolutions(records)
    torch.testing.assert_close(folded['conv.weight'], convolution.weight, rtol=0, atol=0)
    assert {item['key'] for item in transforms['conv.weight']['sources']} == {
        'conv.parametrizations.weight.original0', 'conv.parametrizations.weight.original1'}
    for source in transforms['conv.weight']['sources']:
        assert source['sha256'] == hashlib.sha256(weights[source['key']].numpy().tobytes()).hexdigest()
    output = tmp_path / 'codec.safetensors'
    manifest = writer.stream_checkpoint(records, None, output, {'architecture': 'fixture_codec'})
    assert 'tokenizer_sha256' not in manifest
    assert load_file(output).keys() == weights.keys()


@pytest.mark.skipif(not os.environ.get('MOSS_VOICEGENERATOR_SOURCE'), reason='Requires pinned original VoiceGenerator bundle')
def test_pinned_tokenizer_matches_local_transformers():
    source = Path(os.environ['MOSS_VOICEGENERATOR_SOURCE'])
    tokenizer = Tokenizer.from_file(str(source / 'tokenizer.json'))
    reference = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    for token, token_id in converter.SPECIAL_TOKENS.items():
        assert tokenizer.encode(token, add_special_tokens=False).ids == [token_id]
    text = ['Hello, world!', '你好，欢迎使用语音生成。', '[whisper] Softly… <think>test</think>']
    text.extend(str(value) for value in reference.added_tokens_decoder.values())
    for value in text:
        assert tokenizer.encode(value, add_special_tokens=False).ids == reference.encode(value, add_special_tokens=False)
    # The stale sidecar must not overwrite the effective tokenizer's audio slots.
    legacy = json.loads((source / 'added_tokens.json').read_text())
    assert legacy['<|audio_pad|>'] == 151654
    assert tokenizer.encode('<|audio_pad|>', add_special_tokens=False).ids != [151654]
