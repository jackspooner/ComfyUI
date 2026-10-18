import json

from comfy_kitchen.float_utils import from_blocked, swap_nibbles
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts import convert_qwen3_omni as converter
from scripts.qwen3_omni_checkpoint import inspect_source


def fixture_source(tmp_path):
    weights = {}
    generator = torch.Generator().manual_seed(17)
    for expert in range(2):
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            prefix = f'thinker.model.layers.0.mlp.experts.{expert}.{projection}'
            weights[prefix + '.weight'] = torch.randint(256, (32, 16), dtype=torch.uint8, generator=generator)
            weights[prefix + '.weight_scale'] = torch.ones(32, 2).to(torch.float8_e4m3fn)
            weights[prefix + '.weight_scale_2'] = torch.tensor((expert + 1) / 8, dtype=torch.bfloat16)
    prefix = 'thinker.visual.blocks.0.mlp.linear_fc2'
    weights[prefix + '.weight'] = torch.randint(256, (32, 24), dtype=torch.uint8, generator=generator)
    weights[prefix + '.weight_scale'] = torch.ones(32, 3).to(torch.float8_e4m3fn)
    weights[prefix + '.weight_scale_2'] = torch.tensor(.125, dtype=torch.bfloat16)
    weights['thinker.model.norm.weight'] = torch.arange(32, dtype=torch.bfloat16)
    weights['talker.unused.weight'] = torch.ones(4, dtype=torch.bfloat16)
    save_file(weights, tmp_path / 'weights.safetensors')
    total = sum(value.numel() * value.element_size() for value in weights.values())
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': total},
        'weight_map': {key: 'weights.safetensors' for key in weights}}))
    return weights, inspect_source(tmp_path)[0]


def test_stream_fused_expert_banks_and_exact_padding(tmp_path):
    original, records = fixture_source(tmp_path)
    assets = {'tokenizer_json': b'{"fixture":true}', 'qwen3_omni_chat_template': b'template'}
    plan, consumed, excluded = converter.make_plan(records, assets, 2)
    assert len(consumed) + len(excluded) == len(original)
    assert set(excluded) == {'talker.unused.weight'}
    path = tmp_path / 'converted.safetensors'
    manifest = converter.stream_checkpoint(tmp_path, records, plan, path, {'source_revision': 'fixture'})
    assert manifest['output_sha256'] == converter.sha256_file(path)
    with safe_open(path, framework='pt') as result:
        bank = result.get_tensor('model.layers.0.mlp.experts.gate_up_proj.weight')
        scales = result.get_tensor('model.layers.0.mlp.experts.gate_up_proj.weight_scale')
        globals_ = result.get_tensor('model.layers.0.mlp.experts.gate_up_proj.weight_scale_2')
        assert bank.shape == (2, 64, 16)
        assert globals_.tolist() == [.125, .25]
        for expert in range(2):
            prefix = f'thinker.model.layers.0.mlp.experts.{expert}'
            assert torch.equal(swap_nibbles(bank[expert, :32]), original[prefix + '.gate_proj.weight'])
            assert torch.equal(swap_nibbles(bank[expert, 32:]), original[prefix + '.up_proj.weight'])
            assert torch.equal(from_blocked(scales[expert], 64, 2).view(torch.uint8), torch.ones(64, 2).to(torch.float8_e4m3fn).view(torch.uint8))
        native = result.get_tensor('visual.blocks.0.mlp.linear_fc2.weight')
        assert native.shape == (32, 32)
        assert torch.equal(swap_nibbles(native[:, :24]), original['thinker.visual.blocks.0.mlp.linear_fc2.weight'])
        assert torch.count_nonzero(native[:, 24:]) == 0
        assert result.get_tensor('tokenizer_json').numpy().tobytes() == assets['tokenizer_json']
    with pytest.raises(FileExistsError):
        converter.stream_checkpoint(tmp_path, records, plan, path, {})


def test_unequal_gate_up_scales_refuse_publication(tmp_path):
    original, records = fixture_source(tmp_path)
    original['thinker.model.layers.0.mlp.experts.0.up_proj.weight_scale_2'] *= 2
    save_file(original, tmp_path / 'weights.safetensors')
    plan, _, _ = converter.make_plan(records, {}, 2)
    with pytest.raises(ValueError, match='Gate/up global scales differ'):
        converter.stream_checkpoint(tmp_path, records, plan, tmp_path / 'converted.safetensors', {'source_revision': 'fixture'})
    assert not list(tmp_path.glob('*converted*'))
    assert not list(tmp_path.glob('.*converted*'))


def test_interruption_removes_only_owned_unpublished_files(tmp_path, monkeypatch):
    _, records = fixture_source(tmp_path)
    plan, _, _ = converter.make_plan(records, {}, 2)
    sentinel = tmp_path / '.unrelated'
    sentinel.write_bytes(b'keep')
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(converter, 'transformed_parts', interrupt)
    with pytest.raises(KeyboardInterrupt):
        converter.stream_checkpoint(tmp_path, records, plan, tmp_path / 'converted.safetensors', {'source_revision': 'fixture'})
    assert not list(tmp_path.glob('*converted*'))
    assert not list(tmp_path.glob('.*converted*'))
    assert sentinel.read_bytes() == b'keep'


def test_dangling_destination_is_preserved(tmp_path):
    destination = tmp_path / 'converted.safetensors'
    destination.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        converter.convert(tmp_path, destination, 'invalid')
    assert destination.is_symlink()
    assert not (tmp_path / 'missing').exists()
