import json
from types import SimpleNamespace

from comfy_kitchen.float_utils import from_blocked, swap_nibbles
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from comfy.cli_args import args
args.cpu = True
import comfy.ops
from comfy.text_encoders.qwen35 import Qwen35VisionPatchMerger
from comfy.text_encoders.qwen3vl import Qwen3VLDeepstackMerger
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
    for name, norm_size in [('merger', 8), ('merger_list.0', 32)]:
        prefix = 'thinker.visual.' + name
        weights[prefix + '.ln_q.weight'] = torch.ones(norm_size, dtype=torch.bfloat16)
        weights[prefix + '.ln_q.bias'] = torch.zeros(norm_size, dtype=torch.bfloat16)
        for layer in (0, 2):
            target = prefix + f'.mlp.{layer}'
            weights[target + '.weight'] = torch.randint(256, (32, 16), dtype=torch.uint8, generator=generator)
            weights[target + '.weight_scale'] = torch.ones(32, 2).to(torch.float8_e4m3fn)
            weights[target + '.weight_scale_2'] = torch.tensor(.125, dtype=torch.bfloat16)
            weights[target + '.bias'] = torch.zeros(32, dtype=torch.bfloat16)
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
        operations = comfy.ops.mixed_precision_ops({}, torch.bfloat16, full_precision_mm=True)
        for prefix, component in [('visual.merger.', Qwen35VisionPatchMerger),
                                  ('visual.deepstack_merger_list.0.', Qwen3VLDeepstackMerger)]:
            native = component(8, 2, 32, dtype=torch.bfloat16, device='cpu', ops=operations)
            state = {key.removeprefix(prefix): result.get_tensor(key) for key in result.keys() if key.startswith(prefix)}
            missing, unexpected = native.load_state_dict(state, strict=False)
            assert not missing and not unexpected
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


def original_fixture(tmp_path):
    generator = torch.Generator().manual_seed(73)
    weights = {}
    for expert in range(2):
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            key = f'thinker.model.layers.0.mlp.experts.{expert}.{projection}.weight'
            weights[key] = torch.randn(32, 32, dtype=torch.bfloat16, generator=generator) if expert else torch.zeros(32, 32, dtype=torch.bfloat16)
    weights['thinker.visual.blocks.0.mlp.linear_fc2.weight'] = torch.randn(32, 48, dtype=torch.bfloat16, generator=generator)
    weights['thinker.model.layers.0.self_attn.q_proj.weight'] = torch.randn(32, 32, dtype=torch.bfloat16, generator=generator)
    weights['thinker.model.layers.0.mlp.gate.weight'] = torch.randn(2, 32, dtype=torch.bfloat16, generator=generator)
    weights['thinker.lm_head.weight'] = torch.randn(64, 32, dtype=torch.bfloat16, generator=generator)
    weights['thinker.model.embed_tokens.weight'] = torch.randn(64, 32, dtype=torch.bfloat16, generator=generator)
    weights['talker.unused.weight'] = torch.ones(4, dtype=torch.bfloat16)
    save_file(weights, tmp_path / 'weights.safetensors')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {
        'total_size': sum(value.numel() * value.element_size() for value in weights.values())},
        'weight_map': {key: 'weights.safetensors' for key in weights}}))
    return weights, inspect_source(tmp_path)[0]


def test_original_fp8_streaming_scales_zero_experts_and_padding(tmp_path):
    weights, records = original_fixture(tmp_path)
    plan, consumed, excluded = converter.make_plan(records, {}, 2, 'fp8')
    assert set(consumed) | set(excluded) == set(weights)
    destination = tmp_path / 'fp8.safetensors'
    manifest = converter.stream_checkpoint(tmp_path, records, plan, destination, {'source_revision': 'fixture'})
    assert manifest['output_sha256'] == converter.sha256_file(destination)
    with safe_open(destination, framework='pt') as native:
        for expert in range(2):
            prefix = 'model.layers.0.mlp.experts.gate_up_proj'
            original = torch.cat([weights[f'thinker.model.layers.0.mlp.experts.{expert}.{projection}.weight']
                                  for projection in ('gate_proj', 'up_proj')])
            scale = native.get_tensor(prefix + '.weight_scale')[expert]
            expected_scale = (original.float().abs().max() / 448).clamp_min(torch.finfo(torch.float32).tiny)
            torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
            expected = (original.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            actual = native.get_tensor(prefix + '.weight')[expert]
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert torch.isfinite(actual.float()).all()
        padded = native.get_tensor('visual.blocks.0.mlp.linear_fc2.weight')
        assert padded.shape == (32, 64)
        assert torch.count_nonzero(padded[:, 48:].float()) == 0
        descriptor = json.loads(native.get_tensor('model.layers.0.mlp.experts.gate_up_proj.comfy_quant').numpy().tobytes())
        assert descriptor == {'format': 'float8_e4m3fn', 'num_experts': 2, 'full_precision_matrix_mult': True}
        for key in ('thinker.model.layers.0.mlp.gate.weight', 'thinker.lm_head.weight', 'thinker.model.embed_tokens.weight'):
            assert torch.equal(native.get_tensor(converter.mapped_name(key)), weights[key])


def test_fp8_nonfinite_source_does_not_publish(tmp_path):
    weights, records = original_fixture(tmp_path)
    weights['thinker.model.layers.0.self_attn.q_proj.weight'][0, 0] = float('nan')
    save_file(weights, tmp_path / 'weights.safetensors')
    plan, _, _ = converter.make_plan(records, {}, 2, 'fp8')
    with pytest.raises(ValueError, match='finite BF16'):
        converter.stream_checkpoint(tmp_path, records, plan, tmp_path / 'candidate.safetensors', {'source_revision': 'fixture'})
    assert not list(tmp_path.glob('*candidate*'))
    assert not list(tmp_path.glob('.*candidate*'))


def test_fp8_requires_original_source_not_decoded_nvfp4(tmp_path):
    with pytest.raises(ValueError, match='pinned source revision for the selected mode'):
        converter.convert(tmp_path, tmp_path / 'fp8.safetensors', converter.SOURCE_REVISION, 'fp8')


def test_insufficient_output_space_does_not_create_partial_files(tmp_path, monkeypatch):
    _, records = original_fixture(tmp_path)
    plan, _, _ = converter.make_plan(records, {}, 2, 'fp8')
    monkeypatch.setattr(converter.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='Insufficient destination space'):
        converter.stream_checkpoint(tmp_path, records, plan, tmp_path / 'candidate.safetensors', {'source_revision': 'fixture'})
    assert not list(tmp_path.glob('*candidate*'))
    assert not list(tmp_path.glob('.*candidate*'))


def test_published_block_fp8_preserves_weights_scales_and_dense_vision_padding(tmp_path):
    generator = torch.Generator().manual_seed(721)
    weights = {}
    for expert in range(2):
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            prefix = f'thinker.model.layers.0.mlp.experts.{expert}.{projection}'
            weights[prefix + '.weight'] = torch.randint(-12, 13, (128, 256), generator=generator).to(torch.float8_e4m3fn)
            weights[prefix + '.weight_scale_inv'] = torch.rand(1, 2, generator=generator).bfloat16() + .01
    weights['thinker.visual.blocks.0.mlp.linear_fc2.weight'] = torch.randn(32, 48, dtype=torch.bfloat16, generator=generator)
    save_file(weights, tmp_path / 'weights.safetensors')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {
        'total_size': sum(v.numel() * v.element_size() for v in weights.values())},
        'weight_map': {key: 'weights.safetensors' for key in weights}}))
    records, _ = inspect_source(tmp_path)
    plan, consumed, excluded = converter.make_plan(records, {}, 2, 'fp8-block128')
    assert set(consumed) == set(weights) and not excluded
    destination = tmp_path / 'block-fp8.safetensors'
    converter.stream_checkpoint(tmp_path, records, plan, destination, {'source_revision': 'fixture'})
    with safe_open(destination, framework='pt') as result:
        for expert in range(2):
            target = 'model.layers.0.mlp.experts.gate_up_proj'
            for suffix, source_suffix in [('weight', 'weight'), ('weight_scale', 'weight_scale_inv')]:
                expected = torch.cat([weights[f'thinker.model.layers.0.mlp.experts.{expert}.{p}.{source_suffix}']
                                      for p in ('gate_proj', 'up_proj')])
                actual = result.get_tensor(target + '.' + suffix)[expert]
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        padded = result.get_tensor('visual.blocks.0.mlp.linear_fc2.weight')
        assert padded.shape == (32, 64) and not padded[:, 48:].count_nonzero()
        assert torch.equal(padded[:, :48], weights['thinker.visual.blocks.0.mlp.linear_fc2.weight'])
