import json
import struct

import pytest
from safetensors import SafetensorError
from safetensors.torch import save_file
import torch

from scripts.check_qwen3_omni_quant import check_inventory, check_projection, inspect_source, source_decode
from scripts.qwen3_omni_checkpoint import SOURCE_REVISION, validate_layout, validate_source


@pytest.mark.parametrize('rows,cols', [(3, 16), (128, 64), (144, 80), (16, 4304)])
def test_all_fp4_codes_and_padded_scale_tiles(rows, cols):
    weight = torch.arange(rows * cols // 2, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(rows, cols // 2)
    scales = torch.arange(rows * cols // 16, dtype=torch.float32).remainder(31).add(1).reshape(rows, cols // 16).to(torch.float8_e4m3fn)
    result = check_projection(weight, scales, torch.tensor(.125, dtype=torch.bfloat16))
    assert result['fp32_decode_max_abs_error'] == 0
    assert result['logical_shape'] == [rows, cols]
    assert result['native_shape'] == [rows, (cols + 31) // 32 * 32]


def test_source_nibble_order_and_signed_zero():
    weight = torch.tensor([[0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7]], dtype=torch.uint8)
    decoded = source_decode(weight, torch.ones(1, 1).to(torch.float8_e4m3fn), torch.tensor(1.))
    assert decoded.tolist() == [[0., -0., .5, -.5, 1., -1., 1.5, -1.5, 2., -2., 3., -3., 4., -4., 6., -6.]]
    assert torch.signbit(decoded[0, 1])


def fixture_source(path):
    prefix = 'thinker.model.layers.0.self_attn.q_proj'
    tensors = {prefix + '.weight': torch.zeros(16, 8, dtype=torch.uint8),
               prefix + '.weight_scale': torch.ones(16, 1).to(torch.float8_e4m3fn),
               prefix + '.weight_scale_2': torch.tensor(.125, dtype=torch.bfloat16)}
    save_file(tensors, path / 'model.safetensors')
    index = {'metadata': {'total_size': 146}, 'weight_map': {k: 'model.safetensors' for k in tensors}}
    (path / 'model.safetensors.index.json').write_text(json.dumps(index))
    return tensors, index


def test_stale_size_is_reported_as_failed_integrity(tmp_path):
    _, index = fixture_source(tmp_path)
    records, integrity = inspect_source(tmp_path)
    assert integrity['size_matches']
    assert check_inventory(records)['quantized_projections'] == 1
    index['metadata']['total_size'] += 2
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))
    _, integrity = inspect_source(tmp_path)
    assert not integrity['size_matches']
    assert integrity['actual_payload_bytes'] == 146


def test_metadata_reconciliation_requires_exact_revision(tmp_path):
    with pytest.raises(ValueError, match='Only the verified source revision'):
        validate_source(tmp_path, 'unverified')


@pytest.mark.parametrize('fault', ['missing', 'extra', 'shape', 'dtype'])
def test_complete_architecture_inventory_is_required(tmp_path, fault):
    fixture_source(tmp_path)
    records, _ = inspect_source(tmp_path)
    expected = {key: (value[2]['dtype'], value[2]['shape'][:]) for key, value in records.items()}
    key = next(iter(expected))
    if fault == 'missing':
        del records[key]
    elif fault == 'extra':
        records['unexpected.weight'] = records[key]
    elif fault == 'shape':
        records[key][2]['shape'] = [999]
    else:
        records[key][2]['dtype'] = 'F32'
    with pytest.raises(ValueError):
        validate_layout(records, expected)


@pytest.mark.parametrize('fault', ['missing_scale', 'scale_shape', 'scale_dtype', 'packed_rank'])
def test_invalid_quantization_inventory(tmp_path, fault):
    tensors, _ = fixture_source(tmp_path)
    records, _ = inspect_source(tmp_path)
    key = next(k for k in tensors if k.endswith('.weight_scale'))
    if fault == 'missing_scale':
        del records[key]
    elif fault == 'scale_shape':
        records[key][2]['shape'] = [16, 2]
    elif fault == 'scale_dtype':
        records[key][2]['dtype'] = 'F16'
    else:
        records[key.removesuffix('_scale')][2]['shape'] = [16, 8, 1]
    with pytest.raises(ValueError):
        check_inventory(records)


@pytest.mark.parametrize('fault', ['truncated', 'escape', 'symlink_escape', 'misassigned', 'duplicate_json'])
def test_invalid_shards(tmp_path, fault):
    _, index = fixture_source(tmp_path)
    shard = tmp_path / 'model.safetensors'
    if fault == 'truncated':
        shard.write_bytes(shard.read_bytes()[:-1])
    elif fault in ('escape', 'symlink_escape'):
        if fault == 'symlink_escape':
            shard.unlink()
            shard.symlink_to(tmp_path.parent / 'outside.safetensors')
        else:
            index['weight_map'] = {k: '../outside.safetensors' for k in index['weight_map']}
    elif fault == 'misassigned':
        index['weight_map'].pop(next(iter(index['weight_map'])))
    else:
        raw = shard.read_bytes()
        length = struct.unpack('<Q', raw[:8])[0]
        header = raw[8:8 + length].rstrip()
        key = next(iter(json.loads(header)))
        duplicate = header[:-1] + b',' + json.dumps(key).encode() + b':{}}'
        duplicate += b' ' * (-len(duplicate) % 8)
        shard.write_bytes(struct.pack('<Q', len(duplicate)) + duplicate + raw[8 + length:])
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))
    with pytest.raises((ValueError, OSError, SafetensorError)):
        inspect_source(tmp_path)
