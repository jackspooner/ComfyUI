"""Pinned Tokenizer v2 component math and native inventory; no full inference."""
from contextlib import nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.ldm.audio.moss_audio_tokenizer import ProjectedTransformer, ResidualLFQ
from comfy.ldm.audio.moss_audio_tokenizer_v2 import CodecV2Attention, MossAudioTokenizerV2
from scripts.convert_moss_codec import mapped_name


@pytest.fixture(scope='module')
def reference():
    selected = os.environ.get('MOSS_CODEC_V2_REFERENCE')
    if not selected:
        pytest.skip('Requires pinned MOSS_CODEC_V2_REFERENCE; no weights are loaded')
    source = Path(selected)
    manifest = json.loads((Path(__file__).parents[2] / 'scripts/moss_codec_v2.json').read_text())
    for name in ('configuration_moss_audio_tokenizer.py', 'modeling_moss_audio_tokenizer.py', 'config.json'):
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == manifest['files'][name]['sha256']
    package = types.ModuleType('moss_codec_v2_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + '.modeling_moss_audio_tokenizer')


def fold_and_map(state):
    output = {}
    for key, value in state.items():
        if key.endswith('.parametrizations.weight.original0'):
            root = key.removesuffix('.parametrizations.weight.original0')
            output[mapped_name(root+'.weight', 'v2')] = torch._weight_norm(
                state[root+'.parametrizations.weight.original1'], value, 0)
        elif not key.endswith('.parametrizations.weight.original1'):
            output[mapped_name(key, 'v2')] = value
    return output


@pytest.fixture(autouse=True)
def deterministic():
    with torch.random.fork_rng(devices=[]), torch.set_grad_enabled(False):
        torch.manual_seed(7401)
        yield


@pytest.mark.parametrize('bf16', [False, True])
def test_projected_math_and_call_private_ring_match_pinned_source(reference, bf16):
    original = reference.MossAudioTokenizerProjectedTransformer(12, 8, 16, conv_layout=True, module_type='Transformer',
        num_heads=2, num_layers=2, dim_feedforward=40, causal=True, context=10,
        positional_embedding='rope', norm='layer_norm', layer_scale=.01, attention_implementation='sdpa')
    for layer in original.transformer.layers:
        for norm in (layer.norm1, layer.norm2):
            norm.weight.uniform_(.5, 1.5)
            norm.bias.uniform_(-.3, .3)
    native = ProjectedTransformer(12, 8, 16, 2, 40, 2, 10, operations=manual_cast,
                                  attention_class=CodecV2Attention, force_projection=True)
    native.load_state_dict(fold_and_map(original.state_dict()), strict=True)
    for layer in native.transformer.layers:
        layer.preserve_norm_weight_dtype = True
        layer.layer_scale_1.preserve_weight_dtype = True
        layer.layer_scale_2.preserve_weight_dtype = True
    def compute():
        return torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False) if bf16 else nullcontext()
    x = torch.randn(1, 12, 17)
    with compute():
        expected, _ = original(x, torch.tensor([17]))
        actual = native(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5 if not bf16 else .02, atol=5e-7 if not bf16 else .008)
    for _ in range(2):
        state = {'offset': 0, 'layers': [[], []]}
        with original.streaming(batch_size=1), compute():
            for length in (8, 16, 3):
                x = torch.randn(1, 12, length)
                expected, _ = original(x, torch.tensor([length]))
                actual = native(x, state)
                torch.testing.assert_close(actual, expected, rtol=1e-5 if not bf16 else .02, atol=5e-7 if not bf16 else .008)
        assert state['offset'] == 27
    assert not any('cache' in name for name in vars(native))


@pytest.mark.parametrize('count', [1, 4, 12])
def test_fp32_quantizer_code_ids_and_decoding_match(reference, count):
    original = reference.MossAudioTokenizerResidualLFQ(input_dim=12, rvq_dim=16, output_dim=12,
        num_quantizers=12, codebook_size=32, codebook_dim=4)
    native = ResidualLFQ(input_dim=12, dim=16, output_dim=12, count=12, size=32, code_dim=4, operations=manual_cast)
    native.load_state_dict(fold_and_map(original.state_dict()), strict=True)
    codes = torch.randint(0, 32, (2, count, 7))
    torch.testing.assert_close(native.decode(codes), original.decode_codes(codes.transpose(0, 1)), rtol=0, atol=0)
    x = torch.randn(2, 12, 7)
    expected = original(x, torch.tensor([7, 4]), n_quantizers=count)[1].transpose(0, 1)
    torch.testing.assert_close(native.encode(x, [7, 4], count), expected, rtol=0, atol=0)


def test_stereo_order_matches_original_interleave_and_restore(reference):
    owner = types.SimpleNamespace(number_channels=2, enable_channel_interleave=True, downsample_rate=3840)
    x = torch.stack((torch.arange(7680), -torch.arange(7680)), 0).float()[None]
    lengths = torch.tensor([7680])
    expected, doubled = reference.MossAudioTokenizerModel._flatten_channels_for_codec(owner, x, lengths)
    torch.testing.assert_close(MossAudioTokenizerV2.interleave(x), expected, rtol=0, atol=0)
    restored, returned = reference.MossAudioTokenizerModel._restore_channels_from_codec(owner, expected, doubled)
    torch.testing.assert_close(MossAudioTokenizerV2.restore(expected), restored, rtol=0, atol=0)
    assert torch.equal(returned, lengths)
    assert torch.equal(restored, x)


def test_complete_meta_inventory_matches_pinned_architecture(reference):
    config = reference.MossAudioTokenizerConfig.from_pretrained(os.environ['MOSS_CODEC_V2_REFERENCE'], local_files_only=True)
    with torch.device('meta'):
        original = reference.MossAudioTokenizerModel(config)
        native = MossAudioTokenizerV2(device='meta', operations=manual_cast)
    expected, actual = fold_and_map(original.state_dict()), native.state_dict()
    assert actual.keys() == expected.keys()
    assert len(actual) == 2028
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
