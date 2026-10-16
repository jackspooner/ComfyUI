import hashlib
import importlib
import os
from pathlib import Path
import sys
import types

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.audio.moss_audio_tokenizer import CodecAttention, CodecTransformerLayer, ProjectedTransformer, Patch, ResidualLFQ, MossAudioTokenizer, rotary_matrix


@pytest.fixture(scope='module')
def reference():
    source = os.environ.get('MOSS_CODEC_REFERENCE')
    if not source:
        pytest.skip('Requires pinned MOSS codec source and isolated Transformers 5 overlay')
    source = Path(source)
    digest = hashlib.sha256((source / 'modeling_moss_audio_tokenizer.py').read_bytes()).hexdigest()
    assert digest == '65cae7744845f1b8ac65957e918cea508efe331a38e87b882b7530b6c8d7caa5'
    package = types.ModuleType('moss_codec_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    return importlib.import_module('moss_codec_reference.modeling_moss_audio_tokenizer')


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.is_grad_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        torch.set_grad_enabled(False)
        yield
    torch.set_grad_enabled(previous)


@pytest.mark.parametrize('length,offset', [(1, 0), (7, 0), (3, 19)])
def test_rotary_matches_interleaved_reference(reference, length, offset):
    q = torch.randn(2, 3, length, 8)
    k = torch.randn_like(q)
    expected = reference.apply_rope(q, k, torch.full((2,), offset))
    actual = apply_rope(q, k, rotary_matrix(length, 8, offset, q.device))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=5e-7)


def test_attention_matches_full_and_ring_streaming(reference):
    ref = reference.MossAudioTokenizerMultiheadAttention(16, 2, causal=True, context=10,
                                                       rope=reference.MossAudioTokenizerRotaryEmbedding())
    native = CodecAttention(16, 2, 10, operations=manual_cast)
    native.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 23, 16)
    torch.testing.assert_close(native(x, rotary_matrix(23, 8, 0, x.device)), ref(x, x, x), rtol=1e-5, atol=5e-7)
    cache = []
    offset = 0
    with ref.streaming(batch_size=2):
        for chunk in x.split(8, dim=1):
            actual = native(chunk, rotary_matrix(chunk.shape[1], 8, offset, x.device), offset, cache)
            torch.testing.assert_close(actual, ref(chunk, chunk, chunk), rtol=1e-5, atol=5e-7)
            offset += chunk.shape[1]
    assert not any('cache' in name for name in vars(native))


def test_transformer_layer_matches_reference(reference):
    ref = reference.MossAudioTokenizerTransformerLayer(16, 2, 40, causal=True, context=10,
                                                      rope=reference.MossAudioTokenizerRotaryEmbedding(), layer_scale=0.01)
    native = CodecTransformerLayer(16, 2, 40, 10, operations=manual_cast)
    native.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 7, 16)
    torch.testing.assert_close(native(x, rotary_matrix(7, 8, 0, x.device)), ref(x), rtol=1e-5, atol=5e-7)


def test_projected_stack_streaming_matches_reference(reference):
    ref = reference.MossAudioTokenizerProjectedTransformer(12, 8, 16, conv_layout=True, module_type='Transformer',
              num_heads=2, num_layers=2, dim_feedforward=40, causal=True, context=10,
              positional_embedding='rope', norm='layer_norm', layer_scale=0.01)
    native = ProjectedTransformer(12, 8, 16, 2, 40, 2, 10, operations=manual_cast)
    native.load_state_dict(ref.state_dict(), strict=True)
    state = {'offset': 0, 'layers': [[], []]}
    with ref.streaming(batch_size=1):
        for length in [8, 8, 3]:
            x = torch.randn(1, 12, length)
            expected, _ = ref(x, torch.tensor([length]))
            torch.testing.assert_close(native(x, state), expected, rtol=1e-5, atol=5e-7)
    assert state['offset'] == 19


@pytest.mark.parametrize('size', [2, 240])
def test_patch_layout_matches_reference(reference, size):
    x = torch.randn(2, 3, size * 5)
    ref = reference.MossAudioTokenizerPatchedPretransform(size, True, 'PatchedPretransform')
    expected, _ = ref(x, torch.tensor([size * 5] * 2))
    actual = Patch(size, True)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(Patch(size, False)(actual), x, rtol=0, atol=0)


def fold_weight_norm(state):
    result = {}
    for key, value in state.items():
        if key.endswith('.parametrizations.weight.original0'):
            prefix = key.removesuffix('.parametrizations.weight.original0')
            result[prefix + '.weight'] = torch._weight_norm(state[prefix + '.parametrizations.weight.original1'], value, 0)
        elif not key.endswith('.parametrizations.weight.original1'):
            result[key] = value
    return result


@pytest.mark.parametrize('count', [1, 2, 4])
def test_residual_codebook_matches_fp32_reference(reference, count):
    ref = reference.MossAudioTokenizerResidualLFQ(input_dim=12, rvq_dim=16, output_dim=12,
                                                 num_quantizers=4, codebook_size=32, codebook_dim=4)
    native = ResidualLFQ(input_dim=12, dim=16, output_dim=12, count=4, size=32, code_dim=4, operations=manual_cast)
    native.load_state_dict(fold_weight_norm(ref.state_dict()), strict=True)
    codes = torch.randint(0, 32, (2, count, 7))
    torch.testing.assert_close(native.decode(codes), ref.decode_codes(codes.transpose(0, 1)), rtol=0, atol=0)
    x = torch.randn(2, 12, 7)
    expected = ref(x, torch.tensor([7, 4]), n_quantizers=count)[1].transpose(0, 1)
    torch.testing.assert_close(native.encode(x, [7, 4], count), expected, rtol=0, atol=0)


def test_codec_state_inventory_matches_folded_checkpoint(reference):
    source = Path(os.environ['MOSS_CODEC_REFERENCE'])
    config = reference.MossAudioTokenizerConfig.from_pretrained(str(source), local_files_only=True)
    with torch.device('meta'):
        ref = reference.MossAudioTokenizerModel(config)
        native = MossAudioTokenizer(device='meta', operations=manual_cast)
    expected = fold_weight_norm(ref.state_dict())
    actual = native.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
