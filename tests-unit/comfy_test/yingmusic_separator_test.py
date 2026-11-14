"""Pinned preparation/component fixtures only; no full-checkpoint inference."""
import ast
import hashlib
import importlib.util
from functools import partial
import os
from pathlib import Path
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from comfy.cli_args import args
args.cpu = True
from comfy import yingmusic_separator as native
from scripts.convert_yingmusic import SEPARATOR_SOURCE_FILES


@pytest.fixture(scope='module')
def reference():
    source = os.environ.get('YINGMUSIC_SEPARATOR_REFERENCE')
    if not source:
        pytest.skip('Requires pinned local YINGMUSIC_SEPARATOR_REFERENCE; no weights are loaded')
    root = Path(source)/'src/third_party/MusicSourceSeparationTraining'
    for name, digest in SEPARATOR_SOURCE_FILES.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    # Execute only the original pure preparation functions, avoiding unrelated
    # training/optimizer dependencies and preserving their actual implementation.
    parsed = ast.parse((root/'utils/model_utils.py').read_text())
    selected = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)
        and node.name in ('demix', '_getWindowingArray', 'prefer_target_instrument')], type_ignores=[])
    namespace = dict(np=np, torch=torch, nn=nn, dist=types.SimpleNamespace(is_initialized=lambda: False))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *selected.body], type_ignores=[])), str(root/'utils/model_utils.py'), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


class Config(dict):
    __getattr__ = dict.__getitem__


class FakeSeparator(nn.Module):
    dtype = torch.float32

    def forward(self, value, options=None):
        # Nonconstant edge-sensitive response exposes misplaced padding/fades.
        return (value*.3 + torch.roll(value, 7, -1)*.1)[:, None]


@pytest.mark.parametrize('length', [1, 19, 40, 41, 79, 80, 81, 119, 240, 321])
def test_exact_reference_batches_padding_and_edge_windows(reference, length):
    waveform = torch.randn(2, length, generator=torch.Generator().manual_seed(7401))
    config = Config(audio=Config(chunk_size=80), inference=Config(num_overlap=2, batch_size=4),
        training=Config(target_instrument='vocals', instruments=['vocals', 'other'], use_amp=False))
    expected = reference.demix(config, FakeSeparator(), waveform.numpy(), 'cpu', 'mel_band_roformer')['vocals']
    actual = native.demix(FakeSeparator(), waveform, torch.device('cpu'), {}, chunk=80)
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0, atol=0)


def test_stereo_residual_and_native_admission_cleanup():
    component = types.SimpleNamespace(metadata={})
    waveform = torch.randn(2, 2, 42)
    def demix(model, item, device, options):
        return item*.25
    with patch.object(native, 'validate_metadata'), patch.object(native.runtime, 'activate',
            return_value=(FakeSeparator(), torch.device('cpu'), torch.float32)) as activate, \
         patch.object(native.runtime, 'transformer_options', return_value={}), patch.object(native, 'demix', side_effect=demix), \
         patch.object(native.comfy.model_prefetch, 'cleanup_prefetch_queues') as cleanup:
        vocal, residual = native.separate(component, {'waveform': waveform, 'sample_rate': 44100})
        activate.assert_called_once()
        cleanup.assert_called_once()
    torch.testing.assert_close(vocal['waveform']+residual['waveform'], waveform)
    assert vocal['sample_rate'] == residual['sample_rate'] == 44100


def test_cancelled_preparation_does_not_publish_partial_audio():
    with patch.object(native.mm, 'throw_exception_if_processing_interrupted',
            side_effect=native.mm.InterruptProcessingException):
        with pytest.raises(native.mm.InterruptProcessingException):
            native.demix(FakeSeparator(), torch.zeros(2, 81), torch.device('cpu'), {}, chunk=80)


def test_cpu_roformer_does_not_dispatch_the_global_cuda_attention_backend():
    from comfy import ops
    from comfy.soulx_singer import separation
    from comfy.ldm.modules import attention
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(7410)
        model = separation.Attention(16, 2, 8, torch.float32, 'cpu', ops.manual_cast)
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=.05)
        x = torch.randn(2, 11, 16)
        position = torch.arange(11, dtype=torch.float32)
        normalized = model.norm(x)
        q, k, v = model.to_qkv(normalized).reshape(2, 11, 3, 2, 8).permute(2, 0, 3, 1, 4).unbind(0)
        phase = torch.outer(position, model.rotary_embed.freqs)
        matrix = torch.stack((phase.cos(), -phase.sin(), phase.sin(), phase.cos()), -1).unflatten(-1, (2, 2))[None, None]
        q, k = separation.apply_rope(q, k, matrix)
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, 11, 16)
        expected = model.to_out(attended*model.to_gates(normalized).sigmoid().repeat_interleave(8, -1))
        with patch.object(attention, 'optimized_attention', side_effect=AssertionError('CUDA-only global backend')), \
             patch.object(separation, 'optimized_attention', side_effect=AssertionError('CUDA-only imported backend'), create=True):
            actual = model(x, position)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_small_complete_roformer_matches_pinned_source_including_dc_filter(reference):
    import einops
    import librosa
    from comfy.ops import manual_cast
    from comfy.soulx_singer.separation import MelBandRoformer
    rotary_path = os.environ.get('YINGMUSIC_ROTARY_REFERENCE')
    if not rotary_path:
        pytest.skip('Requires the pinned rotary-embedding-torch 0.8.9 source file; no environment changes')
    path = Path(rotary_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '37c7560fd47121ff1bd51488f46e10cb399d62e6e439e6b18732902f526f59b6'
    spec = importlib.util.spec_from_file_location('ying_rotary_reference', path)
    rotary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rotary)
    root = Path(os.environ['YINGMUSIC_SEPARATOR_REFERENCE'])/'src/third_party/MusicSourceSeparationTraining/models/bs_roformer'
    attention = {}
    exec(compile((root/'attend.py').read_text(), str(root/'attend.py'), 'exec'), attention)
    tree = ast.parse((root/'mel_band_roformer.py').read_text())
    # Preserve released numerical bodies; annotations/decorators are type checks
    # only, with the genuine pinned rotary and attention implementations.
    statements = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    namespace = dict(torch=torch, F=F, nn=nn, Module=nn.Module, ModuleList=nn.ModuleList,
        tensor=torch.tensor, partial=partial, filters=librosa.filters, beartype=lambda value: value,
        Attend=attention['Attend'], RotaryEmbedding=rotary.RotaryEmbedding, _HAS_POPE=False)
    namespace.update({name: getattr(einops, name) for name in ('pack', 'unpack', 'rearrange', 'reduce', 'repeat')})
    compiled = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *statements], type_ignores=[])
    exec(compile(ast.fix_missing_locations(compiled), str(root/'mel_band_roformer.py'), 'exec'), namespace)
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(7401)
        original = namespace['MelBandRoformer'](dim=32, depth=1, stereo=True, num_stems=1,
            time_transformer_depth=1, freq_transformer_depth=1, heads=4, dim_head=8,
            attn_dropout=0., ff_dropout=0., stft_hop_length=441, mask_estimator_depth=2,
            flash_attn=False, zero_dc=True).eval()
        bank = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=60)
        bank[0, 0] = bank[-1, -1] = 1
        bands = [np.flatnonzero(row > 0).tolist() for row in bank]
        candidate = MelBandRoformer(bands, stems=1, dim=32, depth=1, heads=4, head_dim=8,
            dtype=torch.float32, device='cpu', operations=manual_cast, zero_dc=True)
        state = {**original.state_dict(), 'freq_indices': candidate.freq_indices,
                 'num_bands_per_freq': candidate.num_bands_per_freq}
        candidate.load_state_dict(state, strict=True)
        audio = torch.randn(2, 2, 4410)*.1+.7
        expected = original(audio)
        actual = candidate(audio)[:, 0]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-6)
        candidate.zero_dc = False
        assert (candidate(audio)[:, 0]-expected).abs().max() > 1e-4
