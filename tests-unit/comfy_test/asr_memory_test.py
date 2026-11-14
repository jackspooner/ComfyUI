"""Tiny CPU cache/admission regressions; no checkpoints or serving state."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import WhisperConfig
from transformers.generation import GenerationMixin
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration

from comfy.cli_args import args
args.cpu = True
from comfy.asr import qwen, whisper
from comfy.model_patcher import ModelPatcher, PromptModelTracker
from comfy_execution import caching


@pytest.fixture(autouse=True)
def execution():
    with torch.set_grad_enabled(False):
        yield


def whisper_decoder(dtype):
    config = WhisperConfig(vocab_size=32, num_mel_bins=8, d_model=8, encoder_layers=1, decoder_layers=2,
                           encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                           max_source_positions=4, max_target_positions=12, pad_token_id=0, bos_token_id=1,
                           eos_token_id=2, decoder_start_token_id=1, begin_suppress_tokens=[], suppress_tokens=[])
    config._attn_implementation = 'sdpa'
    reference = WhisperForConditionalGeneration(config)
    _, decoder = whisper.networks(config, dtype)
    # F32 checkpoint storage with any supported compute precision is valid.
    state = reference.model.decoder.state_dict()
    state['proj_out.weight'] = reference.proj_out.weight
    decoder.load_state_dict(state, strict=True, assign=True)
    component = SimpleNamespace(model=decoder, load_device=torch.device('cpu'))
    model = SimpleNamespace(decoder=component, dtype=dtype, config=config)
    return model, whisper.AdmittedDecoder(model)


def cache_tensors(cache):
    return [tensor for owner in (cache.self_attention_cache, cache.cross_attention_cache)
            for layer in owner.layers for tensor in (layer.keys, layer.values)]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('batch', [1, 5])
def test_whisper_admits_real_cache_storage_and_reorder_transient(dtype, batch):
    model, admitted = whisper_decoder(dtype)
    decoder, config = model.decoder.model, model.config
    encoded = torch.randn(batch, config.max_source_positions, config.d_model, dtype=dtype)
    ids = torch.ones(batch, config.max_target_positions, dtype=torch.long)
    with patch.object(whisper.mm, 'load_models_gpu') as load:
        output = admitted(ids, encoder_hidden_states=encoded)
    tensors = cache_tensors(output.past_key_values)
    assert all(tensor.dtype == dtype for tensor in tensors)
    actual_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    largest = max(tensor.numel() * tensor.element_size() for tensor in tensors)
    initial = admitted.memory_required(batch, ids.shape[1], encoded.shape[1], dtype)
    workspace = admitted.memory_required(batch, ids.shape[1], encoded.shape[1], dtype, use_cache=False)
    assert initial - workspace == actual_bytes + largest
    assert load.call_args.kwargs['memory_required'] >= initial
    old_formula = batch * config.decoder_layers * config.d_model * (config.max_target_positions + encoded.shape[1]) * 4
    assert actual_bytes == old_formula * (2 if dtype == torch.float32 else 1)

    order = torch.arange(batch - 1, -1, -1)
    snapshots = [tensor.clone() for tensor in tensors]
    output.past_key_values.reorder_cache(order)
    for before, after in zip(snapshots, cache_tensors(output.past_key_values)):
        torch.testing.assert_close(after, before[order])
    # Full caches need only transient/workspace headroom, not a second full cache.
    resident = admitted.memory_required(batch, 1, encoded.shape[1], dtype, output.past_key_values)
    resident_workspace = admitted.memory_required(batch, 1, encoded.shape[1], dtype, output.past_key_values, use_cache=False)
    assert resident - resident_workspace == largest


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_whisper_cached_step_subtracts_resident_storage(dtype):
    model, admitted = whisper_decoder(dtype)
    encoded = torch.randn(3, 4, 8, dtype=dtype)
    ids = torch.ones(3, 3, dtype=torch.long)
    with patch.object(whisper.mm, 'load_models_gpu'):
        first = admitted(ids, encoder_hidden_states=encoded)
        cache = first.past_key_values
        resident_bytes = sum(t.numel() * t.element_size() for t in cache_tensors(cache))
        required = admitted.memory_required(3, 1, 4, dtype, cache)
        workspace = admitted.memory_required(3, 1, 4, dtype, cache, use_cache=False)
        uncached = admitted.memory_required(3, 1, 4, dtype)
        uncached_workspace = admitted.memory_required(3, 1, 4, dtype, use_cache=False)
        assert (uncached - uncached_workspace) - (required - workspace) == resident_bytes
        with patch.object(model.decoder.model, 'forward', wraps=model.decoder.model.forward) as forward:
            with patch.object(whisper.mm, 'load_models_gpu', side_effect=lambda *a, **kw: assert_not_called(forward)):
                second = admitted(ids[:, :1], encoder_hidden_states=encoded, past_key_values=cache)
    assert second.past_key_values.get_seq_length() == 4
    assert second.last_hidden_state.dtype == dtype


def assert_not_called(mock):
    assert not mock.called


def test_whisper_beam_encoder_expansion_is_admitted_before_allocation():
    model, admitted = whisper_decoder(torch.float32)
    generation = object.__new__(whisper.Generation)
    torch.nn.Module.__init__(generation)
    generation.model = SimpleNamespace(decoder=admitted)
    encoded = torch.randn(2, 4, 8)
    output = whisper.BaseModelOutput(last_hidden_state=encoded)
    ids = torch.ones(2, 3, dtype=torch.long)
    calls = []
    original = GenerationMixin._expand_inputs_for_generation

    def expand(*a, **kw):
        assert calls == ['admit']
        return original(*a, **kw)

    with (patch.object(whisper.mm, 'load_models_gpu', side_effect=lambda *a, **kw: calls.append('admit')) as load,
          patch.object(GenerationMixin, '_expand_inputs_for_generation', expand)):
        expanded_ids, expanded = generation._expand_inputs_for_generation(5, True, ids, encoder_outputs=output)
    assert expanded_ids.shape[0] == expanded['encoder_outputs'].last_hidden_state.shape[0] == 10
    required = admitted.memory_required(10, 3, 4, torch.float32) + encoded.numel() * encoded.element_size() * 5
    assert load.call_args.kwargs['memory_required'] == max(256 * 1024**2, required)


class DynamicPatcher(ModelPatcher):
    """Exercise tracker/dynamic contracts without constructing accelerator state."""
    def is_dynamic(self):
        return True

    def set_in_use_by_current_prompt(self, in_use):
        self.in_use = in_use


def component(dynamic=False):
    cls = DynamicPatcher if dynamic else ModelPatcher
    return cls(torch.nn.Linear(2, 2), torch.device('cpu'), torch.device('cpu'))


def handle(kind, encoder, decoder):
    if kind == 'qwen':
        return qwen.Model(encoder, decoder, None, {}, None, False, {})
    return {'model': whisper.Model(encoder, decoder, None, {}, None, None, torch.float32)}


def pressure_cache(outputs, active=False):
    cache = caching.RAMPressureCache(caching.CacheKeySetID)
    cache.cache = {key: SimpleNamespace(outputs=value) for key, value in outputs.items()}
    cache.generation = 2
    cache.used_generation = {key: 2 if active else 1 for key in outputs}
    cache.timestamps = {key: index for index, key in enumerate(outputs)}
    cache.children = {}
    return cache


@pytest.mark.parametrize('kind', ['qwen', 'whisper'])
@pytest.mark.parametrize('which', ['encoder', 'decoder'])
def test_old_multicomponent_handle_matches_bare_model_priority(kind, which):
    patcher = component()
    wrapped = handle(kind, patcher if which == 'encoder' else object(), patcher if which == 'decoder' else object())
    for value in (patcher, wrapped):
        cache = pressure_cache({'model': [value], 'tensor': [torch.zeros(1024)]})
        with patch.object(caching, 'virtual_memory_available', side_effect=lambda: 0 if 'model' in cache.cache else 2):
            cache.ram_release(1)
        assert set(cache.cache) == {'tensor'}
    legacy = SimpleNamespace(encoder=patcher, decoder=patcher)
    cache = pressure_cache({'model': [legacy], 'tensor': [torch.zeros(1024)]})
    with patch.object(caching, 'virtual_memory_available', side_effect=lambda: 0 if 'tensor' in cache.cache else 2):
        cache.ram_release(1)
    assert set(cache.cache) == {'model'}


@pytest.mark.parametrize('kind', ['qwen', 'whisper'])
def test_active_handles_keep_dynamic_and_prompt_protection(kind):
    encoder, decoder = component(True), component(True)
    wrapped = handle(kind, encoder, decoder)
    tracker = PromptModelTracker()
    tracker.add(wrapped)
    assert len(tracker.models) == 2 and encoder.in_use and decoder.in_use
    for value in (encoder, wrapped):
        cache = pressure_cache({'model': [value]}, active=True)
        with patch.object(caching, 'virtual_memory_available', return_value=0):
            assert cache.ram_release(1, free_active=True) == 0
            assert 'model' in cache.cache
    assert encoder.in_use and decoder.in_use
    tracker.end()
    assert not encoder.in_use and not decoder.in_use


@pytest.mark.parametrize('kind', ['qwen', 'whisper'])
def test_mixed_handles_and_shared_tensor_storage_follow_existing_contract(kind):
    wrapped = handle(kind, component(True), component())
    assert not caching.all_outputs_dynamic([wrapped])
    storage = torch.zeros(1024)
    cache = pressure_cache({'mixed': [wrapped, {'views': [storage[:2], storage[-2:], storage]}]}, active=True)
    with patch.object(caching, 'virtual_memory_available', return_value=0):
        assert cache.ram_release(1) == 0
        freed = cache.ram_release(1, free_active=True)
    assert freed == caching.RAM_CACHE_DEFAULT_RAM_USAGE + storage.untyped_storage().nbytes()
    assert cache.active_evictions and cache.full_evictions and not cache.cache
    assert not caching.all_outputs_dynamic([handle(kind, component(True), storage)])
    assert not caching.all_outputs_dynamic({'model': component(True), 'name': 'metadata'})


@pytest.mark.parametrize('outputs, expected', [(None, False), ([], True), ((), True), ({}, True), ([None], False)])
def test_dynamic_discovery_preserves_empty_contract(outputs, expected):
    assert caching.all_outputs_dynamic(outputs) is expected


@pytest.mark.parametrize('dynamic', [False, True])
def test_explicit_dynamic_capability_precedes_tensor_discovery(dynamic):
    output = SimpleNamespace(is_dynamic=lambda: dynamic,
                             _comfy_cache_tensors=lambda: pytest.fail('explicit capability owns this decision'))
    assert caching.all_outputs_dynamic([output]) is dynamic


def test_bad_discovery_hook_keeps_sizing_failure_recovery(caplog):
    class Broken:
        def _comfy_cache_tensors(self):
            raise ValueError('bad cache hook')
    cache = pressure_cache({'broken': [Broken()]}, active=True)
    with patch.object(caching, 'virtual_memory_available', return_value=0):
        assert cache.ram_release(1, free_active=True) > 0
    assert not cache.cache
    assert 'Could not size cache entry broken; evicting it first' in caplog.text


@pytest.mark.parametrize('corruption', [None, 'start', 'language', 'task', 'timestamp', 'out-of-range'])
def test_whisper_required_token_ids_before_network_construction(corruption):
    vocab = {'<|startoftranscript|>': 1, '<|endoftext|>': 2, '<|nospeech|>': 5,
             '<|notimestamps|>': 6, '<|0.00|>': 7, '<|startofprev|>': 8, '<|en|>': 9,
             '<|transcribe|>': 10, '<unk>': 0}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    config = {'vocab_size': 16, 'decoder_start_token_id': 1, 'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0}
    generation = {'decoder_start_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 2,
                  'no_timestamps_token_id': 6, 'prev_sot_token_id': 8,
                  'lang_to_id': {'<|en|>': 9}, 'task_to_id': {'transcribe': 10}}
    if corruption == 'start':
        config['decoder_start_token_id'] = 3
    elif corruption == 'language':
        generation['lang_to_id']['<|en|>'] = 10
    elif corruption == 'task':
        generation['task_to_id']['transcribe'] = 9
    elif corruption == 'timestamp':
        generation['no_timestamps_token_id'] = 7
    elif corruption == 'out-of-range':
        config['vocab_size'] = 9
    metadata = {'config': json.dumps(config), 'generation_config': json.dumps(generation), 'tokenizer_config': '{}'}
    state = {'tokenizer_json': torch.tensor(list(tokenizer.to_str().encode()), dtype=torch.uint8)}
    with (patch.object(whisper, 'package', return_value=(metadata, state)),
          patch.object(whisper, 'networks', side_effect=RuntimeError('network boundary reached')) as networks):
        if corruption is None:
            with pytest.raises(RuntimeError, match='network boundary reached'):
                whisper.load('unused')
            networks.assert_called_once()
        else:
            with pytest.raises(ValueError, match='Unexpected native Whisper tokenizer ID'):
                whisper.load('unused')
            networks.assert_not_called()
