import asyncio
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from comfy.cli_args import args
args.cpu = True
import comfy.ops
from comfy.fish_audio import history
from comfy.fish_audio.config import FishOmniTextConfig, AudioDecoderConfig, FishOmniConfig
from comfy.fish_audio.model import FullAttention, DeltaNet, SparseMoE
from comfy.fish_audio.tower import Tower, sample
from comfy.fish_audio.prompt import split_drama3_text
from comfy_extras import nodes_drama3


def config():
    text = FishOmniTextConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'],
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, rope_theta=10000., partial_rotary_factor=.5,
        rms_norm_eps=1e-6, max_position_embeddings=256, linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=8, shared_expert_intermediate_size=8)
    audio = AudioDecoderConfig(dim=16, intermediate_size=24, n_head=2, n_local_heads=1, head_dim=8, norm_eps=1e-6,
        rope_base=10000., rotary_percent=.5, vocab_size=16, num_codebooks=3, shared_enrichment_layers=2, vq_backbone_codebooks=1)
    return FishOmniConfig(text=text, audio_decoder=audio, semantic_start_token_id=32, semantic_end_token_id=47, eos_token_id=2)


def initialize(model):
    for p in model.parameters():
        p.data.copy_(torch.randn_like(p) * .1)
    return model


def test_history_roundtrip_and_branch(tmp_path):
    codes = torch.arange(30).view(10, 3)
    ref = history.Reference('hello', 0, codes.clone(), 'fixture')
    first = history.History('fixture', (ref,), (history.Take('hello', (('hello', codes),), {'seed': 3}),))
    path = tmp_path / 'take.drama3.safetensors'
    history.save(first, path)
    restored = history.load(path)
    assert restored.compatibility == first.compatibility
    assert torch.equal(restored.takes[0].chunks[0][1], codes)
    restored.takes[0].chunks[0][1].zero_()
    assert torch.equal(first.takes[0].chunks[0][1], codes)
    with pytest.raises(FileExistsError):
        history.save(first, path)
    with pytest.raises(ValueError, match='does not match'):
        history.validate(first, 'different')
    bad = codes.clone(); bad[2, 0] = 1024
    with pytest.raises(ValueError, match='ranges'):
        history.validate_codes(bad, reference=True)
    history.validate_codes(bad)


def test_history_rejects_unknown_version_and_tensor(tmp_path):
    from safetensors.torch import save_file
    p = tmp_path / 'bad.safetensors'
    save_file({}, str(p), metadata={'format': 'drama3_history', 'history': json.dumps({'version': 2})})
    with pytest.raises(ValueError, match='version'):
        history.load(p)
    data = dict(version=1, compatibility='fixture', references=[dict(transcript='hello', speaker=0, tensor=[])], takes=[])
    save_file({}, str(p), metadata={'format': 'drama3_history', 'history': json.dumps(data)})
    with pytest.raises(ValueError, match='tensor name'):
        history.load(p)


def test_loader_rejects_paths_outside_admitted_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes_drama3.folder_paths, 'get_input_directory', lambda: str(tmp_path / 'input'))
    monkeypatch.setattr(nodes_drama3.folder_paths, 'get_output_directory', lambda: str(tmp_path / 'output'))
    with pytest.raises(ValueError, match='admitted'):
        nodes_drama3.Drama3LoadHistory.execute('../../elsewhere.safetensors [output]')


def test_schema_has_five_nodes_and_optional_references():
    nodes = asyncio.run(nodes_drama3.Drama3Extension().get_node_list())
    assert len(nodes) == 5
    for node in nodes:
        assert node.define_schema().node_id == node.__name__
    schema = nodes_drama3.Drama3Generate.define_schema()
    assert next(i for i in schema.inputs if i.id == 'references').optional


def test_chunker_preserves_protected_tags_and_speakers():
    text = '<|speaker:0|> [whisper] ' + 'Hello there. ' * 50 + '<|speaker:1|> Welcome.'
    chunks = split_drama3_text(text)
    assert len(chunks) > 1
    assert '[whisper]' in chunks[0]
    assert '<|speaker:1|>' in chunks[-1]


def test_sampling_is_local_and_seeded():
    logits = torch.arange(16).float()[None]
    global_state = torch.random.get_rng_state().clone()
    a = sample(logits, .7, 3, .8, torch.Generator().manual_seed(8))
    b = sample(logits, .7, 3, .8, torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert torch.equal(global_state, torch.random.get_rng_state())
    with pytest.raises(FloatingPointError):
        sample(logits * float('nan'), .7, 3, .8, torch.Generator())


@pytest.mark.parametrize('kind', ['full', 'delta'])
def test_attention_cached_chunks_equal_full(kind):
    torch.manual_seed(4)
    c = config().text
    cls = FullAttention if kind == 'full' else DeltaNet
    model = initialize(cls(c, torch.float32, 'cpu', comfy.ops.manual_cast))
    x = torch.randn(1, 7, 16)
    expected = model(x, 0, [], 12)
    cache = []
    actual = torch.cat([model(x[:, :3], 0, cache, 12), model(x[:, 3:6], 3, cache, 12), model(x[:, 6:], 6, cache, 12)], 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def upstream_module(name):
    source = os.environ.get('DRAMA3_UPSTREAM')
    if not source:
        pytest.skip('Set DRAMA3_UPSTREAM to the pinned offline Fish Speech source')
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module('fish_speech.models.text2semantic.fish_omni.' + name)


@pytest.mark.parametrize('kind', ['full', 'delta'])
def test_upstream_attention_parity(kind):
    upstream = upstream_module('layers')
    torch.manual_seed(31)
    c = config().text
    cls = FullAttention if kind == 'full' else DeltaNet
    other = upstream.FullAttention if kind == 'full' else upstream.GatedDeltaNet
    ref = initialize(other(c, dtype=torch.float32, device='cpu'))
    model = cls(c, torch.float32, 'cpu', comfy.ops.manual_cast)
    model.load_state_dict(ref.state_dict())
    x = torch.randn(1, 7, 16)
    inv = c.rope_theta ** (-torch.arange(0, c.rotary_dim, 2).float() / c.rotary_dim)
    phase = torch.arange(7)[:, None] * inv
    phase = torch.cat((phase, phase), -1)
    context = upstream.LayerContext(input_pos=torch.arange(7), cos=phase.cos(), sin=phase.sin(), kv_len=7, continue_state=False, use_cache=False)
    expected = ref(x, context)
    actual = model(x, 0, [], 16)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_upstream_moe_parity():
    upstream = upstream_module('moe')
    c = config().text
    ref = initialize(upstream.SparseMoeBlock(c, fp8=False, dtype=torch.float32, device='cpu'))
    model = SparseMoE(c, torch.float32, 'cpu', comfy.ops.mixed_precision_ops(compute_dtype=torch.float32))
    state = ref.state_dict()
    state['experts.gate_up_proj.weight'] = state.pop('experts.w13_weight')
    state['experts.down_proj.weight'] = state.pop('experts.w2_weight')
    model.load_state_dict(state, strict=True, assign=True)
    x = torch.randn(1, 7, 16)
    torch.testing.assert_close(model(x), ref(x), atol=2e-6, rtol=2e-5)


def test_upstream_tower_teacher_force_and_decode_parity():
    upstream = upstream_module('tower')
    c = config()
    torch.manual_seed(38)
    ref = initialize(upstream.FishOmniAudioDecoder(c, dtype=torch.float32, device='cpu'))
    model = Tower(c, torch.float32, 'cpu', comfy.ops.manual_cast)
    model.load_state_dict(ref.state_dict())
    frames = torch.zeros(5, 4, dtype=torch.int64)
    frames[:, 0] = torch.tensor([3, 33, 35, 37, 39])
    frames[:, 1:] = torch.randint(0, 16, (5, 3))
    hidden = torch.randn(5, 16)
    embedded = torch.randn(5, 16)
    torch.testing.assert_close(model.embed_backbone(frames, embedded), ref.embed_backbone(frames, embedded))
    ref.setup_caches(12)
    caches = ([[] for _ in model.shared_layers], [[] for _ in model.residual_layers])
    ref.prefill(hidden[:4], torch.arange(4), frames[:4], frames[1:])
    model.prefill(hidden[:4], 0, frames[:4], frames[1:], caches, 12)
    for source, cache in zip((*ref.shared_layers, *ref.residual_layers), (*caches[0], *caches[1])):
        torch.testing.assert_close(cache[0][:, :, :4], source.transformer_layer.attention.k_cache[:, :, :4], atol=2e-6, rtol=2e-5)
    logits = torch.randn(1, 64)
    torch.manual_seed(44)
    expected = ref.decode_step(logits, hidden[-1:], torch.tensor([4]), frames[-1:], frames[:, 0], temperature=.7, top_k=10, top_p=.8)
    actual = model.decode(logits, hidden[-1:], 4, frames[-1:], frames[:, 0], caches, 12, (.7, 10, .8), torch.Generator().manual_seed(44))
    assert torch.equal(actual, expected)


def test_native_codec_covers_checkpoint_shapes():
    from comfy.fish_audio.codec import Codec
    with torch.device('meta'):
        model = Codec(torch.bfloat16, 'meta', comfy.ops.manual_cast)
    assert model.state_dict()['quantizer.semantic_quantizer.quantizers.0.codebook.weight'].shape == (4096, 8)
    assert not tuple(model.named_buffers())


def test_converter_checks_fp8_metadata_and_writer_completeness(tmp_path):
    from scripts.convert_drama3 import decode_fp8, Writer
    with pytest.raises(ValueError, match='Invalid block'):
        decode_fp8(torch.ones(128, 128), torch.ones(1, 1))
    w = Writer(tmp_path / 'partial.safetensors', {'a': ((3,), torch.float32)}, {})
    try:
        with pytest.raises(ValueError, match='completely'):
            w.finish({})
        assert not (tmp_path / 'partial.safetensors').exists()
    finally:
        w.abort()


def test_converter_refuses_incomplete_or_corrupt_pinned_source(tmp_path, monkeypatch):
    from scripts import convert_drama3 as converter
    payload = tmp_path / 'config.json'
    payload.write_bytes(b'{}')
    lock = {'revision': converter.SOURCE_REVISION, 'files': {'config.json': {'size': 3}}}
    monkeypatch.setattr(converter, 'read_json', lambda _: lock)
    with pytest.raises(ValueError, match='Incomplete source'):
        converter.verify_source(tmp_path, tmp_path / 'codec.pth')
    lock['files']['config.json'] = {'size': 2, 'blobId': '0' * 40}
    with pytest.raises(ValueError, match='hash mismatch'):
        converter.verify_source(tmp_path, tmp_path / 'codec.pth')


def test_writer_verifies_and_publishes(tmp_path):
    from scripts.convert_drama3 import Writer
    from safetensors.torch import load_file
    p = tmp_path / 'complete.safetensors'
    w = Writer(p, {'a': ((2, 3), torch.float32)}, {'test': 'fixture'})
    try:
        w.put('a', torch.ones(1, 3), 0)
        w.put('a', torch.zeros(1, 3), 1)
        manifest = w.finish({'fixture': True})
    finally:
        w.abort()
    assert len(manifest['output_sha256']) == 64
    assert load_file(p)['a'].tolist() == [[1., 1., 1.], [0., 0., 0.]]


def test_manifest_failure_never_publishes_model(tmp_path, monkeypatch):
    from scripts import convert_drama3 as converter
    path = tmp_path / 'unpublished.safetensors'
    writer = converter.Writer(path, {'a': ((1,), torch.float32)}, {})
    writer.put('a', torch.ones(1))
    def fail(*args, **kwargs):
        raise OSError('fixture manifest failure')
    monkeypatch.setattr(converter.json, 'dump', fail)
    try:
        with pytest.raises(OSError, match='manifest failure'):
            writer.finish({})
    finally:
        writer.abort()
    assert not path.exists() and not path.with_suffix('.manifest.json').exists()
    assert not list(tmp_path.glob('.*'))


def test_tiny_fp8_conversion_loads_native_expert_banks(tmp_path):
    from dataclasses import asdict
    from safetensors.torch import save_file, load_file
    from scripts.convert_drama3 import convert, InventoryOperations
    from comfy.fish_audio.runtime import GenerationModel
    c = config()
    c.text.hidden_size = 128
    c.text.num_hidden_layers = 1
    c.text.layer_types = ['full_attention']
    c.text.moe_intermediate_size = 128
    c.text.shared_expert_intermediate_size = 128
    c.text.num_experts = 2
    c.text.num_experts_per_tok = 1
    from dataclasses import replace
    c.audio_decoder = replace(c.audio_decoder, dim=128)
    t = asdict(c.text)
    t.update(model_type='fish_omni_text', rope_parameters={'rope_theta': t['rope_theta'], 'partial_rotary_factor': t['partial_rotary_factor']})
    a = asdict(c.audio_decoder)
    a.update(n_layer=a['shared_enrichment_layers'], text_dim=a['dim'], vq_feature_placement='backbone_partial')
    raw = {'model_type': 'fish_omni', 'speech_template_version': 'reference-bracket-v1', 'text_config': t,
           'audio_decoder_config': a, 'semantic_start_token_id': 32, 'semantic_end_token_id': 47, 'eos_token_id': 2,
           'quantization_config': {'quant_method': 'bsgemm-moe', 'weight_block_size': [128, 128]}}
    model = initialize(GenerationModel(c, torch.bfloat16, 'cpu', InventoryOperations))
    source = {}
    for key, value in model.named_parameters():
        name = key.replace('backbone.', 'model.language_model.', 1)
        for before, after in [('tower.codebook_embeddings.', 'audio_decoder.codebook_embeddings.'),
                              ('tower.shared_layers.', 'audio_decoder_mtp_shared.layers.'),
                              ('tower.residual_layers.', 'audio_decoder_mtp.layers.'), ('tower.codebook_heads.', 'audio_decoder_mtp.heads.')]:
            if key.startswith(before):
                name = after + key[len(before):]
        source[name] = value.detach().contiguous()
    for e in range(2):
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            base = f'model.language_model.layers.0.mlp.experts.{e}.{projection}'
            source[base + '.weight'] = torch.randn(128, 128).to(torch.float8_e4m3fn)
            source[base + '.weight_scale_inv'] = torch.full((1, 1), .02)
    save_file(source, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'model.safetensors' for k in source}}))
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    (tmp_path / 'tokenizer.json').write_text('{}')
    (tmp_path / 'LICENSE').write_text('fixture')
    output = tmp_path / 'converted.safetensors'
    convert(tmp_path, output, 'fixture-codec')
    restored = GenerationModel(c, torch.bfloat16, 'cpu', comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=True))
    restored.load_state_dict(load_file(output), strict=True, assign=True)
    bank = restored.backbone.layers[0].mlp.experts.gate_up_proj
    assert bank.weight.shape == (2, 256, 128)
    assert restored.lm_head.weight.device.type == 'cpu'
    assert restored.backbone.layers[0].self_attn.q_proj.weight.shape == (32, 128)
    assert bank.quant_format == 'nvfp4'
    assert bank.weight._qdata.device.type == 'cpu'
    x = torch.ones(1, 128, dtype=torch.bfloat16)
    with bank.bank_resident(x) as resident:
        assert torch.isfinite(resident.expert_linear(x, 0)).all()
    frames = torch.zeros(2, c.audio_decoder.num_codebooks + 1, dtype=torch.long)
    hidden = restored.backbone(restored.embed(frames)[None], 0, [[]], 8)
    assert hidden.shape == (1, 2, 128) and torch.isfinite(hidden).all()


def fixture_tokenizer(c):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    vocab = {'[UNK]': 0, '<|im_end|>': c.eos_token_id}
    vocab.update({f'<|semantic:{i}|>': c.semantic_start_token_id + i for i in range(c.audio_decoder.vocab_size)})
    for i in range(c.text.vocab_size):
        if i not in vocab.values():
            vocab[f'filler{i}'] = i
    return Tokenizer(WordLevel(vocab, unk_token='[UNK]'))


def test_prompt_rows_match_independent_upstream_and_overflow():
    from comfy.fish_audio.prompt import build_drama3_prompt
    c = config()
    tokenizer = fixture_tokenizer(c)
    codes = torch.randint(0, 16, (3, 4))
    old = [('Original words.', codes)]
    expected = upstream_module('generate').build_drama3_prompt('Changed words.', tokenizer, c, ['Reference words.'], [codes], old)
    actual = build_drama3_prompt('Changed words.', tokenizer, c, ['Reference words.'], [codes], old)
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match='prompt has'):
        build_drama3_prompt('hello', tokenizer, c, ['reference'], [torch.zeros(3, 8193, dtype=torch.long)])


def test_cancellation_keeps_input_history_and_cleans_prefetch(monkeypatch):
    from dataclasses import replace
    from comfy.fish_audio import runtime
    c = config()
    c.audio_decoder = replace(c.audio_decoder, num_codebooks=10)
    tokenizer = fixture_tokenizer(c)
    before = history.History('fixture', (), (history.Take('earlier', (('earlier', torch.zeros(10, 2, dtype=torch.long)),), {}),))
    def interrupted(*a, **kw):
        raise RuntimeError('fixture cancellation')
    patcher = SimpleNamespace(load_device=torch.device('cpu'), model=SimpleNamespace(generate=interrupted))
    model = SimpleNamespace(compatibility='fixture', config=c, tokenizer=tokenizer, generation=patcher, provenance={})
    cleaned = []
    from contextlib import nullcontext
    monkeypatch.setattr(runtime.mm, 'load_models_gpu', lambda *a, **kw: None)
    monkeypatch.setattr(runtime.prefetch, 'malloc_graph_scope', lambda device: nullcontext())
    monkeypatch.setattr(runtime.prefetch, 'cleanup_prefetch_queues', lambda: cleaned.append(True))
    with pytest.raises(RuntimeError, match='cancellation'):
        runtime.synthesize(model, 'revised', (), before, 1, .7, .7, 30, 20, True)
    assert len(before.takes) == 1 and before.takes[0].chunks[0][1].sum() == 0
    assert cleaned == [True]


@pytest.mark.skipif(os.environ.get('DRAMA3_GPU_TESTS') != '1', reason='Requires coordinated exclusive CUDA admission')
def test_blackwell_delta_fused_decode_matches_source():
    upstream = upstream_module('layers')
    from comfy_kitchen.gated_delta import is_available
    c = config().text
    c.linear_key_head_dim = c.linear_value_head_dim = 128
    assert is_available(torch.device('cuda'), 128, 128)
    torch.manual_seed(91)
    reference = initialize(upstream.GatedDeltaNet(c, dtype=torch.bfloat16, device='cuda'))
    native = DeltaNet(c, torch.bfloat16, 'cuda', comfy.ops.manual_cast)
    native.load_state_dict(reference.state_dict())
    reference.allocate_cache(1, torch.bfloat16, 'cuda')
    x = torch.randn(1, 5, c.hidden_size, dtype=torch.bfloat16, device='cuda')
    cache = []
    for start, end in ((0, 4), (4, 5)):
        context = upstream.LayerContext(input_pos=torch.arange(start, end, device='cuda'), cos=None, sin=None,
                                       kv_len=end, continue_state=start > 0, use_cache=True)
        expected = reference(x[:, start:end], context)
        actual = native(x[:, start:end], start, cache, 12)
        torch.testing.assert_close(actual, expected, atol=.005, rtol=.05)


@pytest.mark.skipif(os.environ.get('DRAMA3_GPU_TESTS') != '1', reason='Requires coordinated exclusive CUDA admission')
@pytest.mark.parametrize('cached_layers', [0, 1])
def test_blackwell_streamed_reference_matches_original_backbone(cached_layers):
    from scripts.drama3_reference import install_streaming_backbone
    upstream = upstream_module('backbone')
    context_type = upstream_module('layers').LayerContext
    c = config().text
    c.hidden_size = c.moe_intermediate_size = c.shared_expert_intermediate_size = 128
    c.linear_key_head_dim = c.linear_value_head_dim = 128
    c.num_experts = 2
    c.num_experts_per_tok = 1
    c.num_hidden_layers = 3
    c.layer_types = ['linear_attention', 'full_attention', 'linear_attention']
    original = initialize(upstream.FishOmniBackbone(c, fp8_experts=True, dtype=torch.bfloat16, device='cuda'))
    source = {}
    for key, value in original.state_dict().items():
        if '.mlp.experts.' in key:
            continue
        if key.startswith('model.layers.'):
            source[key.replace('model.', 'model.language_model.', 1)] = value.cpu().clone()
    for i, layer in enumerate(original.model.layers):
        bank = layer.mlp.experts
        for name in ('w13', 'w2'):
            weight = getattr(bank, name + '_weight')
            weight.copy_(torch.randn(weight.shape, device='cuda').to(weight.dtype))
            getattr(bank, name + '_scale_inv').fill_(.01)
        for e in range(c.num_experts):
            for proj, weight, scales in [('gate_proj', bank.w13_weight[e, :128], bank.w13_scale_inv[e, :1]),
                                          ('up_proj', bank.w13_weight[e, 128:], bank.w13_scale_inv[e, 1:]),
                                          ('down_proj', bank.w2_weight[e], bank.w2_scale_inv[e])]:
                base = f'model.language_model.layers.{i}.mlp.experts.{e}.{proj}'
                source[base + '.weight'] = weight.cpu().clone()
                source[base + '.weight_scale_inv'] = scales.cpu().clone()
    with torch.device('meta'):
        streamed = upstream.FishOmniBackbone(c, fp8_experts=True, dtype=torch.bfloat16)
    streamed.load_state_dict({k: v for k, v in original.state_dict().items() if not k.startswith('model.layers.')}, strict=False, assign=True)
    install_streaming_backbone(streamed, SimpleNamespace(mapping=source, get=source.__getitem__),
                               SimpleNamespace(text=c), torch.device('cuda'), torch.bfloat16, 1, context_type, cached_layers)
    original.setup_caches(1, 12)
    streamed.setup_caches(1, 12)
    x = torch.randn(1, 6, 128, device='cuda', dtype=torch.bfloat16)
    for start, end in ((0, 4), (4, 5), (5, 6)):
        positions = torch.arange(start, end, device='cuda')
        expected = original(inputs_embeds=x[:, start:end], input_pos=positions)
        actual = streamed(inputs_embeds=x[:, start:end], input_pos=positions)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert streamed.model.layers[0].input_layernorm.weight.device.type == 'cuda'
    assert all(layer.input_layernorm.weight.device.type == 'meta' for layer in streamed.model.layers[1:])


@pytest.mark.skipif(os.environ.get('DRAMA3_GPU_TESTS') != '1', reason='Requires coordinated exclusive CUDA admission')
def test_blackwell_full_attention_rotary_and_cache_parity():
    upstream = upstream_module('layers')
    c = config().text
    c.head_dim = 256
    torch.manual_seed(93)
    ref = initialize(upstream.FullAttention(c, dtype=torch.bfloat16, device='cuda'))
    native = FullAttention(c, torch.bfloat16, 'cuda', comfy.ops.manual_cast)
    native.load_state_dict(ref.state_dict())
    ref.allocate_cache(1, 16, torch.bfloat16, 'cuda')
    x = torch.randn(1, 7, 16, dtype=torch.bfloat16, device='cuda')
    inv = c.rope_theta ** (-torch.arange(0, c.rotary_dim, 2, device='cuda').float() / c.rotary_dim)
    phase = torch.arange(7, device='cuda')[:, None] * inv
    phase = torch.cat((phase, phase), -1)
    cache = []
    for start, end in ((0, 6), (6, 7)):
        context = upstream.LayerContext(input_pos=torch.arange(start, end, device='cuda'), cos=phase[start:end].cos(),
            sin=phase[start:end].sin(), kv_len=end, continue_state=start > 0, use_cache=True)
        expected = ref(x[:, start:end], context)
        actual = native(x[:, start:end], start, cache, 16)
        torch.testing.assert_close(actual, expected, atol=.005, rtol=.05)
