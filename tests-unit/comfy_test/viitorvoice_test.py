"""Small CPU reference models and exact inventories; no full-model inference."""
import importlib
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.audio_encoders.wav2vec2_bert import Wav2Vec2Bert
from comfy.ldm.audio.dualcodec import DualCodec
from comfy.ops import manual_cast
from comfy.viitorvoice.model import ViiTorVoice


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_bidirectional_qwen_components(dtype):
    from transformers import Qwen3Config, Qwen3Model
    torch.manual_seed(7401)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, rms_norm_eps=1e-6,
        rope_theta=1000000., attn_implementation='eager')
    reference = Qwen3Model(config).eval().to(dtype)
    values = dict(llm_config=config.to_dict(), audio_codebook_sizes=[19, 11, 11])
    native = ViiTorVoice(values, dtype, 'cpu', manual_cast).eval()
    native.llm.load_state_dict(reference.state_dict(), strict=True)
    for layer in (native.semantic_embedding, native.acoustic_embedding, native.semantic_head, native.acoustic_head):
        torch.nn.init.normal_(layer.weight, std=0.02)
    ids = torch.randint(0, 10, (2, 3, 13))
    ids[:, :, :5] = torch.randint(0, 64, (2, 1, 5))
    audio_mask = torch.zeros(2, 13, dtype=torch.bool)
    audio_mask[:, 5:] = True
    allowed = torch.ones(2, 1, 13, 13, dtype=torch.bool)
    allowed[1, :, :8, 8:] = False
    allowed[1, :, 8:] = torch.eye(13, dtype=torch.bool)[8:]
    with torch.inference_mode():
        embeds = native.prepare_embed_inputs(ids, audio_mask)
        expected_embeds = reference.embed_tokens(ids[:, 0])
        semantic = native.semantic_embedding(torch.where(audio_mask, ids[:, 0], 0))
        acoustic = sum(native.acoustic_embedding(torch.where(audio_mask, ids[:, book], 0)+11*(book-1)) for book in (1, 2))
        expected_embeds = torch.where(audio_mask[..., None], semantic+acoustic, expected_embeds)
        torch.testing.assert_close(embeds, expected_embeds, rtol=0, atol=0)
        mask = torch.zeros_like(allowed, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)
        expected = reference(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
        semantic, acoustic = native(ids, audio_mask, allowed)
        tolerance = 0.012 if dtype == torch.bfloat16 else 2e-6
        torch.testing.assert_close(semantic, native.semantic_head(expected), rtol=tolerance, atol=tolerance)
        expected_acoustic = native.acoustic_head(expected).reshape(2, 13, 2, 11).permute(0, 2, 1, 3)
        torch.testing.assert_close(acoustic, expected_acoustic, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_w2v_conformer_relative_keys_and_padding(dtype):
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
    torch.manual_seed(7401)
    config = Wav2Vec2BertConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, feature_projection_input_dim=10, conv_depthwise_kernel_size=7,
        position_embeddings_type='relative_key', left_max_position_embeddings=8,
        right_max_position_embeddings=2, hidden_dropout=0, apply_spec_augment=False,
        attn_implementation='eager')
    reference = Wav2Vec2BertModel(config).eval().to(dtype)
    native = Wav2Vec2Bert(config.to_dict(), dtype, 'cpu', manual_cast).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    features = torch.randn(2, 19, 10).to(dtype)
    valid = torch.ones(2, 19, dtype=torch.bool)
    valid[1, 14:] = False
    with torch.inference_mode():
        expected = reference(features, attention_mask=valid, output_hidden_states=True).hidden_states[2]
        actual = native(features, valid, output_layer=2)
    tolerance = 0.04 if dtype == torch.bfloat16 else 3e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def dualcodec_reference():
    root = os.environ.get('DUALCODEC_REFERENCE')
    if not root:
        pytest.skip('Set DUALCODEC_REFERENCE to the extracted pinned 0.4.2 wheel')
    # Only BaseModel/AudioSignal are unrelated container dependencies. Import the
    # original math files untouched, without its downloader or package __init__.
    package = types.ModuleType('dualcodec_fixture')
    package.__path__ = [str(Path(root)/'dualcodec/model_codec')]
    sys.modules[package.__name__] = package
    audio = types.ModuleType('audiotools')
    audio.AudioSignal = object
    ml = types.ModuleType('audiotools.ml')
    ml.BaseModel = torch.nn.Module
    containers = types.ModuleType('easydict')
    containers.EasyDict = dict  # Only returned training metadata; encode/decode do not use it.
    from unittest.mock import patch
    with patch.dict(sys.modules, {'audiotools': audio, 'audiotools.ml': ml, 'easydict': containers}):
        return importlib.import_module(package.__name__+'.dualcodec_model').DualCodec


def test_dualcodec_math_codes_and_odd_stride_decode():
    reference_class = dualcodec_reference()
    torch.manual_seed(7401)
    reference = reference_class(encoder_dim=2, encoder_rates=[2, 3], latent_dim=1024,
        decoder_dim=32, decoder_rates=[3, 2], n_codebooks=2, codebook_size=16,
        semantic_codebook_size=32, convnext_dim=16, convnext_layers=1, is_causal=True,
        sample_rate=24000).eval()
    for layer in reference.modules():
        if hasattr(layer, 'weight_g'):
            torch.nn.utils.remove_weight_norm(layer)
    native = DualCodec(encoder_dim=2, latent_dim=1024, decoder_dim=32, rates=(2, 3),
        books=2, size=16, semantic_size=32, convnext_dim=16, convnext_layers=1,
        dtype=torch.float32, device='cpu', operations=manual_cast).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    audio = torch.randn(1, 1, 48)*0.1
    semantic = torch.randn(1, 1024, 8)
    with torch.inference_mode():
        semantic_codes, acoustic_codes = reference.encode(audio, semantic_repr=semantic)
        expected_codes = torch.cat((semantic_codes, acoustic_codes), 1)
        codes = native.encode(audio, semantic, books=3)
        assert torch.equal(codes, expected_codes)
        expected = reference.decode_from_codes(codes[:, :1], codes[:, 1:])
        actual = native.decode(codes)
        assert actual.shape[-1] == 46  # odd stride has no output_padding
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        assert torch.equal(native.encode(audio, semantic, books=1), semantic_codes)


def test_pinned_full_checkpoint_inventories():
    root = os.environ.get('VIITORVOICE_REFERENCE')
    if not root:
        pytest.skip('Set VIITORVOICE_REFERENCE to the pinned local model bundle')
    from safetensors import safe_open
    root = Path(root)
    llm = json.loads((root/'llm/0p6_emotion/config.json').read_text())
    semantic = json.loads((root/'dualcodec/w2v-bert-2.0/config.json').read_text())
    for path, model in ((root/'llm/0p6_emotion/model.safetensors', ViiTorVoice(llm, torch.float32, 'meta', torch.nn)),
                        (root/'dualcodec/w2v-bert-2.0/model.safetensors', Wav2Vec2Bert(semantic, torch.float32, 'meta', torch.nn))):
        expected = model.state_dict()
        with safe_open(path, framework='pt', device='cpu') as saved:
            assert set(saved.keys()) == expected.keys()
            for name, tensor in expected.items():
                assert saved.get_slice(name).get_shape() == list(tensor.shape), name
    native = DualCodec(dtype=torch.float32, device='meta', operations=torch.nn).state_dict()
    with safe_open(root/'dualcodec/dualcodec_ckpts/dualcodec_25hz_16384_1024.safetensors', framework='pt') as saved:
        inventory = {}
        for name in saved.keys():
            if name.endswith('.weight_g'):
                continue
            inventory[name.removesuffix('_v') if name.endswith('.weight_v') else name] = saved.get_slice(name).get_shape()
        assert native.keys() == inventory.keys()
        assert all(list(value.shape) == inventory[name] for name, value in native.items())


def upstream_generation():
    source = os.environ.get('VIITORVOICE_SOURCE')
    if not source:
        pytest.skip('Set VIITORVOICE_SOURCE to the pinned source archive')
    runtime = types.ModuleType('viitorvoice.llm.runtime')
    runtime.DEFAULT_CHECKPOINT_DIR = Path('.')
    runtime.clear_proxies = lambda: None
    runtime.LLMOnnxConfig = runtime.LLMOnnxStepRunner = object
    model = types.ModuleType('viitorvoice.llm.model')
    model.ViiTorVoiceLLMModel = object
    model.LLMForwardOutput = types.SimpleNamespace
    # The original sampler never constructs an ONNX runner in these tests.
    # Keep its sampler/text functions untouched and avoid importing that backend.
    sys.path.insert(0, source)
    try:
        with patch.dict(sys.modules, {'viitorvoice.llm.runtime': runtime, 'viitorvoice.llm.model': model}):
            return importlib.import_module('viitorvoice.llm.generation')
    finally:
        sys.path.remove(source)


@pytest.mark.parametrize('cfg', [0., 2.])
def test_private_mask_refinement_matches_pinned_reference(cfg):
    from transformers import PreTrainedTokenizerFast
    from comfy.viitorvoice import sampling
    reference = upstream_generation()
    root = Path(os.environ['VIITORVOICE_REFERENCE'])/'llm/0p6_emotion'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root, local_files_only=True)
    values = json.loads((root/'config.json').read_text())
    values['llm_config'].update(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    torch.manual_seed(11)
    native = ViiTorVoice(values, torch.float32, 'cpu', manual_cast).eval()
    for name, parameter in native.named_parameters():
        torch.nn.init.normal_(parameter, mean=1. if 'norm' in name else 0., std=.02)
    class Runner:
        def forward_step(self, input_ids, audio_mask, attention_mask):
            semantic, acoustic = native(input_ids, audio_mask, attention_mask)
            return types.SimpleNamespace(semantic_logits=semantic, acoustic_logits=acoustic)
    fixture = object.__new__(reference.ViiTorVoiceLLMGenerator)
    fixture.device, fixture.config, fixture.tokenizer, fixture.model = torch.device('cpu'), values, tokenizer, Runner()
    fixture.num_audio_codebook, fixture.frame_rate = 12, 25
    fixture.audio_mask_ids, fixture.audio_pause_ids, fixture.audio_separator_ids = (values[key] for key in
        ('audio_mask_ids', 'audio_pause_ids', 'audio_separator_ids'))
    target = torch.randint(0, 100, (12, 9))
    editable = torch.zeros(9, dtype=torch.bool)
    editable[2:6] = True
    target[:, editable] = torch.tensor(fixture.audio_mask_ids)[:, None]
    ref = torch.randint(0, 100, (12, 4))
    expected_inputs = fixture._prepare_edit_inference_inputs('Hello. New words.', target, ref_audio_tokens=ref, lang='en')
    ids, audio = sampling.conditioning(tokenizer, values, 'Hello. New words.', target, ref, 'en')
    assert torch.equal(ids[None], expected_inputs['input_ids'])
    assert torch.equal(audio[None], expected_inputs['audio_mask'])
    settings = reference.LLMGenerationConfig(num_step=4, guidance_scale=cfg, position_temperature=1.)
    torch.manual_seed(7401)
    with torch.inference_mode():
        expected = fixture._generate_edit_iterative_single(full_text='Hello. New words.', target_audio_tokens=target,
            editable_audio_mask=editable, gen_config=settings, ref_audio_tokens=ref, lang='en')
        state = torch.random.get_rng_state().clone()
        actual = sampling.generate(native, tokenizer, values, target, editable, ref, 'Hello. New words.', 'en',
            4, cfg, 7401, {}, lambda *_: None)
        assert torch.equal(state, torch.random.get_rng_state())
        assert torch.equal(actual, expected)
        assert torch.equal(actual[:, ~editable], target[:, ~editable])
        # Cancellation leaves the input draft and global generator untouched;
        # a deliberate retry starts with its original seed and private state.
        snapshot = target.clone()
        with patch('comfy.viitorvoice.sampling.mm.throw_exception_if_processing_interrupted', side_effect=RuntimeError('cancelled')):
            with pytest.raises(RuntimeError, match='cancelled'):
                sampling.generate(native, tokenizer, values, target, editable, ref, 'Hello. New words.', 'en',
                    4, cfg, 7401, {}, lambda *_: None)
        assert torch.equal(snapshot, target)
        assert torch.equal(actual, sampling.generate(native, tokenizer, values, target, editable, ref,
            'Hello. New words.', 'en', 4, cfg, 7401, {}, lambda *_: None))


@pytest.mark.parametrize('edited', ['We see a dog now.', 'We see a small cat now.', 'We see now.'])
@pytest.mark.parametrize('rate', [24000, 44100])
def test_sentence_plans_and_pcm_splicing(edited, rate):
    from comfy.audio_codec import AudioCodes, AudioCodeSegment, DUALCODEC_V1
    from comfy.viitorvoice import runtime
    original = 'We see a cat now.'
    words = [('We',0,2), ('see',3,6), ('a',7,8), ('cat',9,12), ('now.',13,17)]
    aligned = dict(segments=[dict(text=word, start=.3+index*.4, end=.6+index*.4) for index,(word,_,_) in enumerate(words)], warnings=[])
    tokens = torch.randint(0, 1024, (1, 12, 75))
    prepared = runtime.plan(AudioCodes(DUALCODEC_V1,(AudioCodeSegment(tokens,(75,)),)), json.dumps(aligned),
        original, edited, 'en', 'word', 40., 1.5, 1.)
    edit = prepared['plan']
    # The pinned word diff may retain duplicate boundary whitespace on deletion.
    assert ' '.join(edit.span.edited_full_text.split()) == edited
    # Independent native waveform fixture with distinguishable source channels.
    source = dict(waveform=torch.linspace(-.4,.4,3*rate).reshape(1,1,-1).repeat(1,2,1),sample_rate=rate)
    source['waveform'][:,1].neg_()
    keep = torch.ones(edit.new_total_frames, dtype=torch.bool)
    keep[runtime.mask_ranges(edit.editable_audio_mask)[0][0]+1] = False
    generated = dict(prepared, keep=keep, seed=7401)
    decoded = dict(waveform=torch.full((1,1,int(keep.sum())*960-4),.12),sample_rate=24000)
    result, metadata = runtime.splice(source, decoded, generated)
    regions = json.loads(metadata)['edits']
    old_cursor = new_cursor = 0
    for region in regions:
        old_end, new_end = region['source_start_sample'],region['output_start_sample']
        assert torch.equal(source['waveform'][...,old_cursor:old_end], result['waveform'][...,new_cursor:new_end])
        old_cursor, new_cursor = region['source_end_sample'],region['output_end_sample']
    assert torch.equal(source['waveform'][...,old_cursor:], result['waveform'][...,new_cursor:])
    assert result['sample_rate'] == rate and result['waveform'].shape[1] == 2


def test_pinned_preprocessing_and_nonverbal_text():
    reference = upstream_generation()
    from comfy.viitorvoice import text
    for value in ('Hello. New words!', '你好，世界。', '<|nv-Cough|> hi', 'これは日本語です。', '(sighs) Wait; go!'):
        assert text.normalize_nvv_tags(value) == reference.normalize_nvv_tags(value)
        assert text.insert_text_pause_anchors(value) == reference.insert_text_pause_anchors(value)
