import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from transformers import Qwen2TokenizerFast
from transformers.generation.logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.text_encoders.moss_tts_local import MossTTSLocal, sample_local_token
from comfy.text_encoders.moss_speech import MossLocalTokenizer, MossLocalClipModel
from comfy.audio_codec import AudioCodes, AudioCodeSegment, MOSS_V1
from scripts.convert_moss_local import expected_shapes, mapped_name, validate_config
from scripts.convert_moss_audio import inspect_shards, stream_checkpoint


@pytest.fixture(autouse=True)
def deterministic():
    with torch.random.fork_rng(devices=[]), torch.set_grad_enabled(False):
        torch.manual_seed(73)
        yield


@pytest.fixture(scope='module')
def reference():
    if not os.environ.get('MOSS_LOCAL_SOURCE'):
        pytest.skip('Requires pinned MOSS_LOCAL_SOURCE')
    source = Path(os.environ['MOSS_LOCAL_SOURCE'])
    manifest = json.loads((Path(__file__).parents[2] / 'scripts/moss_speech_local.json').read_text())
    for name in ['configuration_moss_tts.py', 'modeling_moss_tts.py', 'processing_moss_tts.py', 'inference_utils.py']:
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == manifest['files'][name]['sha256']
    package = types.ModuleType('moss_local_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + '.modeling_moss_tts'), importlib.import_module(package.__name__ + '.processing_moss_tts'), source


@pytest.mark.parametrize('temperature,top_p,top_k,penalty', [(0., .3, 3, 1.2), (1.7, .7, 5, 1.3), (1., 1., 0, 1.), (.5, 0., 50, 1.5)])
def test_sampling_matches_transformers(temperature, top_p, top_k, penalty):
    logits = torch.randn(3, 17)
    logits[:, 2:5] = 1.2  # tied thresholds must retain all equal scores
    history = torch.tensor([[1, 3, 3], [2, 4, 4], [7, 8, 7]])
    processors = LogitsProcessorList()
    if temperature > 0:
        processors.extend([RepetitionPenaltyLogitsProcessor(penalty), TemperatureLogitsWarper(temperature)])
        if top_k:
            processors.append(TopKLogitsWarper(top_k))
        processors.append(TopPLogitsWarper(top_p))
    scores = processors(history, logits.clone())
    expected = scores.argmax(-1) if temperature == 0 else torch.multinomial(scores.softmax(-1), 1, generator=torch.Generator().manual_seed(9)).squeeze(-1)
    rng = torch.random.get_rng_state().clone()
    actual = sample_local_token(logits.clone(), history, torch.Generator().manual_seed(9), temperature, top_p, top_k, penalty)
    assert torch.equal(actual, expected)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_pinned_inventory(reference):
    _, _, source = reference
    validate_config(json.loads((source / 'config.json').read_text()))
    records = inspect_shards(source, expected_shapes())
    assert len(records) == 556
    assert len({mapped_name(k) for k in records}) == len(records)


def test_mapping_preserves_source_accounting(tmp_path):
    from safetensors.torch import save_file, load_file
    source = tmp_path / 'model.safetensors'
    save_file({'a': torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}, source)
    records = inspect_shards(tmp_path, {'a': [2, 3]})
    output = tmp_path / 'converted.safetensors'
    manifest = stream_checkpoint(records, None, output, {'format': 'pt'}, name_mapping={'a': 'b'})
    assert manifest['tensors']['a']['destination'] == 'b'
    torch.testing.assert_close(load_file(str(output))['b'], load_file(str(source))['a'], rtol=0, atol=0)


@pytest.mark.parametrize('prefix_length', [None, 0, 3, 40])
@pytest.mark.parametrize('reference_length', [None, 0, 5])
def test_prompt_and_continuation_match(reference, prefix_length, reference_length):
    original, processing, source = reference
    config = original.MossTTSDelayConfig.from_pretrained(str(source), local_files_only=True)
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    processor = processing.MossTTSDelayProcessor(tokenizer=tokenizer, model_config=config)
    raw = torch.frombuffer(bytearray((source / 'tokenizer.json').read_bytes()), dtype=torch.uint8)
    native = MossLocalTokenizer(tokenizer_data=dict(tokenizer_json=raw, codebooks=32))
    refs = None if reference_length is None else [torch.arange(reference_length * 32).reshape(reference_length, 32) % 1024]
    controls = dict(text='Hello [softly]!!! 你好。', instruction='Use {text} in {language}', language='en', tokens=83,
                    quality='studio', sound_event='laugh', ambient_sound='quiet')
    messages = [dict(role='user', reference=refs, **controls)]
    prefix = None if prefix_length is None else torch.arange(prefix_length * 32).reshape(prefix_length, 32) % 1024
    if prefix is not None:
        messages.append(dict(role='assistant', audio_codes_list=[prefix]))
    expected = processor([messages], mode='generation' if prefix is None else 'continuation')['input_ids']
    def payload(x):
        return AudioCodes(MOSS_V1, (AudioCodeSegment(x.T[None].clone(), (len(x),)),))
    tokens = native.tokenize_with_weights(**controls, references=None if refs is None else [payload(refs[0])],
                                         continuation=None if prefix is None else payload(prefix))
    stage = MossLocalClipModel.__new__(MossLocalClipModel)
    stage.transformer = types.SimpleNamespace(codebooks=32)
    stage.execution_device = torch.device('cpu')
    actual, mask = stage.prepare_inputs(tokens)
    assert mask.all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tiny_global_cache_and_local_logits(reference):
    original, _, _ = reference
    language = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=128, vocab_size=155648, rope_theta=1000000., rms_norm_eps=1e-6)
    config = original.MossTTSDelayConfig(language_config=language, n_vq=3, additional_mlp_ffn_hidden_size=40,
                                       local_hidden_size=24, local_ffn_hidden_size=48, local_num_layers=2)
    config._attn_implementation = 'sdpa'
    config.language_config._attn_implementation = 'sdpa'
    ref = original.MossTTSDelayModel(config).eval()
    native = MossTTSLocal(dict(language={k: v for k, v in language.items() if k != 'head_dim'}, codebooks=3,
                              local=dict(hidden_size=24, intermediate_size=48, num_hidden_layers=2), bridge_size=40), torch.float32, 'cpu', manual_cast)
    state = {mapped_name(k): v for k, v in ref.state_dict().items()}
    state.pop('unused_global_embedding.weight')
    native.load_state_dict(state, strict=True)
    ids = torch.randint(0, 1025, (2, 5, 4)); ids[..., 0] %= 155648
    torch.testing.assert_close(native.combined_embeddings(ids), ref.model._prepare_multi_modal_inputs(ids, 3), rtol=0, atol=0)
    mask = torch.ones(2, 5, dtype=torch.int64); mask[1, 0] = 0
    cache = native.model.init_kv_cache(2, 8, torch.device('cpu'), torch.float32)
    ref_cache = None
    for start, end in [(0, 3), (3, 4), (4, 5)]:
        actual, _, cache = native.model(None, embeds=native.combined_embeddings(ids[:, start:end]), attention_mask=mask[:, :end], past_key_values=cache)
        expected = ref.model(input_ids=ids[:, start:end], attention_mask=mask[:, :end], past_key_values=ref_cache, use_cache=True, n_vq_for_inference=3)
        ref_cache = expected.past_key_values
        torch.testing.assert_close(actual[mask[:, start:end].bool()], expected.last_hidden_state[mask[:, start:end].bool()], atol=2e-6, rtol=1e-4)
        local = [native.speech_embedding_to_local_mlp(actual[:, -1])]
        local_cache = None
        for q in range(4):
            x = torch.stack(local, 1)
            h = ref.local_transformer(inputs_embeds=x).last_hidden_state[:, -1]
            logits = ref.lm_heads[q](ref.layer_norm_before_lm_heads[q](ref.local_to_speech_embedding_mlps[q](h)))
            full, _ = native.channel_logits(x.clone(), q)
            cached, local_cache = native.channel_logits(x[:, -1:].clone(), q, local_cache)
            torch.testing.assert_close(full, logits, atol=3e-6, rtol=1e-4)
            torch.testing.assert_close(cached, logits, atol=3e-6, rtol=1e-4)
            if q < 3:
                local.append(ref.speech_embedding_to_local_mlp(ref.model.embedding_list[q](ids[:, -1, q])))


@pytest.mark.parametrize('budget', [0, 1, 5])
def test_generation_budget_stopping_and_cancel(budget):
    from comfy.text_encoders.moss_tts_delay import AUDIO_START, AUDIO_END, AUDIO_GEN
    from comfy import model_management
    from unittest.mock import patch
    class Scripted:
        dtype = torch.float32
        def __init__(self):
            self.step = 0
            self.model = Global()
            self.speech_embedding_to_local_mlp = lambda x: x
            self.emb_ext = [lambda x, **kw: x[:, None].float()]
        def combined_embeddings(self, ids):
            return ids[..., :1].float()
        def channel_logits(self, x, q, cache):
            logits = torch.full((2, 155648 if q == 0 else 1025), -100.)
            if q == 0:
                self.step += 1
                logits[0, AUDIO_END] = 100
                logits[1, AUDIO_END if self.step == 3 else AUDIO_GEN] = 100
            else:
                logits[:, q + 10] = 100
            return logits, None
    class Global:
        def init_kv_cache(self, *args):
            return None
        def __call__(self, x, embeds, **kwargs):
            return embeds, None, None
        def embed_tokens(self, x, **kwargs):
            return x[:, None].float()
    ids = torch.full((2, 3, 3), 1024, dtype=torch.int64)
    ids[:, :, 0] = torch.tensor([151644, AUDIO_START, AUDIO_GEN])
    ids[:, -1, 1:] = 7
    out = MossTTSLocal.generate_local(Scripted(), ids, torch.ones(2, 3), budget, 9, text_temperature=0, audio_temperature=0)
    assert [n for n, _ in out] == [1, 1]
    assert out[0][1].shape[0] == 2 + min(budget, 3)
    if budget > 1:
        assert out[0][1][-1, 0] == AUDIO_END
        assert out[0][1][-1, 1:].tolist() == [1024, 1024]
        assert out[1][1][-1, 1:].tolist() == [11, 12]
    with patch.object(model_management, 'throw_exception_if_processing_interrupted', side_effect=model_management.InterruptProcessingException):
        with pytest.raises(model_management.InterruptProcessingException):
            MossTTSLocal.generate_local(Scripted(), ids, torch.ones(2, 3), 1, 9)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_checkpoint_local_decoder_and_projection(reference, dtype):
    from safetensors import safe_open
    from comfy.text_encoders.llama import Qwen3_8BConfig, RMSNorm
    from comfy.text_encoders.moss_tts_local import LocalTransformer, SpeechBridge
    filename = os.environ.get('MOSS_LOCAL_WEIGHTS')
    if not filename:
        pytest.skip('Requires converted MOSS_LOCAL_WEIGHTS for actual local-decoder component comparison')
    original, _, source = reference
    config = original.MossTTSDelayConfig.from_pretrained(str(source), local_files_only=True)
    local_config = config.language_config
    local_config.hidden_size = config.local_hidden_size
    local_config.intermediate_size = config.local_ffn_hidden_size
    local_config.num_hidden_layers = config.local_num_layers
    local_config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        ref = original.MossTTSLocalTransformer(local_config).eval()
        bridge = original.MossTTSMLP(1536, 2048, 2048).eval()
        norm = original.MossTTSRMSNorm(2048).eval()
    native = LocalTransformer(Qwen3_8BConfig(hidden_size=1536, intermediate_size=8960, num_hidden_layers=4,
                                           num_attention_heads=16, num_key_value_heads=8), dtype, 'meta', manual_cast)
    native_bridge = SpeechBridge(1536, 2048, 2048, dtype, 'meta', manual_cast)
    native_norm = RMSNorm(2048, eps=1e-6, dtype=dtype, device='meta')
    with safe_open(filename, framework='pt', device='cpu') as weights:
        for left, right, prefix in [(ref, native, 'local_transformer.'), (bridge, native_bridge, 'local_to_speech_embedding_mlps.1.'),
                                    (norm, native_norm, 'layer_norm_before_lm_heads.1.')]:
            values = {k: weights.get_tensor(prefix + k).to(dtype) for k in left.state_dict()}
            left.load_state_dict(values, assign=True)
            right.load_state_dict(values, assign=True)
        head = weights.get_tensor('lm_heads.1.weight').to(dtype)
    # Independent inputs at the real hidden width, with real checkpoint parameters.
    x = torch.randn(1, 5, 1536, dtype=dtype)
    expected = ref(inputs_embeds=x).last_hidden_state
    actual, _ = native(x.clone())
    cached = []
    cache = None
    for i in range(x.shape[1]):
        out, cache = native(x[:, i:i+1].clone(), cache)
        cached.append(out)
    cached = torch.cat(cached, 1)
    tolerance = dict(atol=3e-5, rtol=3e-4) if dtype == torch.float32 else dict(atol=.04, rtol=.03)
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(cached, expected, **tolerance)
    expected_logits = torch.nn.functional.linear(norm(bridge(expected)), head)
    actual_logits = torch.nn.functional.linear(native_norm(native_bridge(cached)), head)
    print({'dtype': str(dtype), 'hidden_max_error': (cached - expected).abs().max().item(),
           'logit_max_error': (actual_logits - expected_logits).abs().max().item(),
           'top1_agreement': (actual_logits.argmax(-1) == expected_logits.argmax(-1)).float().mean().item()})
    if dtype == torch.float32:
        torch.testing.assert_close(actual_logits, expected_logits, atol=2e-4, rtol=3e-4)
