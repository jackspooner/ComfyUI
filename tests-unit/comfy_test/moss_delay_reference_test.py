import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from safetensors.torch import load_file
from transformers import Qwen2TokenizerFast

from comfy.cli_args import args
args.cpu = True
from comfy.ops import manual_cast
from comfy.text_encoders.moss_tts_delay import DelayPattern, MossTTSDelay, sample_delay_token
from comfy.text_encoders.moss_speech import MossSpeechTokenizer


@pytest.fixture(scope='module', params=['voicegenerator', 'delay', 'acting', 'ttsd'])
def reference(request):
    variant = request.param
    variable = f'MOSS_{variant.upper()}_SOURCE'
    source = os.environ.get(variable)
    if not source:
        pytest.skip(f'Requires pinned {variable} source and isolated Transformers 5 overlay')
    source = Path(source)
    manifest = json.loads((Path(__file__).parents[2] / f'scripts/moss_speech_{variant}.json').read_text())
    for name in ('configuration_moss_tts.py', 'modeling_moss_tts.py', 'processing_moss_tts.py', 'inference_utils.py'):
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == manifest['files'][name]['sha256']
    package = types.ModuleType(f'moss_{variant}_generation_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    return (importlib.import_module(package.__name__ + '.modeling_moss_tts'),
            importlib.import_module(package.__name__ + '.inference_utils'))


@pytest.fixture(autouse=True)
def deterministic():
    with torch.random.fork_rng(devices=[]), torch.set_grad_enabled(False):
        torch.manual_seed(17)
        yield


@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('top_k,top_p', [(0, 1.), (3, .6), (999, .05)])
def test_sampling_matches_flattened_reference(reference, sample, top_k, top_p):
    _, original = reference
    logits = torch.randn(6, 11)
    history = torch.randint(0, 11, (2, 4, 3))
    torch.manual_seed(93)
    expected = original.sample_token(logits.clone(), prev_tokens=history, repetition_penalty=1.3,
                                     top_k=top_k, top_p=top_p, do_sample=sample)
    before = torch.random.get_rng_state().clone()
    actual = sample_delay_token(logits.clone(), torch.Generator().manual_seed(93), sample, top_k, top_p, history, 1.3)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), before)


class ScriptedLogits:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.dtype = torch.float32
        self.model = types.SimpleNamespace(init_kv_cache=lambda *args: None)

    def __call__(self, input_ids, **kwargs):
        text = torch.full((input_ids.shape[0], 1, self.config.language_config.vocab_size), -100.)
        desired = [151652, 151656, 151656, 151662, 151645]
        text[..., desired[min(self.step, 4)]] = 100
        text[..., 151645] = 200
        audio = []
        for head in range(self.config.n_vq):
            logits = torch.full((input_ids.shape[0], 1, 1025), -100.)
            logits[..., 10 * self.step + head] = 100
            audio.append(logits)
        self.step += 1
        return types.SimpleNamespace(logits=[text, *audio], past_key_values=None)

    def next_logits(self, input_ids, attention_mask, cache):
        return [x[:, -1] for x in self(input_ids).logits], None


@pytest.mark.parametrize('budget', [0, 2, 4, 12])
@pytest.mark.parametrize('continuation', [False, True])
def test_delay_warmup_flush_stop_and_budget(reference, budget, continuation):
    original, _ = reference
    config = original.MossTTSDelayConfig(n_vq=4, language_config={'vocab_size': 155648})
    prompt = torch.full((1, 4 if continuation else 3, 5), 1024, dtype=torch.int64)
    prompt[0, :3, 0] = torch.tensor([151644, 77091, 198])
    if continuation:
        prompt[0, -1, 0] = 151652
    mask = torch.ones(prompt.shape[:2], dtype=torch.int64)
    expected = original.MossTTSDelayModel.generate(ScriptedLogits(config), input_ids=prompt,
        attention_mask=mask, max_new_tokens=budget, text_temperature=0, audio_temperature=0)
    actual = MossTTSDelay.generate_delay(ScriptedLogits(config), prompt, mask, budget, 9,
        text_temperature=0, audio_temperature=0)
    assert actual[0][0] == expected[0][0].item()
    torch.testing.assert_close(actual[0][1], expected[0][1], rtol=0, atol=0)


def test_native_embeddings_and_cached_logits(reference):
    original, _ = reference
    language = dict(vocab_size=155648, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=128,
                    rms_norm_eps=1e-6, rope_theta=1000000., max_position_embeddings=40960)
    config = original.MossTTSDelayConfig(n_vq=4, language_config=language)
    config.language_config._attn_implementation = 'eager'
    ref = original.MossTTSDelayModel(config).eval()
    language.pop('head_dim')
    native = MossTTSDelay({'language': language, 'codebooks': 4}, torch.float32, 'cpu', manual_cast)
    native.load_state_dict({key.replace('language_model.', 'model.'): value for key, value in ref.state_dict().items()}, strict=True)
    inputs = torch.randint(0, 1025, (2, 5, 5))
    inputs[..., 0] = torch.randint(0, 155648, (2, 5))
    inputs[:, :2, 1:] = 1024
    torch.testing.assert_close(native.combined_embeddings(inputs), ref.get_input_embeddings(inputs), rtol=0, atol=0)
    mask = torch.ones(2, 5, dtype=torch.int64)
    mask[1, 0] = 0
    cache = native.model.init_kv_cache(2, 8, torch.device('cpu'), torch.float32)
    ref_cache = None
    for start, end in [(0, 3), (3, 4), (4, 5)]:
        actual, cache = native.next_logits(inputs[:, start:end], mask[:, :end], cache)
        expected = ref(input_ids=inputs[:, start:end], attention_mask=mask[:, :end], past_key_values=ref_cache, use_cache=True)
        ref_cache = expected.past_key_values
        for head, (a, b) in enumerate(zip(actual, expected.logits)):
            # Native raw projections leave the protocol pad mask to DelayPattern.
            width = -1 if head else a.shape[-1]
            torch.testing.assert_close(a[:, :width], b[:, -1, :width], rtol=1e-4, atol=2e-6)
    assert not any('cache' in name for name in vars(native))


def test_batch_items_stop_independently(reference):
    original, _ = reference
    config = original.MossTTSDelayConfig(n_vq=4, language_config={'vocab_size': 155648})
    class BatchedLogits(ScriptedLogits):
        def __call__(self, input_ids, **kwargs):
            output = super().__call__(input_ids, **kwargs)
            if self.step <= 6:
                output.logits[0][1].fill_(-100)
                token = 151652 if self.step == 1 else 151656 if self.step < 6 else 151662
                output.logits[0][1, :, token] = 300
            return output
    prompt = torch.full((2, 3, 5), 1024, dtype=torch.int64)
    prompt[:, :, 0] = torch.tensor([151644, 77091, 198])
    mask = torch.ones(2, 3, dtype=torch.int64)
    expected = original.MossTTSDelayModel.generate(BatchedLogits(config), input_ids=prompt,
        attention_mask=mask, max_new_tokens=16, text_temperature=0, audio_temperature=0)
    actual = MossTTSDelay.generate_delay(BatchedLogits(config), prompt, mask, 16, 9,
        text_temperature=0, audio_temperature=0)
    for (length, ids), (expected_length, expected_ids) in zip(actual, expected):
        assert length == expected_length.item()
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    assert torch.all(actual[0][1][-2:, 0] == 151643)


@pytest.mark.parametrize('text,instruction,language', [
    ('Hello world.', 'A warm voice.', 'en'),
    (' Hello [softly]!!! ', ' [warm] a {bright} voice!!! ', 'en'),
    ('你好，世界！', '温柔、自然的声音。', 'zh'),
    ('', '', None),
    ('hello', 'say {text} in {language}', 'en'),
])
def test_tokenization_matches_worker_dictionary_path(reference, text, instruction, language):
    original, _ = reference
    source = Path(original.__file__).parent
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    config = original.MossTTSDelayConfig.from_pretrained(str(source), local_files_only=True)
    options = {'normalize_inputs': True} if 'voicegenerator' in original.__name__ else {}
    processor = original.MossTTSDelayProcessor(tokenizer=tokenizer, model_config=config, **options)
    message = processor.build_user_message(text=text, instruction=instruction, language=language)
    expected = processor([[message]], mode='generation')['input_ids'][0, :, 0]
    raw = torch.frombuffer(bytearray((source / 'tokenizer.json').read_bytes()), dtype=torch.uint8)
    native = MossSpeechTokenizer(tokenizer_data={'tokenizer_json': raw, 'dialogue': config.n_vq == 16 and config.language_config.hidden_size == 4096})
    tokens = native.tokenize_with_weights(text, instruction=instruction, language=language,
                                         normalize_text='acting' in original.__name__)['moss_delay'][0]
    assert [pair[0] for pair in tokens] == expected.tolist()


def test_full_checkpoint_cached_logits(reference):
    original, _ = reference
    if 'voicegenerator' not in original.__name__:
        pytest.skip('Full checkpoint logit fixture is specific to VoiceGenerator')
    checkpoint = os.environ.get('MOSS_VOICEGENERATOR_WEIGHTS')
    if not checkpoint:
        pytest.skip('Requires the full original VoiceGenerator checkpoint and about 10 GB RAM')
    source = Path(os.environ['MOSS_VOICEGENERATOR_SOURCE'])
    manifest = json.loads((Path(__file__).parents[2] / 'scripts/moss_speech_voicegenerator.json').read_text())
    with open(checkpoint, 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['files']['model.safetensors']['sha256']
    config = original.MossTTSDelayConfig.from_pretrained(str(source), local_files_only=True)
    config.language_config._attn_implementation = 'eager'
    with torch.device('meta'):
        ref = original.MossTTSDelayModel(config).eval()
        native = MossTTSDelay({}, torch.float32, 'meta', manual_cast)
    ref.language_model.rotary_emb = type(ref.language_model.rotary_emb)(config.language_config, device='cpu')
    weights = {key: value.float() for key, value in load_file(checkpoint).items()}
    ref.load_state_dict(weights, strict=True, assign=True)
    native.load_state_dict({key.replace('language_model.', 'model.'): value for key, value in weights.items()}, strict=True, assign=True)
    raw = torch.frombuffer(bytearray((source / 'tokenizer.json').read_bytes()), dtype=torch.uint8)
    tokenizer = MossSpeechTokenizer(tokenizer_data={'tokenizer_json': raw})
    text = [pair[0] for pair in tokenizer.tokenize_with_weights('Hello world.', instruction='A warm voice.', language='en')['moss_delay'][0]]
    prompt = torch.full((1, len(text), 17), 1024, dtype=torch.int64)
    prompt[0, :, 0] = torch.tensor(text)
    mask = torch.ones(prompt.shape[:2], dtype=torch.int64)
    cache = native.model.init_kv_cache(1, len(text) + 3, torch.device('cpu'), torch.float32)
    ref_cache = None
    for step in range(4):
        actual, cache = native.next_logits(prompt, mask, cache)
        expected = ref(input_ids=prompt, attention_mask=mask, past_key_values=ref_cache, use_cache=True)
        ref_cache = expected.past_key_values
        errors = []
        for head, (a, b) in enumerate(zip(actual, expected.logits)):
            width = -1 if head else a.shape[-1]
            b = b[:, -1, :width]
            a = a[:, :width]
            errors.append((a - b).abs().max().item())
            torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-4)
            assert torch.equal(a.argmax(-1), b.argmax(-1))
        print(f'full checkpoint step={step}, maximum logit error={max(errors):.8f}, all 17 argmax values match')
        prompt = torch.stack([logits[:, -1].argmax(-1) for logits in expected.logits], dim=-1)[:, None]
        mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype)), dim=1)
