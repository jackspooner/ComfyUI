"""Original SoundEffect prompting against its exact pinned processor, no weights."""
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

from comfy.cli_args import args
args.cpu = True
from comfy.text_encoders.moss_speech import MossSpeechTokenizer
from comfy_extras.nodes_moss_speech import MossDiscreteSoundEffect


@pytest.mark.parametrize('seconds', [.5, 2.3, 5.1, 60.])
def test_pinned_prompt_preserves_none_fields_and_discrete_duration(seconds):
    selected = os.environ.get('MOSS_SOUNDEFFECT_DISCRETE_SOURCE')
    if not selected:
        pytest.skip('Requires pinned MOSS_SOUNDEFFECT_DISCRETE_SOURCE, without loading weights')
    source = Path(selected)
    manifest = json.loads((Path(__file__).parents[2] / 'scripts/moss_speech_soundeffect.json').read_text())
    for name in ('configuration_moss_tts.py', 'modeling_moss_tts.py', 'processing_moss_tts.py', 'inference_utils.py'):
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == manifest['files'][name]['sha256']
    package = types.ModuleType('discrete_effect_reference')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    original = importlib.import_module(package.__name__ + '.modeling_moss_tts')
    processor = original.MossTTSDelayProcessor(
        tokenizer=Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True),
        model_config=original.MossTTSDelayConfig.from_pretrained(str(source), local_files_only=True))
    ambient = 'Gentle rain on broad leaves, without speech.'
    message = processor.build_user_message(ambient_sound=ambient, tokens=max(1, int(seconds * 12.5)))
    expected = processor([[message]], mode='generation')['input_ids'][0, :, 0].tolist()
    raw = torch.frombuffer(bytearray((source / 'tokenizer.json').read_bytes()), dtype=torch.uint8)
    # Automatic shape detection shares TTSD's architecture; the explicit effect
    # node must select its own prompt on every call, leaving TTSD unchanged.
    tokenizer = MossSpeechTokenizer(tokenizer_data=dict(tokenizer_json=raw, codebooks=16, dialogue=True))
    captured = {}
    def generate(tokens, **controls):
        captured.update(tokens=tokens, controls=controls)
        return 'codes'
    clip = types.SimpleNamespace(tokenize=tokenizer.tokenize_with_weights, generate_audio_codes=generate)
    MossDiscreteSoundEffect.execute(clip, ambient, seconds, 7401, 256, 1.5, .6, 50, 1.2)
    assert [pair[0] for pair in captured['tokens']['moss_delay'][0]] == expected
    assert captured['controls'] == dict(max_length=256, seed=7401, text_temperature=1.5,
        text_top_p=.6, text_top_k=50, audio_temperature=1.5, audio_top_p=.6,
        audio_top_k=50, audio_repetition_penalty=1.2)
    assert tokenizer.dialogue is True
