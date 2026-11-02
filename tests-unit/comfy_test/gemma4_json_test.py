"""Constrained Gemma4 protocol and native delegation, without loading weights."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.text_json import PreparedJSON, StructuredOutputError
from comfy.text_encoders.gemma4 import Gemma4UnifiedTokenizer, Gemma4Model
from text_json_native_test import embedded_tokenizer, replay, SCHEMA


@pytest.fixture(scope="module")
def wrapper():
    path = os.environ.get("COMFY_GEMMA_JSON_FIXTURE")
    if not path:
        pytest.skip("Set COMFY_GEMMA_JSON_FIXTURE to a complete local Gemma4 12B package")
    backend = embedded_tokenizer(Path(path))
    return Gemma4UnifiedTokenizer(tokenizer_data={"tokenizer_json": backend.to_str().encode()})


def test_actual_gemma_templates_mask_unicode_and_stops(wrapper):
    p = PreparedJSON(wrapper.structured_output_tokenizer(), json.dumps(SCHEMA), 262144, [1, 50, 106], "gemma4_12b")
    p.check_prompt(wrapper.tokenize_with_weights("Describe the image", skip_template=False, thinking=False))
    p.check_prompt(wrapper.tokenize_chat([{"role": "user", "content": "Describe"}], thinking=False))
    text = '{"text":"é中🙂 and <Subject 1>"}'
    attempt = replay(p, text)
    assert attempt.finish()[0] == text
    assert attempt.finish()[2]["family"] == "gemma4_12b"
    assert attempt.allowed_mask(torch.zeros(1, p.vocab_size))[0, list(p.stops)].all()
    with pytest.raises(StructuredOutputError, match="thinking"):
        PreparedJSON(wrapper.structured_output_tokenizer(), json.dumps(SCHEMA), 262144, [1, 50, 106], "gemma4_12b", True)
    with pytest.raises(StructuredOutputError, match="template"):
        p.check_prompt(wrapper.tokenize_with_weights("raw", skip_template=True))


def test_wrapper_passes_constraint_to_shared_generation():
    captured = {}
    marker = object()
    def generate(*a, **kw):
        captured.update(kw)
        return [4]
    model = SimpleNamespace(execution_device="cpu", transformer=SimpleNamespace(generate=generate))
    with patch("comfy.sd1_clip.SDClipModel.process_tokens", return_value=(torch.zeros(1, 2, 3), None, None, [])):
        assert Gemma4Model.generate(model, [[(2, 1), (3, 1)]], False, 10, 1, 0, 1, 0, 1, 0, mtp=False, constraint=marker) == [4]
    assert captured["constraint"] is marker


def test_other_gemma_variants_remain_unadmitted():
    model = SimpleNamespace(transformer=SimpleNamespace(model=SimpleNamespace(config=object())))
    with pytest.raises(StructuredOutputError, match="complete Gemma4 12B"):
        Gemma4Model.prepare_structured_output(model, None, "{}", False)
