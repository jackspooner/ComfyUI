import asyncio

import pytest

from comfy.cli_args import args

args.cpu = True

import execution
import nodes
from comfy_api.latest import io
from comfy_extras.nodes_textgen import TextGenerate


class TextInputFixture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[io.Clip.Output(), io.Audio.Output()])

    @classmethod
    def execute(cls):
        raise AssertionError("Validation must not execute source fixtures")


class TextOutputFixture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.String.Input("text")], outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, text):
        raise AssertionError("Validation must not execute output fixtures")


@pytest.fixture
def prompt(monkeypatch):
    for cls in (TextInputFixture, TextOutputFixture, TextGenerate):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    return {
        "source": {"class_type": "TextInputFixture", "inputs": {}},
        "generate": {"class_type": "TextGenerate", "inputs": {
            "clip": ["source", 0], "prompt": "Describe the audio.",
            "max_length": 512, "sampling_mode": "off",
            "use_default_template": False, "auto_resize": False,
        }},
        "sink": {"class_type": "TextOutputFixture", "inputs": {"text": ["generate", 0]}},
    }


@pytest.mark.parametrize("with_audio", [False, True])
@pytest.mark.parametrize("explicit_text", [False, True])
def test_legacy_text_prompt_accepts_omitted_json_controls(prompt, with_audio, explicit_text):
    if with_audio:
        prompt["generate"]["inputs"]["audio"] = ["source", 1]
    if explicit_text:
        prompt["generate"]["inputs"].update(json_schema="", response_format="text")
    valid, _, outputs, errors = asyncio.run(execution.validate_prompt("text-compat", prompt, None))
    assert valid, errors
    assert outputs == ["sink"]
    assert errors == {}


def test_required_prompt_is_still_enforced(prompt):
    del prompt["generate"]["inputs"]["prompt"]
    valid, _, _, errors = asyncio.run(execution.validate_prompt("text-missing", prompt, None))
    assert not valid
    assert any(e["type"] == "required_input_missing" and e["extra_info"]["input_name"] == "prompt"
               for e in errors["generate"]["errors"])


def test_schema_is_an_optional_input():
    inputs = TextGenerate.INPUT_TYPES()
    assert "json_schema" in inputs["optional"]
    assert "json_schema" not in inputs["required"]
