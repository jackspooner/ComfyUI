import math

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.text_generation import ContextOverflowError
from comfy.text_encoders.gemma4 import prepare_chat_image
from comfy_extras.text_image_budget import plan_image_count
from comfy_extras.nodes_textgen import TextImageBudget


class PlanningClip:
    def __init__(self, capacity):
        self.capacity = capacity
        self.calls = []

    def tokenize_chat(self, messages, thinking):
        assert thinking is False
        images = [part for part in messages[-1]["content"] if part["type"] == "image"]
        assert all(part["data"].device.type == "meta" for part in images)
        return len(images)

    def prepare_generation_tokens(self, tokens, ceiling, output, resize, **kwargs):
        self.calls.append((tokens, ceiling, output, resize, kwargs))
        if tokens * 100 + output > self.capacity:
            raise ContextOverflowError("fixture memory bound")
        return None, {"input_tokens": tokens * 100, "output_allowance": output}


def test_count_reserves_output_and_transport_without_generating_or_pixels():
    clip = PlanningClip(1500)
    count, info = plan_image_count(clip, "describe", "system", 944, 528, 128, 256, reserve_tokens=344)
    assert count == 9
    assert info["context"] == {"input_tokens": 900, "output_allowance": 600}
    assert info["count_limit_reached"] is False
    assert clip.calls[-1][0] == 9
    assert all(call[4] == {"mtp": False, "min_short_edge": 528} for call in clip.calls)
    assert len(clip.calls) <= 10


def test_explicit_count_ceiling_is_reported():
    count, info = plan_image_count(PlanningClip(100000), "", "", 512, 512, 4, 100)
    assert count == 4 and info["count_limit_reached"] is True


def test_no_image_fit_is_an_actionable_failure():
    with pytest.raises(ContextOverflowError, match="No image fits"):
        plan_image_count(PlanningClip(99), "", "", 512, 512, 10, 100, reserve_tokens=0)


def test_non_capacity_errors_are_not_silently_treated_as_a_smaller_budget():
    class BrokenClip(PlanningClip):
        def tokenize_chat(self, *args, **kwargs):
            raise RuntimeError("tokenizer unavailable")
    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        plan_image_count(BrokenClip(10000), "", "", 512, 512, 10, 100)


@pytest.mark.parametrize("key,value", [("width", 0), ("height", True), ("maximum_images", 513), ("max_length", 0), ("reserve_tokens", -1)])
def test_invalid_inputs_fail_before_model_use(key, value):
    inputs = dict(width=512, height=512, maximum_images=4, max_length=100)
    inputs[key] = value
    with pytest.raises(ValueError, match=key):
        plan_image_count(None, "", "", **inputs)


def test_gemma_image_preparation_accepts_pixel_free_shape_probes():
    image = torch.empty((1, 528, 944, 3), device="meta")
    embed = prepare_chat_image(image)
    assert embed["data"].device.type == "meta"
    assert embed["source"] is image


def test_budget_node_is_not_cached_and_adds_schema_to_the_tokenized_prompt():
    class InspectClip(PlanningClip):
        def tokenize_chat(self, messages, thinking):
            assert "JSON Schema" in messages[0]["content"]
            return super().tokenize_chat(messages, thinking)
    assert math.isnan(TextImageBudget.fingerprint_inputs())
    output = TextImageBudget.execute(InspectClip(2000), "describe", "system", 512, 512, 3, 100,
                                     json_schema='{"type":"object"}')
    assert output.result[0] == 3
