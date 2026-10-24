import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, Qwen2Tokenizer

from comfy.cli_args import args
args.cpu = True
from comfy.text_json import PreparedJSON, StructuredOutputError, admit_schema, strict_json
from comfy.text_encoders.llama import BaseGenerate
from comfy_extras.nodes_textgen import TextGenerate

pytest.importorskip("llguidance", reason="Optional requirements-text-json.txt is not installed")

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"type": "object", "properties": {"text": {"type": "string", "minLength": 1}}, "required": ["text"], "additionalProperties": False}


@pytest.fixture(scope="module")
def tokenizer():
    return Qwen2Tokenizer.from_pretrained(str(ROOT.parent / "comfy/text_encoders/qwen25_tokenizer"), local_files_only=True)


@pytest.fixture(scope="module")
def prepared(tokenizer):
    pytest.importorskip("llguidance")
    return PreparedJSON(tokenizer, json.dumps(SCHEMA), 151936, [151643, 151645], "moss_audio", True)


def ordinary_ids(prepared, text):
    # Added control tokens are encoded specially even inside JSON by HF. The
    # grammar can instead select ordinary punctuation/word tokens for literals.
    result = []
    for token in prepared.tokenizer.tokenize_str(text):
        if token in prepared.special_ids:
            for char in prepared.tokenizer.decode_bytes([token]).decode():
                result.extend(prepared.tokenizer.tokenize_str(char))
        else:
            result.append(token)
    return result


def replay(prepared, text, thinking=False):
    attempt = prepared.new_attempt()
    ids = ordinary_ids(prepared, text)
    if thinking:
        ids = [prepared.open_id, *ordinary_ids(prepared, "reason é"), prepared.close_id, *ordinary_ids(prepared, "\n\n"), *ids]
    logits = torch.zeros(1, prepared.vocab_size)
    for token in ids:
        assert attempt.allowed_mask(logits)[0, token], (token, attempt.phase)
        attempt.commit(token)
    return attempt


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e9999", '{}{}', '{"a":1,"a":2}', '"\\ud800"', '{} trailing', '{'])
def test_strict_parse_rejects(value):
    with pytest.raises(StructuredOutputError):
        strict_json(value)


@pytest.mark.parametrize("value", ['"\\ud83d\\ude42"', '"�"', '"é"', '{"quote":"\\\"","slash":"\\/"}'])
def test_strict_unicode(value):
    assert strict_json(value) is not None


@pytest.mark.parametrize("thinking", [False, True])
def test_literal_markers_and_unicode_preserved(prepared, thinking):
    text = json.dumps({"text": "中🙂 é � </think> <|channel>thought\nfoo<channel|>"}, ensure_ascii=False, separators=(",", ":"))
    attempt = replay(prepared, text, thinking)
    answer, reason, evidence = attempt.finish()
    assert answer == prepared.backend.decode(ordinary_ids(prepared, text), skip_special_tokens=False)
    assert "</think> <|channel>thought" in answer
    assert reason == ("reason é" if thinking else "")
    assert evidence["finish_reason"] == "grammar_complete"
    assert evidence["validated"] is True


def test_fresh_matcher_and_budget(prepared):
    first = prepared.new_attempt()
    first.commit(prepared.open_id)
    with pytest.raises(StructuredOutputError, match="reasoning_budget_exhausted"):
        first.finish()
    second = replay(prepared, '{"text":"ok"}')
    assert first.phase == "reasoning"
    assert second.finish()[0] == '{"text":"ok"}'


@pytest.mark.parametrize("text", ['{}', '{"text":"ok","text":"duplicate"}', '{"text":""}'])
def test_required_duplicate_and_minlength_excluded(prepared, text):
    attempt = prepared.new_attempt()
    for token in ordinary_ids(prepared, text):
        if not attempt.allowed_mask(torch.zeros(1, prepared.vocab_size))[0, token]:
            return
        attempt.commit(token)
    pytest.fail("Invalid document was not excluded")


def test_independent_final_validator_is_not_matcher_state(prepared):
    attempt = replay(prepared, '{"text":"ok"}')
    attempt.answer_ids = ordinary_ids(prepared, '{"text":3}')
    with pytest.raises(StructuredOutputError, match="independent JSON Schema"):
        attempt.finish()


def test_schema_resource_limits():
    with pytest.raises(StructuredOutputError, match="128 KiB"):
        admit_schema(" " * 131073)
    nested = {"type": "string"}
    for _ in range(34):
        nested = {"type": "array", "items": nested}
    with pytest.raises(StructuredOutputError, match="maximum depth"):
        admit_schema(json.dumps({**SCHEMA, "properties": {"text": nested}}))


def test_padding_and_all_stop_ids(prepared):
    attempt = prepared.new_attempt()
    mask = attempt.allowed_mask(torch.zeros(1, prepared.vocab_size))
    assert not mask[0, list(prepared.invalid_ids)].any()
    assert not mask[0, list(prepared.stops)].any()
    complete = replay(prepared, '{"text":"ok"}')
    mask = complete.allowed_mask(torch.zeros(1, prepared.vocab_size))
    assert mask[0, list(prepared.stops)].all()
    for token in prepared.stops:
        eos = replay(prepared, '{"text":"ok"}')
        assert eos.commit(token)
        assert eos.finish()[2]["finish_reason"] == "eos"


@pytest.mark.parametrize("top_k", [0, 1, 4, 10])
@pytest.mark.parametrize("min_p,top_p", [(0, 1), (1, 0), (0.9, 0.2)])
def test_hard_mask_after_filters(top_k, min_p, top_p):
    logits = torch.tensor([[100., 2., -3., 99., 98.]])
    allowed = torch.tensor([[False, True, True, False, False]])
    probs, indices = BaseGenerate().processed_probs(logits, 0.7, top_k, top_p, min_p, 1., [], allowed_mask=allowed)
    permitted = allowed if indices is None else allowed.gather(1, indices)
    assert (probs[~permitted] == 0).all()
    assert torch.isfinite(probs).all() and probs.sum() == 1


@pytest.mark.parametrize("do_sample,temp", [(False, 1), (True, 0), (True, 1)])
def test_greedy_and_sampled_exclusions(do_sample, temp):
    if do_sample and temp != 0:
        with pytest.raises(ValueError, match="structured_output.logits"):
            BaseGenerate().sample_token(torch.tensor([[100., 1., 99.]]), temp, 3, 0, 1, 0., [1], torch.Generator().manual_seed(1), do_sample=do_sample, allowed_mask=torch.tensor([[False, True, False]]))
        return
    result = BaseGenerate().sample_token(torch.tensor([[100., 1., 99.]]), temp, 3, 0, 1, 0., [1], torch.Generator().manual_seed(1), do_sample=do_sample, allowed_mask=torch.tensor([[False, True, False]]))
    # Greedy ignores repetition penalty as before. Sampled zero penalty is
    # explicitly invalid rather than silently producing an unconstrained token.
    assert result.item() == 1


@pytest.mark.parametrize("logits,mask", [([1., 2.], [False, False]), ([float('nan'), 1.], [True, False]), ([float('inf'), 1.], [True, False])])
def test_invalid_logits_fail(logits, mask):
    with pytest.raises(ValueError, match="structured_output.logits"):
        BaseGenerate().sample_token(torch.tensor([logits]), 1, 0, 1, 0, 1, [], None, False, allowed_mask=torch.tensor([mask]))


@pytest.mark.parametrize("keyword", ["$ref", "pattern", "format", "oneOf", "x-guidance", "maxLength"])
def test_unknown_keywords_rejected(keyword):
    with pytest.raises(Exception):
        admit_schema(json.dumps({**SCHEMA, keyword: "unsupported"}))


def example(schema):
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][-1]
    kind = schema.get("type")
    if isinstance(kind, list):
        return None
    if kind == "object":
        return {key: example(value) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        return [example(schema["items"])]
    return {"string": "中🙂 sample", "number": max(1.25, schema.get("minimum", 0)), "integer": max(1, schema.get("minimum", 0)), "boolean": False, "null": None}[kind]


@pytest.mark.parametrize("path", sorted((ROOT / "fixtures/text-json").glob("*.json")))
def test_exact_workflow_schemas(tokenizer, path):
    prepared = PreparedJSON(tokenizer, path.read_text(), 151936, [151643, 151645], "qwen3vl")
    assert prepared.evidence["schema_sha256"] == path.stem
    text = json.dumps(example(prepared.schema), ensure_ascii=False, separators=(",", ":"))
    assert replay(prepared, text).finish()[0] == text
    broken = prepared.new_attempt()
    for token in ordinary_ids(prepared, '{"unexpected":true}'):
        if not broken.allowed_mask(torch.zeros(1, prepared.vocab_size))[0, token]:
            break
        broken.commit(token)
    else:
        pytest.fail("Extra key was not excluded")


@pytest.mark.parametrize("schema,text", [
    ({"type":"number","minimum":2}, '1'),
    ({"type":"integer"}, '1.5'),
    ({"type":"string","minLength":2}, '"x"'),
    ({"enum":["yes","no"]}, '"other"'),
    ({"const":True}, 'false'),
])
def test_invalid_values_excluded(tokenizer, schema, text):
    spec = {"type":"object","properties":{"value":schema},"required":["value"],"additionalProperties":False}
    p = PreparedJSON(tokenizer,json.dumps(spec),151936,[151643,151645],"qwen3vl")
    a=p.new_attempt()
    for token in ordinary_ids(p, '{"value":'+text+'}'):
        if not a.allowed_mask(torch.zeros(1,p.vocab_size))[0,token]:
            return
        a.commit(token)
    pytest.fail("Invalid value was not excluded")


def embedded_tokenizer(path):
    # Read the header and tokenizer tensor only, never model tensors.
    with path.open("rb") as source:
        size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(size))
        start, end = header["tokenizer_json"]["data_offsets"]
        source.seek(8 + size + start)
        return Tokenizer.from_str(source.read(end - start).decode())


@pytest.mark.parametrize("filename", ["moss_audio_8b_thinking_bf16.safetensors", "moss_audio_8b_instruct_bf16.safetensors", "gemma4_12b_bf16.safetensors"])
def test_actual_embedded_tokenizer_bytes(filename):
    root = os.environ.get("COMFY_JSON_MODEL_FIXTURES")
    if not root:
        pytest.skip("Set COMFY_JSON_MODEL_FIXTURES for header/tokenizer-only acceptance")
    import llguidance
    backend = embedded_tokenizer(Path(root) / filename)
    stops = [1, 106] if filename.startswith("gemma") else [151643,151645]
    ll = llguidance.LLTokenizer(backend.to_str(), eos_token=stops)
    for text in ['{"text":"é中🙂 � é"}', '{"literal":"</think> <|channel>final\\n"}', ' space  newline\n']:
        ids = backend.encode(text, add_special_tokens=False).ids
        assert ll.decode_bytes(ids).decode("utf-8", errors="strict") == backend.decode(ids, skip_special_tokens=False)
    if filename.startswith("moss"):
        from comfy.text_encoders.moss_audio import MossAudioTokenizer
        p = PreparedJSON(PreTrainedTokenizerFast(tokenizer_object=backend), json.dumps(SCHEMA), 151936, stops, "moss_audio", True)
        assert replay(p, '{"text":"answer </think>"}', True).finish()[0] == '{"text":"answer </think>"}'
        wrapper = MossAudioTokenizer(tokenizer_data={"tokenizer_json": backend.to_str().encode()})
        for thinking in (False, True):
            p = PreparedJSON(wrapper.structured_output_tokenizer(), json.dumps(SCHEMA), 151936, stops, "moss_audio", thinking)
            p.check_prompt(wrapper.tokenize_with_weights("Hello", thinking=thinking))
            p.check_prompt(wrapper.tokenize_chat([{"role": "user", "content": "Hello"}], thinking=thinking))


def test_qwen_actual_template_and_unsupported_thinking(tokenizer):
    from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer
    wrapper = Qwen3VLTokenizer()
    p = PreparedJSON(tokenizer, json.dumps(SCHEMA), 151936, [151643, 151645], "qwen3vl")
    p.check_prompt(wrapper.tokenize_with_weights("Hello"))
    p.check_prompt(wrapper.tokenize_chat([{"role": "user", "content": "Hello"}]))
    with pytest.raises(StructuredOutputError, match="thinking"):
        PreparedJSON(tokenizer, json.dumps(SCHEMA), 151936, [151643, 151645], "qwen3vl", True)


def test_shared_loop_masks_commits_and_completes(prepared):
    ids = ordinary_ids(prepared, '{"text":"ok"}')
    class Generator(BaseGenerate):
        def __init__(self):
            self.step = 0
            self.model = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=100, stop_tokens=list(prepared.stops)),
                graph_dynamic_vbar_blocks=False,
                embed_tokens=lambda ids: torch.zeros(1, 1, 2),
                forward=lambda *a, **kw: (kw["embeds"], None, kw["past_key_values"]),
            )

        def init_kv_cache(self, *args):
            return []

        def logits(self, x):
            logits = torch.full((1, 1, prepared.vocab_size), -100.)
            logits[0, 0, ids[self.step]] = 10.
            logits[0, 0, max(prepared.invalid_ids)] = 100.
            self.step += 1
            return logits

    attempt = prepared.new_attempt()
    result = Generator().generate(torch.zeros(1, 1, 2), do_sample=False, max_length=len(ids), constraint=attempt)
    assert result == ids
    assert attempt.finish()[0] == '{"text":"ok"}'


def test_package_guard_tracks_shards_and_rejects_mtp():
    from comfy.sd import CLIP
    clip = CLIP(no_init=True)
    clip.generation_missing_keys = None
    clip.generation_load_scope = None
    clip.patcher = SimpleNamespace(is_dynamic=lambda: False)
    clip.cond_stage_model = SimpleNamespace(load_state_dict=lambda sd, **kw: ([key for key in ("head", "body") if key not in sd], []))
    clip.load_sd({"head": 1}, full_model=True)
    assert clip.generation_missing_keys == {"body"}
    with pytest.raises(StructuredOutputError, match="capability"):
        clip.prepare_structured_output("{}", mtp=False)
    clip.load_sd({"body": 1}, full_model=True)
    assert clip.generation_missing_keys == set()
    with pytest.raises(StructuredOutputError, match="mtp"):
        clip.prepare_structured_output("{}", mtp=True)


def test_conditioning_package_rejected_but_tied_head_admitted(tokenizer):
    from comfy.sd1_clip import SDClipModel
    config = SimpleNamespace(final_norm=False, lm_head=False, vocab_size=151936, stop_tokens=[151643, 151645])
    model = SimpleNamespace(structured_output_family="qwen3vl", transformer=SimpleNamespace(model=SimpleNamespace(config=config, norm=None)))
    with pytest.raises(StructuredOutputError, match="capability"):
        SDClipModel.prepare_structured_output(model, tokenizer, json.dumps(SCHEMA), False)
    config.final_norm = True
    model.transformer.model.norm = object()
    assert SDClipModel.prepare_structured_output(model, tokenizer, json.dumps(SCHEMA), False).vocab_size == 151936


def test_node_contract_and_early_rejection():
    assert len(TextGenerate.define_schema().outputs) == 5
    clip = SimpleNamespace(prepare_structured_output=lambda *a, **k: (_ for _ in ()).throw(ValueError("preparation reached")))
    with pytest.raises(ValueError, match="preparation reached"):
        TextGenerate.execute(clip, "hi", 50, {}, response_format="json_schema", mtp="off", json_schema=json.dumps(SCHEMA))
    with pytest.raises(ValueError, match="format"):
        TextGenerate.execute(clip, "hi", 50, {}, response_format="json_object")


def test_node_success_bypasses_heuristic_and_commits_after_validation(prepared):
    text = '{"text":"literal </think>"}'
    def generate(tokens, constraint, **kwargs):
        for token in ordinary_ids(prepared, text):
            constraint.commit(token)
        return constraint.raw_ids
    tokens = {"moss_audio": [[(t,1.) for t in prepared.prompt_suffix]]}
    clip = SimpleNamespace(prepare_structured_output=lambda *a, **k: prepared, tokenize=lambda *a, **k: tokens, generate=generate)
    out = TextGenerate.execute(clip,"hi",100,{},response_format="json_schema",mtp="off",thinking=True,auto_resize=False,json_schema=json.dumps(SCHEMA))
    assert out.result[0] == text
    assert out.result[1] == ""
    assert json.loads(out.result[3])["structured_output"]["enforcement"] == "native_schema"
    assert json.loads(out.result[2])["messages"][-1]["content"] == text


def test_oom_retry_uses_fresh_constraint(prepared, monkeypatch):
    attempts = []
    tokens = {"moss_audio": [[(t, 1.) for t in prepared.prompt_suffix]]}
    class Clip:
        def prepare_structured_output(self, *args, **kwargs):
            return prepared

        def tokenize_video(self, *args, **kwargs):
            return tokens, {"sampled_frames": 1}

        def prepare_generation_tokens(self, value, budget, output, auto, memory_budget=None, **kwargs):
            return tokens, {"output_allowance": output, "memory_budget_bytes": 1000,
                            "estimated_working_bytes": 1000, "image_sizes": [[512 if memory_budget is None else 256]],
                            "excluded_message_ranges": []}

        def generate(self, value, constraint, **kwargs):
            attempts.append(constraint)
            if len(attempts) == 1:
                constraint.commit(prepared.open_id)
                raise torch.cuda.OutOfMemoryError("fixture")
            assert not constraint.raw_ids and constraint.phase == "start"
            for token in ordinary_ids(prepared, '{"text":"ok"}'):
                constraint.commit(token)
            return constraint.raw_ids

    monkeypatch.setattr("comfy.model_management.soft_empty_cache", lambda: None)
    result = TextGenerate.execute(Clip(), "describe", 100, {}, video=torch.zeros(1, 1, 1, 3),
                                  response_format="json_schema", thinking=True, mtp="off", json_schema=json.dumps(SCHEMA)).result
    assert len(attempts) == 2 and attempts[0] is not attempts[1]
    assert result[0] == '{"text":"ok"}' and result[2] == ""
    assert json.loads(result[3])["oom_retries"] == 1


@pytest.mark.parametrize("interrupt", [False, True])
def test_failure_never_commits_history(prepared, monkeypatch, interrupt):
    tokens = {"moss_audio": [[(t, 1.) for t in prepared.prompt_suffix]]}
    def generate(value, constraint, **kwargs):
        constraint.commit(prepared.open_id)
        if interrupt:
            raise InterruptedError("cancelled")
        return constraint.raw_ids
    def forbidden(*args, **kwargs):
        pytest.fail("Failed generation attempted history publication")
    monkeypatch.setattr("comfy_extras.nodes_textgen.publish_images", forbidden)
    monkeypatch.setattr("comfy_extras.nodes_textgen.write_history", forbidden)
    clip = SimpleNamespace(prepare_structured_output=lambda *a, **kw: prepared,
                           tokenize=lambda *a, **kw: tokens, generate=generate)
    with pytest.raises((InterruptedError, StructuredOutputError)):
        TextGenerate.execute(clip, "hi", 10, {}, response_format="json_schema", thinking=True,
                             mtp="off", auto_resize=False, json_schema=json.dumps(SCHEMA))
