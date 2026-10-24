"""Request-owned, offline structured text constraints. Optional dependencies are lazy."""
import hashlib
import json
import math

import torch


class StructuredOutputError(ValueError):
    def __init__(self, stage, message):
        self.stage = stage
        super().__init__(f"structured_output.{stage}: {message}")


def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise StructuredOutputError("validation", "Duplicate object key.")
            out[key] = value
        return out

    def constant(value):
        raise StructuredOutputError("validation", "Nonfinite JSON number.")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise StructuredOutputError("validation", "Expected one complete JSON document.") from error

    def check(item):
        if isinstance(item, str):
            # Paired JSON escapes decode to a scalar; unpaired surrogates are rejected.
            try:
                item.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise StructuredOutputError("validation", "Unpaired Unicode surrogate.") from error
        elif isinstance(item, float) and not math.isfinite(item):
            constant(item)
        elif isinstance(item, dict):
            for key, child in item.items():
                check(key)
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
    check(value)
    return value


def dependencies():
    try:
        import llguidance
        import jsonschema
    except ImportError as error:
        raise StructuredOutputError("backend_unavailable", "Install requirements-text-json.txt in the selected ComfyUI environment.") from error
    return llguidance, jsonschema


def admit_schema(text):
    try:
        size = len(text.encode("utf-8")) if isinstance(text, str) else 0
    except UnicodeEncodeError as error:
        raise StructuredOutputError("schema", "Schema contains an unpaired Unicode surrogate.") from error
    if not isinstance(text, str) or not text.strip() or size > 131072:
        raise StructuredOutputError("schema", "Schema must be a JSON string of at most 128 KiB.")
    schema = strict_json(text)
    _, validator = dependencies()
    try:
        validator.Draft202012Validator.check_schema(schema)
    except validator.SchemaError as error:
        raise StructuredOutputError("schema", "Invalid draft 2020-12 schema.") from error
    counts = [0, 0]

    def visit(node, path, depth):
        if depth > 32 or not isinstance(node, dict):
            raise StructuredOutputError("schema", f"{path}: object schemas only, maximum depth 32.")
        allowed = {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "minimum", "minLength", "title", "description", "$schema"}
        for key in node:
            if key not in allowed:
                raise StructuredOutputError("schema", f"{path}/{key}: unsupported by comfy-json-v1.")
        if "$schema" in node and node["$schema"] != "https://json-schema.org/draft/2020-12/schema":
            raise StructuredOutputError("schema", f"{path}/$schema: only draft 2020-12 is supported.")
        kind = node.get("type")
        if isinstance(kind, list):
            if len(kind) != 2 or "null" not in kind or not set(kind) <= {"string", "number", "integer", "boolean", "null"}:
                raise StructuredOutputError("schema", f"{path}/type: only scalar-plus-null unions are supported.")
            kind = next(item for item in kind if item != "null")
        if kind is None and not ("enum" in node or "const" in node):
            raise StructuredOutputError("schema", f"{path}: an explicit type, enum or const is required.")
        if kind == "object":
            if node.get("additionalProperties") is not False:
                raise StructuredOutputError("schema", f"{path}/additionalProperties: closed objects are required.")
            props = node.get("properties", {})
            if not set(node.get("required", [])) <= props.keys():
                raise StructuredOutputError("schema", f"{path}/required: names must exist in properties.")
            counts[0] += len(props)
            for key, child in props.items():
                visit(child, path + "/properties/" + key.replace("~", "~0").replace("/", "~1"), depth + 1)
        elif any(key in node for key in ("properties", "required", "additionalProperties")):
            raise StructuredOutputError("schema", f"{path}: object keywords require type object.")
        if kind == "array":
            if "items" not in node:
                raise StructuredOutputError("schema", f"{path}/items: homogeneous item schema required.")
            visit(node["items"], path + "/items", depth + 1)
        elif "items" in node:
            raise StructuredOutputError("schema", f"{path}/items: requires type array.")
        if "minimum" in node and kind not in ("integer", "number"):
            raise StructuredOutputError("schema", f"{path}/minimum: requires numeric type.")
        if "minLength" in node and kind != "string":
            raise StructuredOutputError("schema", f"{path}/minLength: requires string type.")
        values = node.get("enum", []) + ([node["const"]] if "const" in node else [])
        counts[1] += len(values)
        for value in values:
            if isinstance(value, (dict, list)):
                raise StructuredOutputError("schema", f"{path}: enum/const values must be scalars.")
            if not validator.Draft202012Validator({k: v for k, v in node.items() if k not in ("enum", "const")}).is_valid(value):
                raise StructuredOutputError("schema", f"{path}: enum/const value contradicts constraints.")
        if counts[0] > 4096 or counts[1] > 4096:
            raise StructuredOutputError("schema", "Maximum 4096 properties and 4096 enum/const values.")
    visit(schema, "$", 0)
    if schema.get("type") != "object":
        raise StructuredOutputError("schema", "Root must have type object.")
    return schema


class PreparedJSON:
    def __init__(self, tokenizer, schema_text, vocab_size, stop_tokens, family, thinking=False):
        llg, validator = dependencies()
        self.schema = admit_schema(schema_text)
        self.validator = validator.Draft202012Validator(self.schema)
        # Only authoritative Rust tokenizer serialization is admitted. Older slow
        # Transformers tokenizers fail explicitly instead of per-token decode guesses.
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise StructuredOutputError("tokenizer", "This route requires a verified tokenizers backend; slow tokenizers are not admitted.")
        tokenizer_json = backend.to_str()
        vocab = backend.get_vocab()
        if not stop_tokens or max(vocab.values()) >= vocab_size or any(t not in vocab.values() for t in stop_tokens):
            raise StructuredOutputError("tokenizer", "Tokenizer IDs/stops do not fit model vocabulary.")
        self.vocab_size = vocab_size
        self.backend = backend
        self.stops = tuple(stop_tokens)
        self.tokenizer = llg.LLTokenizer(tokenizer_json, n_vocab=vocab_size, eos_token=list(self.stops))
        self.valid_ids = set(vocab.values())
        self.invalid_ids = set(range(vocab_size)) - self.valid_ids
        self.special_ids = {i for i in self.valid_ids if self.tokenizer.is_special_token(i)}
        self.reasoning_mask = bytes(int(i in self.valid_ids and i not in self.special_ids and i not in self.stops) for i in range(vocab_size)) if thinking else None
        self.thinking = thinking
        self.open_id = vocab.get("<think>")
        self.close_id = vocab.get("</think>")
        if thinking and (family != "moss_audio" or self.open_id is None or self.close_id is None
                         or backend.encode("<think>", add_special_tokens=False).ids != [self.open_id]
                         or backend.encode("</think>", add_special_tokens=False).ids != [self.close_id]
                         or self.tokenizer.decode_bytes([self.open_id]) != b"<think>"
                         or self.tokenizer.decode_bytes([self.close_id]) != b"</think>"):
            raise StructuredOutputError("thinking", "Constrained thinking is currently supported only by MOSS's verified Qwen token protocol.")
        self.whitespace_ids = {i for i in self.valid_ids - self.special_ids if self.tokenizer.decode_bytes([i]) and not self.tokenizer.decode_bytes([i]).strip(b" \t\r\n")}
        self.limits = llg.LLParserLimits(max_grammar_size=100000, max_lexer_states=50000, verbose_errors=False)
        try:
            self.grammar = llg.LLMatcher.grammar_from_json_schema(self.schema, overrides={"lenient": False, "coerce_one_of": False})
            self.template_matcher = llg.LLMatcher(self.tokenizer, self.grammar, limits=self.limits, log_level=0)
        except ValueError as error:
            raise StructuredOutputError("compile", "Schema cannot be compiled within the native grammar policy/limits.") from error
        if self.template_matcher.is_error() or self.template_matcher.get_grammar_warnings():
            raise StructuredOutputError("compile", "Grammar compilation failed or reported warnings; no weakening is permitted.")
        self.evidence = {
            "enforcement": "native_schema", "subset": "comfy-json-v1", "backend": "llguidance",
            "backend_version": llg.__version__, "schema_sha256": hashlib.sha256(json.dumps(self.schema, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_json.encode()).hexdigest(), "family": family,
            "vocab_size": vocab_size, "stop_tokens": list(self.stops), "property_order": "schema",
            "thinking_protocol": "moss-qwen-token-v1" if thinking else "none",
        }
        suffix = "<|im_start|>assistant\n"
        if family == "moss_audio" and not thinking:
            suffix += "<think>\n\n</think>\n\n"
        self.prompt_suffix = backend.encode(suffix, add_special_tokens=False).ids
        self.evidence["template_suffix_sha256"] = hashlib.sha256(suffix.encode()).hexdigest()

    def check_prompt(self, tokens):
        if not isinstance(tokens, dict) or len(tokens) != 1:
            raise StructuredOutputError("template", "Expected one verified tokenizer stream.")
        rows = next(iter(tokens.values()))
        if len(rows) != 1 or [t[0] for t in rows[0][-len(self.prompt_suffix):]] != self.prompt_suffix:
            raise StructuredOutputError("template", "Prompt does not end at the supported family answer boundary.")

    def new_attempt(self):
        return JSONAttempt(self)


class JSONAttempt:
    def __init__(self, prepared):
        self.prepared = prepared
        self.matcher = prepared.template_matcher.deep_copy()
        self.phase = "start" if prepared.thinking else "answer"
        self.answer_ids = []
        self.reasoning_ids = []
        self.raw_ids = []
        self.finish_reason = None
        self.stop_tokens = prepared.stops

    def allowed_mask(self, logits):
        p = self.prepared
        if logits.shape != (1, p.vocab_size):
            raise StructuredOutputError("tokenizer", "Only one sequence with the prepared logits width is supported.")
        if self.phase == "reasoning":
            allowed = bytearray(p.reasoning_mask)
            allowed[p.open_id] = 0
            allowed[p.close_id] = 1
        else:
            allowed = bytearray(self.matcher.compute_logit_bias())
            if self.matcher.is_error():
                raise StructuredOutputError("grammar", "Matcher resource limit or dead end.")
            for token in p.special_ids - set(p.stops):
                allowed[token] = 0
            for token in p.invalid_ids:
                allowed[token] = 0
            if self.phase in ("start", "separator"):
                for token in p.whitespace_ids:
                    allowed[token] = 1
            if self.phase == "start":
                allowed[p.open_id] = 1
        # compute_logit_bias is a byte mask, never a finite logit bias.
        return torch.frombuffer(allowed, dtype=torch.uint8).to(device=logits.device, dtype=torch.bool).unsqueeze(0)

    def commit(self, token):
        p = self.prepared
        self.raw_ids.append(token)
        if self.phase == "start" and token == p.open_id:
            self.phase = "reasoning"
            return False
        if self.phase == "reasoning":
            if token == p.close_id:
                self.phase = "separator"
            else:
                self.reasoning_ids.append(token)
            return False
        if self.phase in ("start", "separator") and token in p.whitespace_ids:
            return False
        self.phase = "answer"
        if not self.matcher.consume_token(token) or self.matcher.is_error():
            raise StructuredOutputError("grammar", "Sampled token rejected by matcher.")
        if token not in p.stops:
            self.answer_ids.append(token)
        if self.matcher.is_accepting():
            self.finish_reason = "eos" if token in p.stops else "grammar_complete"
            return True
        return False

    def finish(self):
        if self.finish_reason is None or not self.matcher.is_accepting():
            stage = "reasoning_budget_exhausted" if self.phase in ("start", "reasoning", "separator") else "incomplete"
            raise StructuredOutputError(stage, "Token budget ended before a complete answer.")
        try:
            text = self.prepared.tokenizer.decode_bytes(self.answer_ids).decode("utf-8", errors="strict")
            reasoning = self.prepared.tokenizer.decode_bytes(self.reasoning_ids).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise StructuredOutputError("validation", "Invalid UTF-8 token stream.") from error
        value = strict_json(text)
        if not self.prepared.validator.is_valid(value):
            raise StructuredOutputError("validation", "Generated answer failed independent JSON Schema validation.")
        return text, reasoning, {**self.prepared.evidence, "finish_reason": self.finish_reason, "validated": True}

    def raw_text(self):
        return self.prepared.backend.decode(self.raw_ids, skip_special_tokens=False)
