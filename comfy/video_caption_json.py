"""Video-caption adapters for the shared, request-owned native JSON grammar."""
import hashlib
import json
from types import SimpleNamespace

from comfy.text_json import PreparedJSON, StructuredOutputError, strict_json
from comfy.text_encoders.llama import BaseGenerate


def prepare_json(tokenizer, family, vocab_size, stops, response_format, json_schema):
    if response_format not in ('text', 'json_schema'):
        raise StructuredOutputError('format', 'Select text or json_schema; no automatic fallback is supported.')
    if response_format == 'text':
        if json_schema.strip():
            raise StructuredOutputError('format', 'Select json_schema to enforce the supplied schema.')
        return None
    prepared = PreparedJSON(SimpleNamespace(backend_tokenizer=tokenizer), json_schema, vocab_size, sorted(stops), family)
    prepared.evidence['request_schema_sha256'] = hashlib.sha256(json_schema.encode()).hexdigest()
    return prepared


def json_prompt(prompt, prepared, per_event=False):
    if prepared is None:
        return prompt
    scope = 'For each event caption, return' if per_event else 'Return'
    return prompt + '\n' + scope + ' exactly one JSON object matching this schema, with no Markdown or commentary:\n' + json.dumps(prepared.schema, ensure_ascii=False, separators=(',', ':'))


def check_json_prompt(prepared, ids):
    if prepared is not None:
        prepared.check_prompt({'caption': [[(token, 1.0) for token in ids[0].tolist()]]})


def select_token(logits, attempt):
    if attempt is None:
        return int(logits.argmax(-1).item())
    # Reuse the existing hard-mask and legal-logit checks, including NaN/dead-end
    # rejection. No repair, retry or unconstrained sampling is performed.
    return int(BaseGenerate().sample_token(logits, 1.0, 0, 1.0, 0.0, 1.0, [], None,
        do_sample=False, allowed_mask=attempt.allowed_mask(logits)).item())


def json_result(attempt, scope='answer'):
    text, _, evidence = attempt.finish()
    return dict(text=text, structured=strict_json(text), structured_output={**evidence, 'scope': scope})
