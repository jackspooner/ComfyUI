# Native constrained JSON text

This fork adds optional, offline llguidance constraints to **Load CLIP → Generate
Text**. Source implementation and CPU/tokenizer tests are available; **full-model
GPU acceptance and serving activation have not been performed**. A schema-valid
answer is a model interpretation, not verification of source observations,
advertising claims, citations or factual accuracy.

## Setup and use

Provision `requirements-text-json.txt` in the intended Python environment through
normal package tooling. It pins llguidance 1.9.1 (MIT) and jsonschema 4.26.0 (MIT).
Nothing is downloaded or installed by node execution. Ordinary text does not import
these optional dependencies. Readiness, without installation or model loading:

```sh
COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py --text-json
```

Set `response_format="json_schema"`, `mtp="off"`, `use_default_template=true`,
and supply a JSON string in `json_schema`, for example:

```json
{"type":"object","properties":{"caption":{"type":"string","minLength":1}},"required":["caption"],"additionalProperties":false}
```

Supply normal task instructions as well. Schema descriptions help prompting but
cannot ensure truth. Schema instructions consume input context; thinking and answer
tokens share the existing `max_length` output allowance. No budget is increased
and no reasoning-close token is forced.

`response_format="text"` remains the default, including the existing prompt-only
behavior of a nonempty `json_schema`; it does **not** imply native enforcement.
Existing API prompts may omit both JSON controls, including raw-template MOSS
audio calls. `json_schema` is optional at prompt admission; selecting native JSON
still requires a nonempty valid schema at execution. Native prompt-validation
regressions cover omission, linked audio and ordinary required-input rejection.
`json_object` and unknown formats are rejected, not downgraded. The specialized
Generate LTX2 Prompt node does not expose these controls.

The five output positions stay unchanged: `generated_text`, `thinking`, `history`,
`context_info`, `raw_text`. In native mode, `generated_text` is the exact decoded
answer span, strictly parsed and independently schema-validated before successful
history/image publication. No Markdown extraction, regex repair or heuristic
`</think>` split is used. Literal marker text inside JSON is preserved. `raw_text`
uses the authoritative tokenizer decoder over all generated IDs; the shared
ordinary CLIP decode contract is unchanged.

## Admission and supported families

| Family/package | Native schema path | Thinking |
| --- | --- | --- |
| Complete MOSS-Audio 8B Instruct/Thinking | Implemented, actual embedded tokenizer CPU tests | Verified Qwen single-token open/close protocol; answer-only output also permitted |
| Complete Qwen3-VL 8B / tied-head 4B | Implemented, bundled Qwen tokenizer and real template CPU tests | Rejected pending separate protocol acceptance |
| H3 Qwen3-VL 32B conditioning-only export | Rejected: truncated package lacks final norm/head | Rejected |
| Gemma3/4, Qwen3.5/3.8, Qwen3-Omni, GLM and other families | Not admitted in this milestone | Not admitted |

Gemma4's actual embedded tokenizer byte decoding was tested, but this is **not**
acceptance of its channel protocol or full constrained generation. No checkpoint
has been accepted by a full-model generation run for this feature yet.

Preparation requires an authoritative `tokenizers` backend, exact vocabulary width
and configured stop IDs. Older slow Transformers tokenizers without that backend
fail clearly; tested Transformers is 5.20.1 with tokenizers 0.23.3, not every version
allowed by the base ComfyUI requirements. No substituted tokenizer or HF download
is attempted. Loaded-key coverage is accumulated across shards. Missing model keys
reject native mode, and the admitted Qwen-family configuration requires its final
norm and separate **or tied** output head. This runtime check does not replace
converter provenance, checkpoint hashes or model-quality acceptance.

The node compiles on CPU before context/media planning or `CLIP.generate` can load
the model onto the execution device. The upstream CLIP loader may already have
constructed its CPU model/tokenizer. The actual prompt must end at the verified
family boundary; custom/raw templates are rejected. MOSS thinking begins outside
the grammar, permits the single `<think>`/`</think>` boundary, and activates JSON
only for the answer. Its nonthinking template includes the empty think block.

MTP is explicitly rejected even for packages without an MTP head; select `off`.
Only one sequence is supported. Speculative decoding, arbitrary custom templates,
general batching, new converters and persistent grammar caches are deferred.

## `comfy-json-v1` subset

- Root object; all objects closed with caller-supplied `additionalProperties:false`.
- Nested objects, homogeneous arrays, scalar types and scalar-plus-null type pairs.
- `properties`, `required`, `items`, scalar `enum`/`const`, `minimum`, `minLength`.
- Optional `title`, `description`, and `$schema` equal to
  `https://json-schema.org/draft/2020-12/schema`. Annotations are not constraints.
- Reject all other keywords, including refs, regex, formats, combinators,
  `x-guidance` and unsupported bounds. No reference resolution or network access.
- Limits: 128 KiB UTF-8 schema, schema depth 32, 4096 properties and 4096 enum/const
  values total. llguidance parser limits: 100000 grammar size, 50000 lexer states.
  Admission within the outer envelope does not promise compilation/completion;
  stricter grammar resource limits may reject a request.

Compilation uses `lenient=false`, `coerce_one_of=false`; warnings are fatal.
Property order follows schema insertion order: this deliberately selects a subset
of valid JSON documents rather than claiming arbitrary key-order generation.
Schema identity is SHA-256 of compact, insertion-order-preserving, UTF-8 JSON.

Sampling applies a hard `-inf` mask to greedy and sampled paths, preserves masks
through top-k gathering and reapplies them after min-p/top-p. Padded IDs and
inappropriate special tokens remain excluded; stops are allowed only when the
grammar permits them. Plain-text greedy penalty semantics are unchanged. Invalid
legal logits or matcher dead ends fail instead of unconstrained fallback.

Generation ends immediately at grammar acceptance, even on the final budget token;
normal completion evidence is `grammar_complete` (the matcher also recognizes legal
EOS). Incomplete reasoning/JSON fails. Each attempt, including OOM media-resize
retry, gets a fresh matcher. Final validation rejects duplicate keys, trailing
documents/prose, nonfinite numbers, incomplete UTF-8 and unpaired surrogates. Genuine
U+FFFD is permitted. No type coercion, defaults or answer rewriting occurs.

## Evidence and recovery

`context_info` is a JSON string. Native success adds `structured_output` containing:

`enforcement: "native_schema"`, `subset: "comfy-json-v1"`, `backend: "llguidance"`,
`backend_version`, `schema_sha256`, `tokenizer_sha256`, `family`, `vocab_size`,
`stop_tokens`, `property_order: "schema"`, `thinking_protocol` (`none` or
`moss-qwen-token-v1`), `template_suffix_sha256`, `finish_reason`, `validated: true`.

This is runtime enforcement evidence, not a checkpoint/package hash or factual
certification. Do not infer it for historical or prompt-only outputs. The node
adds no model registry, router policy, telemetry or workflow identifiers.

Errors use `structured_output.<stage>` prefixes: `format`, `schema`,
`backend_unavailable`, `capability`, `tokenizer`, `template`, `thinking`, `mtp`,
`compile`, `grammar`, `logits`, `reasoning_budget_exhausted`, `incomplete`,
`validation`. Correct deterministic setup/schema/capability failures rather than
repairing through another model. Budget failures require an explicit user decision
to change the budget/prompt. Existing cancellation, OOM and loading owners remain
in control. No successful history is published on failure.

## Verification record (24 October 2026)

CPU tests used Python 3.13.16, Torch 2.14.1+cu130, Transformers 5.20.1, tokenizers
0.23.3 and the two exact optional pins. The optional packages were installed only
in an isolated development target; the serving environment was not changed.

Cached release-wheel hashes used for this prototype:

- llguidance 1.9.1, `cp39-abi3-manylinux_2_31_x86_64`:
  `dfac6021d9c4c5ef16b3a1d336c05ca5e8208ff30dec29d1c63ebcafbee3a346`.
- jsonschema 4.26.0, `py3-none-any`:
  `d489f15263b8d200f8387e64b4c3a75f06629559fb73deb8fdfb525f2dab50ce`.

Other platform wheels were not tested. The requirements file pins releases, not
these platform-specific hashes or the serving environment's transitive packages.

```sh
python -m pytest tests-unit/comfy_test/text_json_native_test.py -q
```

Set `COMFY_JSON_MODEL_FIXTURES` to an existing local text-encoder directory to also
read **only** headers and embedded tokenizer tensors from
`moss_audio_8b_thinking_bf16.safetensors`,
`moss_audio_8b_instruct_bf16.safetensors`, and `gemma4_12b_bf16.safetensors`.
Otherwise those three tests skip. The module skips when llguidance is absent;
run the existing text/history/media/family regression suites separately without
the optional dependency path to verify ordinary operation.

Six schema-only fixtures preserve the exact distinct revision-46 workflow schemas,
including nullable scalar pairs and the 12,102-byte largest schema. Their filenames
are compact-schema SHA-256 identities. `scripts/extract_text_json_schemas.py` can
reproduce them from an explicitly supplied local snapshot; it refuses overwrite
and copies no prompts, media or source paths. No saved workflow was changed.

On Ryzen 7 9700X CPU, one sequential six-schema replay measured request preparation
(including tokenizer construction and compilation) at 0.582–0.636 seconds/schema.
Across 1147 answer-token masks, median was 0.030 ms and p95 0.078 ms, excluding
sampling/model work. Process peak RSS rose from 984012 to 1318512 KiB during the
whole replay; this is not isolated allocator accounting. No torch.compile helper
is used: the CPU byte mask is converted to a request-local Boolean tensor. CUDA
transfer/synchronization overhead, model latency, cancellation during real compute,
VRAM behavior and other accelerators remain unmeasured. These numbers are not an
end-to-end performance promise.

Activation is a separate operation: install the declared optional dependencies,
deploy tested source safely, rediscover sockets, and perform explicitly authorized
normal-queue model acceptance before enabling a route. This source change does not
restart services, resume a corpus, promote candidates or certify observations.
