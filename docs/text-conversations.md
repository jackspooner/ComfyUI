# Native text conversations

This fork extends the existing `CLIPLoader → TextGenerate` path with explicit,
versioned text/image history and exact context accounting. These source changes require
a ComfyUI restart before the running node exposes them. No model is downloaded,
implicit session stored, tool executed or external service contacted by this feature.

## Models and prerequisites

Complete native Qwen2.5-VL, Qwen3-VL, Qwen3.5/3.8, Qwen3-Omni,
Gemma3, Gemma4 and GLM-OCR support text/image history. Native MOSS and Audio
Flamingo retain text-only history. Use the existing loader and a complete
checkpoint; conditioning-only/truncated encoders are not made generative.
GLM-OCR remains a task-specific recognizer: replay support is not evidence of
conversational quality. Discover filenames from `/object_info/CLIPLoader`.
Tokenizers own formatting and models own their context limit and visual count.
Unsupported adapters fail explicitly.

## Node interface

All existing inputs and the first two output indices keep their meanings. New
inputs are optional:

| Input | Behavior |
| --- | --- |
| `history` | STRING containing version 1 text or version 2 image JSON. Connect `{"version":1,"messages":[]}` to start a conversation. Empty/omitted keeps the existing single-prompt path unless another new control enables conversation mode. |
| `prompt_content` | Optional STRING JSON array of ordered text/image parts. Requires empty `prompt` and no IMAGE socket. |
| `context_budget` | Total conversation token ceiling, including generated output. `0` uses the loaded model limit; a positive value enables text/image conversation mode and may only lower that limit. Does not apply to audio/video inputs. |
| `history_policy` | `error` (default) or `drop_oldest_turns`. The latter explicitly enables exclusion of complete old user/assistant pairs from this generation. Does not apply to audio/video inputs. |

Conversation mode uses `prompt` as a new user message. `system_prompt`, when
nonempty, replaces the leading system message; otherwise an existing one is
preserved. MOSS supplies its ordinary helpful-assistant default when absent.
Use `use_default_template=true`. IMAGE batches become individual image parts before
the current prompt; different turns can have different dimensions. Templated image
requests automatically enter history mode, including an image-only first turn.
Explicit history rejects raw-template bypass and audio/video. With empty `history`
and `prompt_content`, audio/video inputs automatically select the existing media
path regardless of `context_budget` or `history_policy`; those conversation-only
settings are ignored. `max_length` and the model's media-aware context checks still
apply. For example, MOSS audio with `context_budget=32000` needs no manual history
mode change. Raw prompts retain their existing behavior when history controls are absent.

Outputs are `generated_text` at 0, `thinking` at 1, `history` at 2 and
`context_info` at 3, and `raw_text` at 4. Ordinary templated text also returns its first turn as history
so it can connect directly to another generation; its generation path stays
unchanged and context_info is empty. Raw-prompt and audio/video paths return empty
history. Templated image paths return reusable history and exact context_info. Supply the empty envelope or a positive budget to budget the first turn
exactly. The specialized Generate LTX2 Prompt node retains its original inputs
and two outputs, and never snapshots images.

## History and state

```json
{"version":1,"messages":[
  {"role":"system","content":"Remember facts from our conversation."},
  {"role":"user","content":"My code is BLUE."},
  {"role":"assistant","content":"Understood."}
]}
```

Version 1 accepts only role and string content per message; version 2 also accepts ordered content arrays for user/assistant messages. There may be one leading
system message; remaining messages may be user or assistant. Role order is
preserved, including consecutive messages of the same role. Tools, unsupported media parts,
additional fields and unknown envelope versions fail before generation.

The returned history contains the complete normalized input and the new user and
final assistant message. Reasoning stays separate and is not replayed. Excluded
turns remain in returned history; they are absent only from this generation's
active context. The input is never mutated. Failures and cancellation return no
updated history; saving a successful result remains the caller's responsibility.

Connect output 2 to the next TextGenerate history input to chain turns. For a
later queued execution, explicitly save and load the JSON string or supply it
through an API caller. Graph feedback cycles and hidden persistent sessions are
not introduced. Use an existing SaveText output node to expose strings in
ComfyUI `/history`; sockets alone do not appear there automatically. Saved JSON
is ordinary plaintext and follows the caller's file retention policy.

## Image attachments

`prompt_content` is a JSON string encoding this array (not a separate prompt):

```json
[{"type":"text","text":"Compare this with the first image."},
 {"type":"image","file":{"filename":"second.png","subfolder":"chat"}}]
```

An image part may instead use `{"type":"image","image_id":"<64 lowercase hex characters>"}`.
The same parts are accepted in version 2 history message content. URLs, absolute
paths and escaping references are rejected. File references resolve only inside
ComfyUI's input directory, select the first frame, apply EXIF orientation and decode
RGB. IMAGE sockets preserve their exact supplied tensor pixels, shape and dtype.
Model-specific resizing/normalization happens only after resolution.

On successful generation, snapshots are published as safetensors in the configured
system-user directory's `text_history/images/` folder (normally
`user/__system/text_history/images/`). IDs hash dtype, shape and
pixel bytes. Atomic publication deduplicates identical snapshots. Returned histories
use IDs only; they do not depend on the original uploads. Copy the JSON **and all
referenced safetensors** together when backing up or migrating. Retention is explicit:
there is no expiry or automatic garbage collection. Branches may share attachments.
Deleting a RouterText chat does not delete these independently reusable files.

Generation/cancellation/write failures return no updated history and leave the
input unchanged. Temporary files are removed; already published shared attachments
are never rolled back (partial publication may leave an unused snapshot). Restore
missing/corrupt snapshots under their original IDs; do not substitute other pixels.
Traversal and escaping symlinks fail. File content participates in node cache
fingerprints; linked history that cannot be inspected invalidates conservatively.
Models/tokenizers receive tensors and content only, never IDs or filesystem paths.

## Context accounting

The model tokenizer formats role delimiters and the generation/thinking prefix
before counting. Text tokens are measured from the actual integer token stream;
visual tokens use the same geometry/configuration as preprocessing, without vision
inference during context selection. The actual expanded embeddings must match the
count. Excluding a turn excludes its images from inference while retaining all
references in returned history. No persistent GPU/session cache is introduced.

`input_tokens + output_allowance <= context_limit`. Positive `max_length`
reserves that many new tokens, including thinking. `max_length=0` uses remaining
capacity after the selected input and never removes valid history merely to
enlarge output. At least one output token must fit.

With `drop_oldest_turns`, only a leading complete user/assistant pair after the
optional system message is removable. Selection repeats until input fits. It
never splits messages, removes the system/current user message or guesses how to
pair an irregular transcript. An unremovable overflowing transcript fails. The
full history is not silently compacted or summarized.

`context_info` is version 1 JSON containing `model_context_limit`, `context_limit`,
`input_tokens`, `output_allowance`, `generated_tokens` and
`excluded_message_ranges`. Ranges are zero-based, half-open indices into the full
normalized input transcript, including any explicit system replacement and the
current user message. Generated tokens count the returned model IDs, including
thinking and any returned stop token. No unreported stop reason or cost is invented.

Context is a logical ceiling, not a memory-readiness guarantee. Native model,
KV allocation, MTP workspace and offload owners remain unchanged. There is no
persistent KV reuse; every turn replays its selected input. Larger output/context
can require substantially more memory. Do not retry a failed job with different
budgets or another model without a deliberate new request.

## API example

Use the exact discovered loader filename and the loader type for that model.
This is the generation portion of a normal `/prompt` graph:

```json
{
  "generate": {
    "class_type":"TextGenerate",
    "inputs":{
      "clip":["model",0],
      "prompt":"What is my code?",
      "history":"{\"version\":1,\"messages\":[{\"role\":\"user\",\"content\":\"My code is BLUE.\"},{\"role\":\"assistant\",\"content\":\"Understood.\"}]}",
      "use_default_template":true,
      "context_budget":2048,
      "history_policy":"error",
      "max_length":64,
      "sampling_mode":"off",
      "thinking":false,
      "mtp":"off"
    }
  },
  "save_history":{
    "class_type":"SaveText",
    "inputs":{"text":["generate",2],"filename_prefix":"text-history/history","format":"txt"}
  }
}
```

Queue through the normal supported API and follow the returned prompt ID. After
uncertain submission, inspect queue/history for that ID before retrying. Cancel
only the correlated job. On `Context overflow`, shorten input/output, increase a
lower working budget within the runtime limit, or explicitly select whole-turn
removal. Missing adapters require a supported model; schema validation errors
require correcting the envelope. Never clear another user's queue to recover.

## Development verification

Prepare a worktree with the root AGENTS.md setup command. Run:

```sh
"$COMFYUI_PYTHON" -m pytest tests-unit/comfy_test/image_history_test.py tests-unit/comfy_test/text_history_test.py tests-unit/comfy_test/text_generation_length_test.py tests-unit/comfy_test/moss_audio_test.py tests-unit/comfy_test/moss_audio_reference_test.py tests-unit/comfy_test/qwen_vl_image_channels_test.py -q
```

The tests cover exact boundaries, a real Qwen tokenizer, system replacement,
immutable branching, complete-turn exclusion, failure, LTX compatibility and
existing MOSS/media behavior. Model inference and long-context memory acceptance
are separate operational checks; no unit test establishes those properties.
