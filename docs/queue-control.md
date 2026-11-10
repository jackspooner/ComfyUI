# Pending prompt controls

This fork retains upstream [ComfyUI](https://github.com/Comfy-Org/ComfyUI) queue
execution. The local queue adapter adds persistent holds under its dequeue mutex.
No model executor, global interrupt, cancellation or queue clearing is involved.

`GET /queue/control` returns version 1, the current global control revision,
total retained identities and up to 200 ID/state entries. It returns no graphs.
`POST /queue/control` accepts exactly:

```json
{"action":"pause","promptIds":["recorded_native_prompt_id"],"operationId":"unique_control_id","expectedRevision":0}
```

Use `action:"resume"` with a new operation ID and the current revision to release
the same held prompt. Select 1–200 unique safe IDs. The entire selection is
validated before any mutation; running, unknown or terminal targets return 409
with `blocked:[{promptId,reason}]`. A confirmed response includes `receiptId`,
`revision`, `state`, `action`, `operationId` and the exact `promptIds`.
Retry an identical operation after a lost reply. Reusing an operation ID with
changed arguments, or after a newer control on the same IDs, returns a revision conflict.
An old acknowledgement cannot override a newer control. Read owner status again.

Access is loopback-only and refuses Origin, browser fetch and forwarded headers.
The private `user/queue-controls.sqlite` stores original executable payloads with
mode 0600. Preserve it with the original user directory and maintenance receipts.
Holds are indefinite; no expiry, automatic replay or background unpause occurs.
Normal `/queue` pending counts describe runnable work; held work is visible through
the control endpoint. Held identities reject a second `/prompt` acceptance.

Resume records durable intent before enqueue. After an interrupted resume, a
`resuming` or `released` row absent from queue/history is uncertain. Never submit a
replacement prompt or infer completion from absence. Reconcile its original
owner and native process. Retrying even a previously confirmed resume returns
`queue_control_uncertain` when its selected prompts lack queue/history evidence;
the old receipt cannot establish that the released work survived a restart.
A durable `paused` row can be explicitly resumed after
restart; matching maintenance receipts are restored only after digest validation.
Maintenance stays blocked while held executable work exists. It must not be
abandoned for a restart. Timing acceptance is retained and resume does not create
another acceptance event; paused wall time is not an execution-time measurement.

Whole-prompt control is inappropriate for one child in a shared stage prompt.
Use the native stage owner's child controls; routers own dependency closure,
all-item selection and any persisted submission fences. A 404 from an older
native server means unsupported control, never permission for delete/re-submit.

CPU checks: run the documented worktree setup, then `python -m pytest
tests-unit/app_test/queue_control_test.py tests-unit/app_test/test_maintenance.py
tests-unit/app_test/run_timing_test.py -q` with the prepared interpreter.
