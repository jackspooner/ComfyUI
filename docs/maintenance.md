# Local maintenance fencing and prompt receipts

This fork implements an **opt-in, single-process/single-native-worker** maintenance
owner. It closes prompt admission, drains every already accepted prompt (including
pending prompts), commits completion evidence, and starts again behind the same
durable fence. It does not deploy, stop a service, cancel work, replay executable
queues, or checkpoint tensors. The feature makes no outbound requests.

## Startup and authentication

Set both environment variables in the existing service's startup environment:

- `COMFYUI_MAINTENANCE_DIR`: absolute, persistent local directory, owned by the
  service user, mode `0700`. Missing directories are created. Keep it outside the
  checkout and preserve it across deployments/rollbacks. SQLite schema version 2
  and `owner.lock` live here. A lifetime `flock` prevents a second owner.
  Version-1 receipt stores are upgraded by adding history visibility; unknown
  schema versions fail startup rather than being silently overwritten.
  Rollback binaries must support the retained schema; never restore an older
  database over newer evidence to accommodate an incompatible binary.
- `COMFYUI_MAINTENANCE_TOKEN_FILE`: absolute regular file owned by that same user,
  no group/other permissions, not a symlink. Contents: a cryptographically random
  32–4096-character non-whitespace ASCII token, optionally followed by a newline.
  Provision securely; never put the token in command arguments, logs or Git.

Optional `COMFYUI_MAINTENANCE_MAX_RECEIPTS` sets the receipt count limit at startup
(default 10,000; positive decimal integer of at most nine digits, or the exact
value `unlimited` to disable only the receipt-count limit). Invalid values
fail startup. Size this for the expected seven-day prompt volume and available
disk and memory: terminal history is restored in memory at startup. For example,
`100000` allows a larger OCR corpus while retaining existing receipt evidence.
`COMFYUI_MAINTENANCE_MAX_RECEIPTS=unlimited` admits receipts without a count cap;
it does not provide unlimited disk or memory. Seven-day terminal pruning, the
per-receipt payload limit, operation limit and unresolved-work safeguards remain.
Changing the setting requires a coordinated safe restart with the same store;
it does not clear receipts, change retention, release a fence or resolve unknown
work. Lowering the limit below the retained count preserves existing receipts and
rejects new admissions until eligible expiry or a subsequent capacity increase.

Partial/invalid configuration fails startup. Without either variable, the normal
`PromptQueue` is used and maintenance routes are absent. **Do not remove these
variables, switch directories, restore old databases or run an old binary to
bypass a fence.** The first installation cannot fence an already running old
binary and needs a separately verified bootstrap window.

Every `/maintenance` route (reads included) requires `Authorization: Bearer TOKEN`,
a loopback TCP peer, and a loopback/`localhost` Host. IPv4-mapped IPv6 loopback peers
are supported. Browser `Origin`/`Sec-Fetch-Site` and proxy forwarding headers are
rejected, regardless of CORS settings. Use the direct local service endpoint, not
a reverse proxy or browser. This is one operator credential; `ownerId` provides
logical ownership/conflict detection, not separate credential-based identities.
The usual `/api` route aliases have identical protections.

The normal prompt/history endpoints retain their existing access policy. Receipt
outputs can contain private generated text; the database is private, but restoring
native history does not add a new authentication policy to `/history`.

## Version-1 HTTP contract

Strings in command bodies match `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`; revisions are
nonnegative JSON integers (not booleans). Unknown/missing fields are rejected.
Use fresh, never-reused UUIDs for operations and prompt IDs.

### `GET /maintenance`

Returns one queue-mutex-consistent snapshot:

```json
{
  "schemaVersion": 1,
  "revision": 2,
  "epoch": 1,
  "generation": "process-uuid",
  "invocationId": "systemd-INVOCATION_ID-or-process-uuid",
  "operationId": "maintenance-uuid",
  "ownerId": "deployment-cli",
  "targetRevision": "full-target-git-commit",
  "phase": "ready",
  "accepting": false,
  "safeToStop": true,
  "counts": {
    "pending": 0, "running": 0, "validating": 0, "finishing": 0,
    "publishing": 0, "accepted": 0, "interruptedUnknown": 0
  },
  "blockers": []
}
```

`phase` is `open`, `draining`, `ready` or `blocked`. When no operation is active,
the three operation identity fields are null; `phase=open` alone is **not** an
acceptance/readiness guarantee—check booleans and blockers. `revision` changes on
startup, start and release, not each drain progress update. `epoch` increases only
when a new fence is acquired; `generation` changes every process start.

`safeToStop=true` means this exact active fence is held, there are no native
validation reservations, pending/running prompts, unpublished completions, prior
unknown receipts or storage faults, and the native worker has entered its next
dequeue after completion notifications/housekeeping. Counts are sampled under the
same mutex as admission and operator mutations. A receipt commit precedes removal
from the running set. Failed/cancelled executions can be terminal; readiness does
not mean every generation succeeded. The idle native worker does not perform its
periodic housekeeping while fenced; already accepted pending prompts still drain.

This is **native prompt-stop safety**, not universal process/GPU quiescence:
background custom-node threads, asset indexing, direct executor calls, external
resource owners and producer continuations require their own checks. The CLI must
bind `operationId`, `ownerId`, `epoch`, `revision`, `generation`, target and systemd
`invocationId`, verify producers and GPU hold ownership, and stop that invocation
without another actor releasing the gate. Unreachable/stale status is not ready.
There is no timeout or lease expiry that silently opens the fence.

### `POST /maintenance/start`

```json
{"operationId":"maintenance-uuid","ownerId":"deployment-cli","targetRevision":"full-target-git-commit","expectedRevision":1}
```

Persistently fences admission before returning the status snapshot, plus
`command: {"action":"start","operationId":"maintenance-uuid"}`. Same active
identity/owner/target is an idempotent attach even with the original revision.
A different active identity conflicts; changing the owner/target of an existing
identity conflicts. A released identity returns `operation_already_released`,
never reacquires a gate. Starting while receipts are blocked is permitted so the
operator can establish ownership for investigation/publication repair; it does
not discard blockers or claim the drain can finish.

### `POST /maintenance/resume` and `/maintenance/abort`

```json
{"operationId":"maintenance-uuid","ownerId":"deployment-cli","expectedRevision":2,"expectedGeneration":"process-uuid"}
```

Both require the exact active owner, current revision/generation and `safeToStop`.
They release **only that operation** without cancelling or clearing any queue.
`abort` is allowed only in the process generation that started the fence; after a
restart use `resume` only after deployment/rollback verification. This prevents an
old abort request reopening an unverified restarted service. Exact successful
release requests are retained and replay without releasing any newer operation.
Responses include the current snapshot and `command` action/operationId; on a
normal release the top-level `operationId` is null. A replay while another
operation is active reports that current operation and leaves it fenced.

### `POST /maintenance/retry-publication`

Same body as `resume`. Explicitly retry writing retained in-memory completion
results after restoring storage, without rerunning any node. Requires the active
owner and current revision/generation. Successful writes remove corresponding
`publishing` blockers; an explicit successful database write probe clears the
transient storage fault. Unknown/missing receipts still block readiness. Repeating
the same request is harmless; this action does not release the gate or change its
revision. This is not crash recovery of executable work.
The HTTP 200 snapshot additionally includes `publicationErrors`, an array of
`{promptId,code}` for entries still unrepaired (empty on success). Entries are
retried independently, so one invalid result cannot prevent other publications.
A 200 response does not imply repair or readiness: inspect this array and blockers.

### `GET /maintenance/receipts/{prompt_id}`

Returns `{id,number,state,created,terminal,history,visible}`. `created`/`terminal`
are Unix seconds (`terminal` may be null). States are `accepted`, `terminal`, or
`interrupted_unknown`. History is null until committed, otherwise a native-like
`{prompt,outputs,status}` projection. `visible` is 0/1 for native history display.
Missing receipt: HTTP 404, `{"error":{"type":"receipt_not_found"},"acceptance":"unknown"}`;
absence never proves rejection. Internal digests are not exposed.

### Errors and native `/prompt`

Errors use `{error:{type,message,details,extra_info:{}},accepted,node_errors:{}}`.
HTTP 400: malformed commands/input; 401/403: auth/local-origin restrictions;
409: operation/identity/revision conflict or not ready; 503: fenced, capacity or
publication/storage failures. Codes include `maintenance_fenced`,
`revision_conflict`, `operation_conflict`, `operation_identity_conflict`,
`operation_already_released`, `abort_after_restart_forbidden`,
`maintenance_not_ready`, `maintenance_drain_protected`, `receipt_store_failure`,
`receipt_serialization_failure`, `completion_publication_failure`, `receipt_missing`, `receipt_capacity`,
`operation_capacity`, `interrupted_work_unresolved`, and `invalid_receipt_input`.

Native `POST /prompt` while fenced returns **503 `maintenance_fenced`,
`accepted:false`**: this attempt did not enqueue. It does not deny acceptance of a
previous attempt with the same ID. Validation already in flight is counted and
rechecked atomically at final `put`; it is rejected if fencing won the race.
An error with `accepted:null`, a disconnected request or an unstructured failure
is uncertain—read the original ID, never blindly resubmit.

With maintenance configured, retained same-ID/same-digest queue submissions do
not reexecute. Native response adds `replayed:true` and `receiptState` to the
usual `{prompt_id,number,node_errors}`. Changed content returns 409
`prompt_id_conflict`. Digest covers the post-replacement prompt, outputs selected,
extra data except `create_time`/`client_id`, and sensitive execution arguments;
only its SHA-256 is saved, not these inputs. Changing metadata or credentials can
therefore conflict intentionally. Queue priority and server timestamps do not.
Custom producers must handle the typed `PromptReplay`/`MaintenanceError` outcome,
not assume every `put` succeeds. The HTTP admission precheck rejects all new
attempts while fenced/unknown; receipt reads remain available for recovery.

## Persistence, retention and failure recovery

This is a receipt journal, **not an executable queue**. It saves no prompt graph,
extraData, credentials, arbitrary error payloads or node inputs. Terminal outputs
(including generated text and file descriptors), creation time, success/error
status and allowlisted lifecycle event names are retained. Raw error details and
workflow/asset metadata are not restored. `outputs` are node-authored data, not a
general secret scrubber. Artifact files themselves must remain available; the
journal does not checksum/copy every output or guarantee external durability.

The Jobs API omits optional `execution_error` details for sanitized or incomplete
error events; the job remains failed, and raw retained status/outputs are unchanged.
Emitting a prompt-ID-only marker as a full error object makes strict frontend
history-page validation fail, also hiding history-backed generated Assets. This
applies at finite and unlimited capacities. The read-side correction handles
existing receipts without rewriting, deleting or replaying them. The separate
asset database API is opt-in (`--enable-assets`); its disabled state is not proof
that the history-backed generated-assets view should be empty.

Native history loads the sanitized terminal projection after restart. Clearing
or deleting native history persists its display suppression, but intentionally
does not erase protected receipt/idempotency evidence. By default 10,000 receipts
are admitted (startup-configurable as above), with a 4 MiB strict-JSON terminal payload limit. Terminal receipts
older than seven days are pruned on new admission. Unknown/accepted receipts are
never silently aged away. With a finite count limit, admission at capacity fails rather than evicting
unresolved work. Operation identities are retained up to 1,024 operations and
then fail closed; there is no automatic operation-archive/reset endpoint in v1.
Never reuse an expired ID; bounded retention is not lifetime exactly-once.

`receipt_capacity` can occur while `GET /maintenance` reports `phase=open` and
`accepting=true`: those fields describe the gate and recovery blockers, not
remaining journal capacity. A restart with unchanged capacity does not repair
this condition. Preserve retained evidence and increase configured capacity in a
verified safe activation window; do not delete the store or hide history as a fix.

Publication failure closes admission and blocks readiness, retaining completion
data in memory while the native worker continues draining previously accepted
pending prompts. Restore disk/permissions and call `retry-publication` **before
restart**. Strict-JSON/size failures require investigation; no silent truncation
or fabricated terminal success is used. V1 has no endpoint to replace or discard
an invalid retained completion. Such a result remains blocked; restarting merely
turns its receipt into unknown, not a successful repair. A native history or
notification publication failure after receipt commit is retriable without
writing that receipt again. Ordinary `/queue` counts can include finished entries
awaiting publication; maintenance separates these as `publishing` rather than
`running`. An uncontrolled restart loses unpublished
in-memory results: all persisted accepted-but-not-terminal receipts become
`interrupted_unknown`, and the executable queue stays empty. They block both
admission and readiness, even if no operation was active at the crash. Neither
resume, abort nor publication retry acknowledges these away. V1 has no generic
unknown-work reconciliation endpoint: retain evidence and escalate for explicit
operator recovery; never delete the database or replay to make it look idle.

## Rollout and bypass limits

Fence upstream producers only at their safe continuation/collection boundaries,
then start native drain. Otherwise a repair call or custom-node requeue can be
definitively rejected mid-logical-operation. Every caller using the server's
`prompt_queue.put` is fenced, including inspected Cutlery-Remote direct enqueues
and VideoHelperSuite requeues. Their custom validation before `put` is not a
native `/prompt` reservation; those routes need producer-level coordination.
Direct heap mutation, replacing the server queue, base-class calls that bypass
the adapter, direct `PromptExecutor` calls and background workers are outside the
contract. The installed Cutlery `/cutlery/restart` shortcut globally interrupts
and clears queues; **do not use it for this protocol**. This change does not patch
or secure third-party restart/shutdown routes. Restrict those independently.

Preserve the exact existing systemd launch flags (no added `--fast-disk`), token,
store and service invocation identity across activation. Do not release foreign
GPU holds. Start the tested binary with the same store, verify the new generation
is still fenced and receipts are readable, independently verify deployment, then
resume explicitly. No live service edits/restarts or GPU inference are part of
the isolated CPU fixture acceptance.

Tests: `tests-unit/app_test/test_maintenance.py`, `test_prompt_worker.py`,
`tests-unit/execution_test` and `tests-unit/jobs_cancel_test`, with the standard
worktree readiness command and declared test dependencies. Set
`comfy.cli_args.args.cpu=True` before importing execution/server for CPU fixtures.
