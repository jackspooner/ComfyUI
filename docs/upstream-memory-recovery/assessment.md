# Upstream memory recovery assessment

Assessed on 18 October 2026. Prepare **two independent PRs**, not the original
fork commit wholesale. Neither candidate has been merged into the serving fork.
No PR, issue, comment, deployment or restart was made for this assessment.
Candidate branches are preserved on the personal fork; no main branch is changed.

## Delivery receipts

| Branch on the personal fork | Candidate commit | Production / total files |
| --- | --- | --- |
| [upstream-pin-registration-recovery](https://github.com/jackspooner/ComfyUI/tree/upstream-pin-registration-recovery) | `663ffe61bc359b1a16d4d54b3ba6f73f43fb9668` | 1 / 2 |
| [upstream-partial-prefetch-cleanup](https://github.com/jackspooner/ComfyUI/tree/upstream-partial-prefetch-cleanup) | `c96b7d086e4b47301506968cddce2033d9ad2f19` | 2 / 3 |

Both candidates are single commits whose direct parent is `872eb6a3`. Both were
pushed with upstream tracking to `personal`; neither has an integration commit.
The assessment and its README link are on a third branch based on fork main.
The three task-owned worktrees and indexes are retained for review because the
candidates are deliberately unmerged. Their only untracked files are the local
GitNexus guidance installed by its owner. No unrelated worktree was removed.

## Scope and provenance

- Fork starting point: committed `main`, `7aede7957b21300da9797392473bf7110ee65ce2`,
  equal to `personal/main` after fetching. Unrelated canonical working files were
  left in place and were not copied or stashed.
- Source repair: `d5ab5f8682df91e1deef567ef9696a440eae7ef4`.
- Upstream baseline: `Comfy-Org/ComfyUI` `master`,
  `872eb6a3a5497721f53ebe82c0da9a3ba3414c13`, refreshed before preparation.
- Assessment branch: `upstream-memory-recovery-assessment`, based on fork main.
- Independent comparison branches start directly from the upstream baseline:
  `upstream-pin-registration-recovery` and `upstream-partial-prefetch-cleanup`.
  This deliberately excludes our fork's model integrations and other commits.
- Only the three relevant production-file edits were extracted from the source
  commit. Tests were strengthened for this assessment. Fork setup, README,
  AGENTS, model integrations and reasoning changes are excluded from both patches.
  Upstream's existing README, credits and license are preserved; its documented
  interfaces and installation requirements are unchanged by these repairs.

## Results

| Candidate | Unchanged upstream, new regression module | With its own fix | Adjacent CPU checks included |
| --- | --- | --- | --- |
| Registration retry | 7 failed, 3 passed | 10 passed | 55 passed total |
| Partial prefetch | 4 failed, 4 passed | 8 passed | 53 passed total |

These are deterministic tests of real Python owner paths with injected CUDA,
host-buffer and VBAR boundaries. They establish failure ordering, reachability
and accounting; they do not reproduce driver-level memory exhaustion or prove
the cause of a reporter's model workflow. No GPU/model workload was run.

### Registration retry

`pin_memory()` originally held a temporary view across `free_registrations()` and
retried registration with its old pointer. Eviction can shorten the shared host
buffer, so retry can address discarded memory; rollback can also truncate to an
offset beyond the current allocation. Separately, `HostBuffer.extend()` can grow
the buffer before raising, which the old `extended` flag failed to roll back.

The fix discards the uncommitted view, rolls back actual growth, then evicts and
allocates again at the current offset. Both registration attempts share this
cleanup. Accounting is published only on success; rollback failures propagate
without a second rollback attempt. The successful first attempt, unchanged-offset
retry, exhausted retry, partial allocation failures and failure during retry are
covered. One test traverses the real registration eviction policy and model
selection using a fake model's RAM-eviction boundary for an inactive fast-pin
buffer; it verifies that the new pin's offset and counters match the shortened
buffer. Native CUDA registration and native HostBuffer allocation remain mocked.

Production scope: `comfy/pinned_memory.py` only.

### Partial prefetch

`prefetch_queue_pop()` originally replaced the raw next-module entry only after
`pin_modules()` returned. On failure, terminal cleanup skipped that raw entry.
Also, a newly faulted nonresident module did not publish `_prefetch` until its
preparation finished, so recording the batch alone missed the failing module.

The fix registers the batch with queue cleanup before pinning, and publishes the
minimal per-module cleanup state before preparation only for batched prefetch.
The exception still propagates. Existing cleanup unpins each successful fault
once and removes incomplete state. Tests cover a resident module followed by a
preparation failure, failures at every batch position, a null fault signature,
failure in real `pin_modules()` after casting, successful consumption, repeated
cleanup and the unchanged direct-cast failure path.

Removing either production edit proves it necessary: queue-only has 1 failure;
ops-only has 5 failures. Both together pass all 8 regression cases.

Production scope: `comfy/model_prefetch.py` and `comfy/ops.py`.

## Existing upstream work

Read the actual diffs, not just their titles. Searches covered issues and PRs
with `HostBuffer.truncate`, `cudaHostRegister`, `pin_memory`, `vbar_unpin`,
registration, partial prefetch, prefetch cleanup and prefetch failure. Results are
a dated search, not a guarantee that no duplicate exists.

| Work inspected | Relationship |
| --- | --- |
| [#16901](https://github.com/Comfy-Org/ComfyUI/pull/16901), open, `a901b8cd` | Waits for async copies before stealing pins. Distinct from registration retry and incomplete-batch cleanup. Its patch applies cleanly over each candidate. |
| [#16615](https://github.com/Comfy-Org/ComfyUI/pull/16615), open, `2d25dc1a` | Falls back when the cast buffer cannot grow. Avoids one possible failure trigger but does not make other failures cleanable. Its patch applies cleanly over each candidate. |
| [#15944](https://github.com/Comfy-Org/ComfyUI/pull/15944), open, `df23b95a` | Broad Windows pin-budget, eviction and scheduling work. Rewrites the retry and prefetch regions; still retries the same temporary view. Needs manual reconciliation if it lands. Its current patch already fails to apply to current master as well as the candidates. No combined runtime validation claimed. |
| [#16481](https://github.com/Comfy-Org/ComfyUI/pull/16481), open, `7fc8ca2c` | Drains Wan Animate2 prefix-cache transfers before unpinning. Different owner/path; patch applies cleanly. |
| [#17123](https://github.com/Comfy-Org/ComfyUI/pull/17123), open, `4e718e03` | Resets cast buffers with dynamic VRAM disabled. Does not register partial batches; patch applies cleanly. |
| [#16919](https://github.com/Comfy-Org/ComfyUI/pull/16919), open, `20135a80` | Makes repeated model unload idempotent. Explicitly leaves the HostBuffer traceback in #16907 unaddressed. Different owner/path. |

[Issue #16907](https://github.com/Comfy-Org/ComfyUI/issues/16907) is an existing
report with tracebacks at the registration rollback site, but also a separate
finalizer error. Reference it as related; do not claim this patch fixes the
reporter's entire failure. [#15591](https://github.com/Comfy-Org/ComfyUI/issues/15591)
has an illegal CUDA access and a different eviction traceback, so it is not an
established reproduction of this defect. No specific existing issue for partial
prefetch cleanup was identified; the broad crashes in #16838 and second-generation
index failure in #16441 do not establish that cause.

## Validation and limits

Run from the corresponding candidate checkout with a prepared upstream-compatible
Python environment:

```bash
python -m pytest -q tests-unit/comfy_test/test_pinned_memory_recovery.py tests-unit/comfy_test/model_management_test.py tests-unit/comfy_test/system_memory_test.py tests-unit/comfy_quant/test_mixed_precision.py
python -m pytest -q tests-unit/comfy_test/test_prefetch_recovery.py tests-unit/comfy_test/model_management_test.py tests-unit/comfy_test/system_memory_test.py tests-unit/comfy_quant/test_mixed_precision.py
```

Each branch contains its own regression module, so select the corresponding
command. To reproduce a baseline, apply only that test file from the supplied
patch to `872eb6a3`, run it, then apply the production hunks and rerun. The
regression modules force CPU initialization and mock device operations.

The actual interpreter was `/home/jack/comfyui/.venv/bin/python`: Python 3.13.16,
PyTorch 2.14.1+cu130, comfy-aimdo 0.5.6, comfy-kitchen 0.2.37, pytest 9.1.1,
einops 0.8.2 and safetensors 0.8.0. Both candidates passed `py_compile` and
`git diff --check`. Ruff was unavailable and was not installed. The full upstream
unit suite was not run: a broader model-patcher import failed because the installed
fork kitchen lacks upstream's `comfy_kitchen.int8_decode`. This exploratory
collection failure was removed from the focused tests; the environment limitation
remains. Windows, ROCm, other accelerators, real CUDA streams, malloc/CUDA graph
capture, live models and performance were not validated.

The fork assessment checkout ran its committed `scripts/setup_worktree.py`.
Upstream lacks that helper: its comparison checkouts invoked the assessment
checkout's committed helper by absolute path, with `COMFYUI_PYTHON` explicitly set.
It checked prerequisites only, did not install/copy dependencies or start services,
and rerunning it preserved each tracked diff. No helper is bundled upstream.

Exact-checkout GitNexus preparation used owner `index_ensure`, embeddings disabled.
The baseline graphs were complete/current; registry-name collisions were resolved
by giving each checkout a unique alias. A transient context-query transport error
succeeded on retry. Graph callers were checked against source/text search:
`handle_pin` owns registration; the two cast callers are batched `pin_modules`
and direct `cast_bias_weight`. The cast function's wide reach was flagged critical
by impact analysis. The candidate changes only batched state publication and
checks the direct path explicitly; static graph coverage does not replace GPU
or platform validation.

After the changes, both candidate graphs were refreshed and complete with no
skill-installation warnings. Before each commit, `detect_changes` used that exact
repository and linked worktree: registration covered 2 changed files and prefetch
covered 3, without a partial-result warning. The assessment branch is a graph-free
documentation change; no serving-main code or canonical graph input was changed.

These are task-specific contribution artifacts using existing capabilities, so
no reusable service, MCP/API contract, model interface or shared skill was added.

## Submission decision

Registration recovery is a justified focused candidate. Use the related existing
issue cautiously and recheck #15944 before submission. Partial-prefetch cleanup
is independently justified, but needs a specific bug issue before submission under
the [upstream contribution guide](https://github.com/Comfy-Org/ComfyUI/wiki/How-to-Contribute-Code).
Both require a later explicit decision to publish, upstream CI/lint and any
requested platform validation. Do not add a `Fixes` link to an unverified symptom.

Proposed text: [registration PR](pin-registration-pr.md) and
[prefetch PR](partial-prefetch-pr.md), plus an unposted
[specific bug-issue draft](partial-prefetch-issue.md). Branch/patch receipts and test logs are in
the accompanying delivery bundle. No upstream candidate should be merged into
the serving fork as part of this assessment.
