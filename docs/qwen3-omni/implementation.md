# Implementation and delivery plan

[Project index](README.md)

M0 preparation is complete in the resumed worktree. M1 now has a verified complete
source inventory, actual ModelOpt decoding, a reconstructed full-model GPU fixture
run and a viable accelerated CUDA candidate. Full native precision acceptance
remains pending.
M2's converter code and isolated tests are implemented; full-artifact validation
and installation remain pending. M3/M4 adapter code and FP32 component tests are
implemented; full-model and installation acceptance remain pending. See
[execution evidence](status.md). M5/M6 remain pending.

| Milestone | Work | Exit evidence |
| --- | --- | --- |
| M0: preparation | Pin checkpoint and reference implementation; establish task worktree/runtime; recover exact-checkout graph readiness | Reproducible environment, source inventory and current graph or explicit blocker |
| M1: quantization gate | Run the small projection/expert experiment in the quantization document | Written supported-path decision with numerical, timing and memory results |
| M2: converter | Implement checked, bounded-memory conversion and asset packaging | Inventory/checksum manifest, negative tests and exact representation comparisons |
| M3: Thinker integration | Port native MoE language model, tokenization, generation and loader detection | Text reference comparisons and existing-loader regression checks |
| M4: audio and image | Integrate both encoders, preprocessing and combined-input positions/masks | R2–R4 passed independently and jointly; media influence demonstrated |
| M5: production behavior | Validate offload, cancellation, cache lifetime and representative performance | Resource measurements and R5/R7 acceptance on named hardware |
| M6: delivery | Save workflow, publish usage/recovery docs, update owner skill, review and integrate | R1–R7 evidence, commit/push status and separately recorded activation status |

M2 depends on M1's storage decision. M3 can use small fixtures during development,
but full-model acceptance depends on validated conversion. M4 and M5 depend on
the complete Thinker. Do not label M3 as a completed release.

## Development procedure

Use the repository's applicable AGENTS.md and supported lifecycle. At the inspected
baseline uses the repository's native setup with the
existing setup command, from committed main without importing uncommitted work:

```bash
COMFYUI_PYTHON=/absolute/path/to/prepared/python python3 scripts/setup_worktree.py
```

The runtime needs Python >=3.10, PyTorch, einops and safetensors. Setup checks the
explicit runtime without installing packages, loading models or starting services.

Do not downgrade serving dependencies for a reference model. If reference
execution needs incompatible packages, use an isolated, explicitly pinned test
environment. Keep production inference native. Select dependencies and kernels
from installed capabilities; do not assume a Transformers version from examples.

Submit authorized ComfyUI inference through its normal queue even when busy.
Preserve prompt IDs after ambiguous responses and inspect before retrying. Avoid
concurrent uncoordinated reference workloads on the same GPU. Check idle state
before any separately authorized service restart or deployment.

## Review and release

Review the cumulative diff and numerical evidence. Run targeted unit/reference
tests, the modality matrix, other-loader regressions and owner graph change checks.
Record precise test commands once implemented; no fictitious commands or scripts
are supplied here. Update root README and operating guidance in the same
implementation change. The existing shared `comfyui-model-runtimes` skill owns
native integration guidance; follow `shared-agent-skills` to edit and refresh its
authoritative source. Do not create a per-model skill or register a task worktree.

Commit only task-owned files, push the task branch to the authenticated personal
fork, recheck main/concurrent work and integrate safely. Preserve pushed commits
when unrelated dirty files or serving safety block integration. Source merge
does not authorize restart. Cleanup only a clean unused task worktree after
verified integration and index retirement; otherwise report its retained path.
