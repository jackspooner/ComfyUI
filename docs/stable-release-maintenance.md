# Stable-release maintenance

Deploy published non-draft, non-prerelease tags from the actual configured GitHub
remote, retaining the fork's local memory-recovery patch. Do not deploy a later
branch tip or downgrade a checkout containing the stable tag. The 14 October 2026
candidate is [v0.40.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.40.0),
release commit `3541246a9c997f4b8c32f9f88f3c985d02021c79`.

Before changing source or packages, save Git revision/diffs, custom-node revisions,
package inventory/constraints, service configuration, workflows and baseline
health privately. Retain a complete recoverable copy of the existing virtual
environment and custom nodes. Preserve the `models` symlink, external weights,
disabled nodes, private configuration and all uncommitted work. Never stash/reset
the deleted tracked model placeholders as if they were ordinary source edits.

Prepare a task worktree from committed `main` with the repository's setup command,
then prepare its exact GitNexus graph with embeddings disabled. Merge only the
stable tag and review the combined memory-loading changes. Run pinned-memory,
prefetch, H3 VAE, relevant node/API and installed Sol-H3 integration checks before
activation. The frontend, templates and docs packages must satisfy the tag's
requirements. Preserve the compatible CUDA PyTorch/Triton/Comfy Kitchen, CUTLASS,
GGUF and separately managed speech runtimes through explicit constraints.

ComfyUI v0.40.0 requires allocator 0.5.6. Its upstream null-page release guard
addresses the repeated escaped-alias crash previously repaired by local 0.5.5.post1.
Verify the repeated-alias and failure-injection regressions against the actual
candidate binary before replacing that deployment pin; the historical 0.5.5.post1
recovery manifest alone does not authorize overriding a newer verified allocator.
Retain the prior runtime for rollback. This statement does not certify all
allocator behavior, platforms or model combinations.

Inspect active custom-node requirements and source compatibility. Keep intentional
pins and local changes; update a pack when compatibility requires it. Missing
remote access is a documented hold, not evidence that a pack is current. Existing
FishAudio installs descript-audiotools without dependencies to avoid its obsolete
protobuf upper bound; do not force that bound into the shared runtime. The existing
decord wheel also reports unsupported platform metadata. Compare `pip check`
before/after and distinguish these baseline issues from new conflicts.

Immediately before stopping the service, confirm `/queue` has no running/pending
prompts, Workflows5 has no active dependent runs and RuntimeHost has no inference
in flight. Preserve invocation-bound GPU cleanup and current operator launch flags.
Replace packages only while stopped; start through the managed service. Confirm
CUDA, frontend assets/API health and custom-node registration, then execute a small
existing workflow with local checkpoints and validate its saved output. API
liveness and imports alone never mark an upgrade successful. Record prompt IDs,
output dimensions/frame count, package versions and verification limitations.

If verification fails, repair the focused incompatibility and repeat acceptance.
If repair cannot succeed, stop safely, restore source/configuration and the complete
previous environment, restart and verify the old installation. Keep a durable
incomplete-update record for the next scheduled retry. Never report rollback as a
successful upgrade. Backups and logs may contain secrets and remain private.
