# Stable-release maintenance

Deploy published non-draft, non-prerelease tags from the actual configured GitHub
remote, retaining the fork's local memory-recovery patch. Do not deploy a later
branch tip or downgrade a checkout containing the stable tag. The 27 October 2026
release is [v0.43.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.43.0),
release commit `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`.

Before changing source or packages, save Git revision/diffs, custom-node revisions,
package inventory/constraints, service configuration, workflows and baseline
health privately. Retain a complete recoverable copy of the existing virtual
environment and custom nodes. Preserve the `models` symlink, external weights,
disabled nodes, private configuration and all uncommitted work. Never stash/reset
the deleted tracked model placeholders as if they were ordinary source edits.

Prepare a task worktree from committed `main` with the repository's setup command.
Merge only the stable tag and review the combined memory-loading changes. Run
pinned-memory, prefetch, H3 VAE, relevant node/API and installed Sol-H3 integration
checks before activation. The frontend, templates and docs packages must satisfy the tag's
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

The v0.41.0 release keeps frontend 1.55.17, embedded docs 0.5.14, Kitchen
0.2.37 and allocator 0.5.6; it requires workflow templates 0.11.84. Preserve
the operator-selected floating frontend flag and record the actually served
version separately. Upstream adds Flux3 image models, H3 LightVAE variants,
reference/geometry nodes and `GET /default_workflow`; discover node schemas
and local weights before using them. Existing-model acceptance does not certify
these new variants or full-size/audio quality. The authoritative general
`comfyui` skill links the v0.41 operating notes.


The v0.42.1 merge requires frontend 1.55.18, workflow templates 0.11.89 and
Kitchen 0.2.39; embedded docs 0.5.14 and allocator 0.5.6 remain unchanged.
Qwen deferred DeltaNet/INT8 decoding depends on the matching Kitchen operations.
Retain the fork's NVFP4/INT8 operation imports and all existing CLIP selectors;
`moss_image` is a distinct selector from `moss_audio`. Merge the upstream
unfinished-thinking template fix while preserving history, context and raw-text
outputs. Decode the tokenized template tail only when needed, accepting an empty
token sequence instead of failing after a successful generation. Run the Qwen decode, conversation, native encoder/quantization and
prefetch regressions against candidate dependencies before activation.

Upstream MOSS-Image uses its native diffusion model, Qwen3.5-9B conditioning
(`CLIPLoader` type `moss_image`) and Flux2-format VAE/latents. Its experimental
layout node emits centered pixel boxes from Create Bounding Boxes; it performs
no inference. `RemeshMesh` adds a slower solid graph-cut mode with fill/QEF
controls. Discover deployed schemas and weights before use. Existing H3 smoke
acceptance does not verify these new weights or large/complex remeshing.
The shared general ComfyUI skill owns detailed operating guidance; this release
adds no MCP service or router profile. Record baseline warnings and failed past
requests separately from new startup/import/inference regressions.

The v0.43.0 merge requires frontend 1.57.4, workflow templates 0.11.97,
embedded docs 0.5.15, Kitchen 0.2.40 and allocator 0.6.0. Writable runtime
weights/KV storage depend on the matching allocator and pin-state contract;
preserve the pinned-registration retry and partial-prefetch recovery tests.
Qwen adaptive tree MTP adds `tree2s1`, `tree2s2`, `tree3s1` and `tree4s1m8` to
Generate Text. Forward those modes unchanged; constrained JSON still requires
`mtp=off`. Preserve automatic output-length resolution before allocating caches.

Fibo uses upstream NTK-aware RoPE and negative-prompt chat wrapping, while the
fork retains its CLIP enum identities, native history and optional schema inputs.
The Flux2 VAE fused path and MOSS reference/control additions require their own
checkpoint acceptance; an H3 smoke test does not certify them. The optional local
agent host remains disabled without its explicit flag and separately provisioned
package. No agent package is required for ordinary generation.

The fork excludes upstream cloud deployment routes, installation-snapshot upload
code, builder URL configuration and the persistent headless-use marker. CLI
feature overrides cannot advertise that excluded path. Existing partner nodes
remain their own explicit external integrations; no cloud deployment is activated
by release maintenance. Upstream attribution and release ancestry are preserved.
