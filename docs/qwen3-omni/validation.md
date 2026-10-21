# Validation and acceptance

[Project index](README.md)

Status: complete source accounting, actual ModelOpt component decoding and 24
synthetic integrity tests pass. The reconstructed reference has passed its first
text/EOS fixtures. The reconstructed GPU reference completed all 24 cases (20
assertion passes, four source failures). Native FP32 component parity passes;
full native precision, lifecycle and live acceptance remain pending. See
[recorded evidence](status.md) and
[offline validation commands](reference.md). Use owned, licensed or synthetic
fixtures and record their checksums. Keep test fixtures and
configuration isolated from production. GPU acceptance is separate from CPU
fixture success and node registration.

## Test matrix

| Area | Cases | Required evidence |
| --- | --- | --- |
| Artifact integrity | Complete conversion; missing/truncated shard; missing scale; unsafe shard path; existing output | Complete inventory or clear refusal, untouched originals, no published partial artifact |
| Quantized operations | Attention projection; multiple experts; fused gate/up ordering; representative prefill/decode shapes | Decoded-weight comparison and measured kernel outputs under declared precision |
| Tokenization | Unicode, special tokens, default/raw templates, system override, empty prompt | Reference token IDs and preserved media placement |
| Text | Short deterministic prompts; long prompt; EOS; explicit/automatic output budget | Logit/token comparison, correct stop behavior, no prompt echo |
| Audio | Speech, silence, non-speech events, stereo/mono, differing sample rates and durations | Reference features/lengths, correct input adaptation and text output |
| Image | Objects, readable text, non-square dimensions and existing multi-image input behavior | Reference preprocessing/features and correct token/position expansion |
| Combined | Text + image + audio; replace each modality while holding others fixed | Both media paths reach the model and materially influence appropriate answers |
| Boundaries | Input fills/exceeds context after media expansion; insufficient memory; malformed artifact | No silent truncation or ignored input; actionable failures and recoverable queue |
| Lifecycle | Repeated runs, cancellation in preprocessing/prefill/decode, offload/reload, transition to another model | No stale features or retained per-run KV cache, cancellation recovery, measured residency |
| Regression | Existing Qwen3-VL and MOSS-Audio loader/generation paths; model-detection near misses | Existing workflows remain functional and do not select Omni incorrectly |

Use full-model greedy comparisons where a working source runtime exists. Compare
prefill logits and cached decode as well as final strings: one changed token can
alter an entire answer. Exact greedy agreement is strongest under matching
precision; alternate activation/cache precision needs independently recorded
numerical and task-quality acceptance. Equal seeds do not guarantee equal sampled
outputs across backends. Human-readable fluent captions alone do not prove that
both input modalities were consumed or that observations are accurate.

A reference environment that cannot actually load this checkpoint is a blocker
to reference parity claims. A decoded-weight reference supports narrower claims;
document that limitation rather than reporting upstream parity.

## Performance record

Record GPU model/VRAM, driver, Torch/Comfy Kitchen versions, model/converter commits,
weight/activation/cache dtypes, backend path, offload settings, input dimensions,
audio duration, expanded token counts and output budget. Measure cold and warm
time-to-first-token, generation tokens/second, peak GPU allocation and driver use,
host RAM, conversion peak RAM, file sizes and post-run residency. Explain cache
and warmup conditions. Test with and without offload where supported.

No latency or VRAM ceiling has been agreed. Publish measurements and practical
supported limits instead of inventing a target. Unsupported hardware remains
unverified; do not extrapolate one GPU result to other backends.

## Acceptance record template

For each run, save: requirement/test ID; source revision; implementation commit;
artifact and fixture hashes; environment; exact command or ComfyUI prompt ID;
expected/observed behavior; numerical tolerances and rationale; result/log paths;
resource measurements; pass/fail; known limitations. Record failures as failures
and preserve evidence before retrying. A release needs R1–R7 coverage, reviewed
precision deviations, a saved example workflow and explicit activation status.
