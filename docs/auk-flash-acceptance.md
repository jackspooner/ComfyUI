# AuK-Flash development acceptance

Status: implemented and exercised locally; not deployed. This record does not
certify listening quality or all memory/backends. Source pins and commands are in
[the integration guide](auk-flash.md).

## Test environment and cases

Linux, AMD Ryzen 7 9700X, 44 GiB host RAM, NVIDIA RTX 5090 with 32 GiB VRAM,
PyTorch 2.14.1+cu130. Reference Transformers: 4.57.1. Native GPU workflow:
BF16 transformer/Qwen, FP32 VAE, Comfy Kitchen attention and comfy-aimdo dynamic
VRAM. Stage comparisons also used PyTorch SDPA. CPU transformer/VAE comparisons
used FP32 and four threads; conditioning used BF16.

Three five-second cases used a fixed initial-noise tensor per case:

- Synthesis: warm conversational voice saying “Hello there. This is a test of
  local audio generation.”
- Cloning: the same speech with a five-second reference voice.
- Editing: “Change the emotion to happy.” with the same reference.

Captures include instruction, reference preprocessing, token IDs, feature/masks,
all 37 Qwen hidden states, fused conditioning, reference latents, initial noise,
each velocity and Euler result, and decoded WAVs. Stage comparisons inject the
same saved conditioning, reference latents and initial noise. Separate end-to-end
runs use the native conditioning node and its seeded stochastic VAE encoding;
those are execution/quality checks, not identical-intermediate comparisons.

## Numerical comparisons

RMSE below is native versus pinned upstream. Differences are not bitwise parity.

| Stage | Synthesis | Cloning | Editing |
| --- | ---: | ---: | ---: |
| CPU BF16 fused conditioning | 0.0712 | 0.2510 | 0.2618 |
| GPU BF16 fused conditioning | 0.1002 | 0.6931 | 0.7261 |
| CPU FP32 final latent, shared conditioning | 0.00000689 | 0.00000253 | 0.00000522 |
| GPU FP32 final latent, shared conditioning | 0.0000260 | 0.00000704 | 0.0000347 |
| GPU BF16 final latent, shared conditioning | 0.0407 | 0.0550 | 0.0579 |
| CPU decoded waveform, shared-conditioning runs | 0.0000157 | 0.000000394 | 0.000000916 |

Tokenization and audio features match upstream exactly in the checked cases.
Fused-conditioning reference RMS is approximately 11.1. The native audio encoder
batches independent attention windows; upstream SDPA flattens them with a block
mask. Full audio-encoder FP32 RMSE is 0.00000168 on CPU and 0.00000122 on GPU;
BF16 RMSE is 0.0197 and 0.0621 respectively. Thus FP32 validates the underlying
math, while BF16 execution accumulates larger differences. The captures locate
those differences in the audio encoder and subsequent Qwen layers; they do not
prove every difference is caused by attention alone. Upstream BF16 uses autocast;
native inference uses ComfyUI's BF16 operations and compound kernels.

The full VAE CPU test matches encoder statistics, normalized stochastic latents
and decoded waveform exactly when the random draw is identical. The four-step
standard loader/BasicGuider/Euler adapter also matches the FP32 upstream trajectory
closely when supplied the same conditioning and noise.

## Execution and resources

All three complete native GPU cases produced finite mono 24 kHz audio with exactly
120,000 samples. The native node path switched between Qwen, transformer and VAE
and explicitly unloaded models between stages. GPU sampler repeat and unload/reload
results were identical; cancellation interrupted the next evaluation. After each
complete case, PyTorch allocated bytes returned to the same 33,555,968 bytes.
This short sequence does not establish long-running leak freedom.

| Measurement | Observed value |
| --- | ---: |
| Full native GPU sequence, three cases | 29.4 s |
| Full native GPU process peak VRAM, NVIDIA telemetry sampled at 100 ms | 9.08 GB |
| Full native GPU process peak host RSS | 4.09 GB |
| CPU native sampler sequence, including repeat/cancel | 64.3 s; 9.82 GB peak RSS |
| CPU native Qwen capture, including saved hidden states | 9.9 s; 13.72 GB peak RSS |

These are decimal GB and process measurements for these inputs, not minimum
hardware requirements or guaranteed peaks. Dynamic allocator memory is outside
PyTorch's normal counters; the 9.08 GB figure uses NVIDIA process telemetry.
Longer references, output durations and batches need separate measurements.

Thirty-four AuK tests passed, including the full VAE comparison; 47 existing detection
and MOSS regression tests passed. All three example API graphs validate, and their
editor layouts pass containment/spacing checks. Original downloads and converted
artifacts remain intact; every retained tensor was verified during conversion.

## Quality and remaining checks

Local MOSS transcribed CPU and GPU upstream/native synthesis and cloning
consistently. The GPU edit transcription differed on a proper noun; content and
speaker preservation still require listening review.
One real MOSS helper request returned valid JSON, preserved the exact speech and
honored an explicit five-second duration. Other helper tasks are not comprehensively
validated. Model-based transcription is not a substitute for listening.

The happy-emotion edit was described as neutral in both CPU implementations. That
case establishes execution and comparison coverage, not successful emotion control.
Listening, speaker preservation, edit effectiveness, longer runs, varied durations,
forced low-memory pressure and other accelerator backends remain unaccepted.
No service restart or deployment has occurred.
