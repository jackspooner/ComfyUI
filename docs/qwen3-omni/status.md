# Execution status: storage gate

[Project index](README.md) · updated 18 October 2026

The selected checkpoint is downloaded. The offline converter is implemented and
its synthetic integrity/publication checks pass. Native loading, full-artifact
validation, accelerated execution, multimodal acceptance and deployment are
**not complete**. The storage diagnostic does not install a model or declare M1
passed.

## Resumed implementation

The task worktree merged main at `db18b279`, preserving both earlier project
commits. Setup and the exact graph ensure passed with embeddings disabled. The
previous unrelated README modification is no longer present; other canonical
local state remains unrelated and must be preserved.

The [complete source reconciliation](results/source-reconciliation.json) checks
all 15 input hashes and matches **all 65,880 expected tensor shapes and dtypes**,
including excluded speech modules, after expanding upstream fused expert banks.
The 1,408-byte stale aggregate is now a documented exception confined to this
exact hash-verified source. Original files and their declared index total remain
unchanged. This resolves source accounting, rather than waiving integrity checks.

ModelOpt 0.42.0 requires Python >=3.10,<3.13. Its optional HF extra additionally
requires Transformers <5; that integration extra is not used. A separate Python
3.12.15 environment now imports the actual ModelOpt decoder alongside Transformers
5.20.0 and PyTorch 2.14.1+cu130. New wheels were obtained through Download Manager 2,
hash verified and installed offline with a complete dependency lock. Compatible
native CUDA libraries are shared read-only from the serving deployment. `pip check`
passes. The serving Python 3.13 environment was not modified. ModelOpt warns that
Transformers 5.20.0 is outside its tested HF integration range; successful decoder
use does not establish support for its quantization/export integrations.

The reconstructed full-model reference preserves upstream routing, attention,
positions and generation, replacing packed-weight access only. It reached the
correct `OMEGA` answer in the CPU smoke case using BF16 compute/cache. The original
run continued past its end token because standalone Thinker generation inherited
no EOS setting; that run is not accepted as EOS parity. Reference generation now
binds EOS and padding to the embedded tokenizer. The incomplete first acceptance
run was stopped and retained with its reason. The corrected suite returned `12`
for `text_arithmetic` and stopped at EOS after 69.39 seconds on CPU; the remaining
cases are running.
No CPU result establishes GPU performance or multimodal quality.

Actual ModelOpt decoding now numerically matches native FP32 decoding on 33
projections, including a 4,304-column vision down projection. The CPU native BF16
decoder differs by up to 0.001953125 because of intermediate rounding; GPU and
full-model differences require separate evaluation. Twenty-four functional
fixtures (six per modality group) are frozen outside Git, with checksums and audio
attribution, including independent modality replacement/removal. Upstream
tokenization/preprocessing executes for text, audio, image and combined fixtures.

The converter uses the verified transformations, fuses expert banks only when
global scales match, embeds inference assets, inventories excluded speech weights
and validates output checksums before publication. See [the command and recovery
procedure](quantization.md#offline-converter). Twenty-four synthetic storage and
conversion tests pass, including zero-padding, ordering, unequal scales, unsafe
source paths, dangling output links and interrupted publication. An artifact is
not installed or accepted merely because the converter tests pass.

## Reproduce the diagnostic

Use the prepared ComfyUI Python after the repository worktree setup command.
From the checkout root:

```bash
/path/to/prepared/python -m scripts.check_qwen3_omni_quant \
  --source /absolute/path/to/pinned-checkpoint \
  --source-revision 0ed3aa2b00eeedbb0d38b6d4738f506af759213b \
  --download-manifest /absolute/path/to/download-verification.json \
  --report /absolute/path/to/new-report.json

/path/to/prepared/python -m pytest tests-unit/comfy_test/qwen3_omni_quant_test.py -q
```

The manifest is local download evidence: `revision` and `files`, each with `name`,
`bytes`, `sha256` and `expected_sha256`. Obtain expected shard hashes from the
pinned Hugging Face API LFS metadata, not from the downloaded bytes themselves.
The diagnostic rehashes every shard, checks safe paths, duplicate keys, complete
inventories and scale layouts, then compares sampled projections and expert paths
on CPU. It streams file hashes and decodes only small components. It does not
download anything, use CUDA, modify source files, repair the index, or emit model
weights. The report refuses an existing path. A stale source byte total is retained
in the report and causes a nonzero exit, even if component comparisons pass.
Other integrity failures abort without a success report. Retry with a new report
path after investigating the failure; preserve original model files and receipts.

## Original diagnostic evidence (17 October)

The following observations describe the earlier equation-based diagnostic. The
resumed results above supersede its pending source-accounting and decoder gates.

### Pinned input and actual layout

- Repository: `shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4`.
- Revision: `0ed3aa2b00eeedbb0d38b6d4738f506af759213b`.
- Download Manager 2 batch: `r3XtLjO-rlf0QxuTleF75`, all seven shards completed.
  All seven sizes and SHA-256 values match the pinned publisher LFS metadata.
  Tokenizer and processor assets are also local. Firefox's scoped cookie read was
  locked; these public downloads completed unauthenticated.
- 65,880 tensors. Actual payload: 25,850,816,336 bytes. The published index declares
  25,850,817,744 bytes, a **1,408-byte discrepancy**. Every indexed key is present
  in its assigned shard, and the safetensors reader accepts each shard.
- Thinker including audio/vision: 18,769,590,222 payload bytes. Talker:
  6,649,192,960; code2wav: 432,033,154. These are source payload sizes, not final
  package sizes or VRAM estimates.
- 18,432 expert projections, 192 language attention projections, 195 audio
  projections and 116 vision projections are packed NVFP4: 18,935 total.
  The encoders are **not entirely BF16**, contrary to the source model card.
- Experts are separate `experts.<index>.gate_proj`, `up_proj`, `down_proj` weights.
  Each has a row-major FP8 E4M3 block scale and a scalar BF16 global scale.
  Language attention scales are stored separately from its early-layer weights.
  No `input_scale`, `pre_quant_scale` or KV calibration tensors were found.

The proposed exact map swaps the two nibbles in each byte and uses Comfy Kitchen's
existing `to_blocked` scale permutation. It never quantizes the weights again.
The independent CPU decoder follows the
[ModelOpt 0.42.0 NVFP4 equation](https://github.com/NVIDIA/Model-Optimizer/blob/0.42.0/modelopt/torch/quantization/qtensor/nvfp4_tensor.py).
This is format-equation evidence, **not execution of the pinned ModelOpt package**
and not a full-model reference comparison.

The [recorded diagnostic](results/storage-gate.json) covers 32 projections: one
language attention projection, four audio/vision projections and gate/up/down
from nine experts (layers 0/23/47, experts 0/63/127). Packed bytes and scale bytes
round-trip exactly; FP32 decodes match exactly. Nine complete CPU decoded-weight
expert paths also match, using fixed random, zero and outlier inputs. All 6,144
expert gate/up pairs have equal global scales, allowing later exact concatenation
without changing those scales. These checks do not measure activation-quantization
error. Peak diagnostic host RSS was 1,379,835,904 bytes (about 1.29 GiB), not a
conversion or inference memory measurement. Fourteen synthetic unit tests passed.

The diagnostic exits nonzero for the published index discrepancy. No conversion
artifact was produced. Non-LFS configuration/template assets were also checked
against their publisher Git blob hashes; the tokenizer's LFS SHA-256 matched.

### Original reference and execution gates

The serving environment remains Python 3.13.16, PyTorch 2.14.1+cu130,
Transformers 5.20.0 and Comfy Kitchen 0.2.37. No dependencies were changed.

`AutoConfig.from_pretrained(..., local_files_only=True)` finds no
`quantization_config` in the source `config.json`. Explicitly constructing
Transformers' `NVFP4HfQuantizer(..., pre_quantized=True)` and validating its
environment raises `Loading pre-quantized NVFP4 checkpoints is not supported yet.`
Its ordinary loading path cannot be treated as a working quantized reference.
The checkpoint separately identifies ModelOpt 0.42.0 in `hf_quant_config.json`;
it does not supply a working reference loader.

A metadata-only Thinker construction on PyTorch's meta device gives 1,407
reference tensors. All common parameter shapes match after accounting for packed
columns. The only inventory differences are 96 fused reference expert banks
versus 18,432 individual source projections. This establishes structural alignment,
not routing, preprocessing, cached decoding or multimodal correctness.

Next required work before accepting a production path:

1. Account for the stale index total against a complete expected tensor inventory;
   preserve original hashes and make any metadata reconciliation explicit.
2. Establish a pinned executable reference loader, including split expert packing
   and quantized audio/vision modules. Do not claim the source W4A8/FP8 cache path
   works merely because its configuration names those formats.
3. Compare the actual pinned source decoder, benchmark eligible native kernels in
   an available GPU window, and evaluate the fixed multimodal fixtures. CPU
   decoded-weight comparisons cannot select an accelerated production path.

The GPU was serving queued ComfyUI work during investigation; no competing GPU
benchmark was launched. No latency, throughput, GPU-memory or multimodal-quality
acceptance is reported. Documentation commit `2813252d` is preserved on the task
branch. Recheck current canonical state and serving ownership before integration
and activation; the earlier README blocker is no longer present.
