# Native MOSS-Audio 8B

This fork integrates [OpenMOSS MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio)
(Apache-2.0) with ComfyUI's native Qwen3 generation, model patcher, offloading,
attention selection and prefetch. Production inference uses no Transformers MOSS
model, external worker, forward hooks or network requests.

## Prepare locally

Use the original BF16 [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking)
checkpoint at revision `aacbb8036eb06dd968c105f91d2663af8fbee720`.
Retain all four safetensors shards, the shard index, config and tokenizer files.
Git LFS pointer files are not weights. Source implementation comparisons use
OpenMOSS/MOSS-Audio revision `66326e6e0db34f036c86a76ba005efa4830c69dd`.

Run with the prepared ComfyUI Python environment:

```sh
python scripts/convert_moss_audio.py \
  --source /absolute/path/MOSS-Audio-8B-Thinking \
  --destination /absolute/model/root/text_encoders/moss_audio_8b_thinking_bf16.safetensors
```

The converter is offline. It validates the architecture, all 901 source tensor
shapes/dtypes, shard completeness and the tokenizer's timestamp IDs. It streams
18,104,926,976 weight bytes without holding a second full weight copy in RAM,
embeds the original tokenizer, and independently hashes every mapped output
payload against its source. The companion `.manifest.json` records mapping,
provenance and tensor hashes. Existing destinations are refused. Temporary output
is removed on handled interruption; after an uncatchable process kill, remove only
that conversion's hidden temporary files before retrying. Publication uses atomic
no-overwrite hard links on the destination filesystem. A manifest without its
checkpoint after a crash is incomplete; preserve or remove it before retrying.
Original shards are never modified. Allow disk space for originals and the new
18.1 GB checkpoint, plus filesystem cache. Quantized formats are not validated.

## Workflow

Select `moss_audio` in **Load CLIP**, then connect it to **Generate Text**.
Connect **Load Audio** to Generate Text's `audio` socket; use **Preview as Text**
for its generated-text and thinking outputs. Text-only generation also works.

The example uses 2,048 output tokens, thinking enabled, the standard template,
temperature 1, top-k 50, top-p 1, min-p 0, repetition penalty 1 and seed 42.
The example asks for a 3–5 sentence caption covering foreground/background sounds
and event order, with uncertainty and no exact timestamps. Precise timestamps in
sampled reasoning can be invented, even beyond the supplied clip duration.
The system-prompt input replaces the standard system turn. Disabling thinking
primes an empty Qwen thinking block; this is a prompt control, not a change to
weights. With the default template disabled, include exactly one
`<|audio_bos|><|AUDIO|><|audio_eos|>` span when supplying audio. Raw text is otherwise
left unchanged. MTP has no effect for this architecture.

One nonempty clip is supported, with arbitrary channel count downmixed to mono
and resampled to 16 kHz. Preprocessing uses 128 mel bins, FFT 400, hop 160, with
no 30-second truncation. The encoder processes 400-frame chunks and inserts
timestamp digits every 25 audio tokens (two seconds), excluding the digits from
DeepStack additions. Audio encoding occurs once per generation. Generation caches
and audio features last only for that execution; model residency belongs to
ComfyUI. Input plus requested output must fit the 40,960-token context.

## Token budgets and caption prompting

Generate Text's `max_length` accepts `0` through `1048576`; its default stays 512.
Positive values retain their existing meaning: maximum **new** tokens, shared by
reasoning and the final answer. `0` resolves to the model's configured context
minus the expanded input length, including audio features and timestamp tokens.
For MOSS this is `40960 - input_tokens`. It stops early at EOS. An input that fills
the context must be shortened; increasing the UI maximum does not enlarge the
checkpoint's context. The same automatic convention applies to native Qwen,
Gemma and Llama generation, including Qwen3.5's MTP path.

Automatic mode reserves cache capacity for the remaining context. It is not a
free-VRAM estimator or an OOM retry mechanism; memory allocation/offloading stays
with ComfyUI. A large context can require substantial memory even if EOS arrives
early. Use an explicit smaller budget if memory is insufficient. For example:
`TextGenerate` inputs `max_length: 0`, `thinking: true`,
`sampling_mode: "off"`, `use_default_template: true` retain deterministic greedy
generation with an automatically sized context budget.

For captions, use a specific request such as: “Describe the audible foreground
and background sounds in 3–5 sentences. Summarize speech at a high level,
distinguish it from music and sound effects, and describe the order of events.
State uncertainty. Do not provide exact timestamps, a transcript, or a one-word
answer.” The generic “Describe this audio” request can produce a transcript,
one-word answer or numeric/hash-like output after otherwise fluent reasoning.
Increasing the output budget does not fix a model that already ended its answer.
Disabling thinking changes the prompt but does not establish caption accuracy.

## Verification status

CPU tests pass for 16/44.1/48 kHz mono/stereo, silence, short clips, partial chunks,
timestamp boundaries and preprocessing through 120 seconds. Isolated upstream
FP32 comparisons pass at `atol=1e-5, rtol=1e-4`, including intermediate audio
features, token expansion, prefill logits and cached decoding. The reference
requires Transformers 4.57.1; keep it in a disposable validation environment,
separate from serving ComfyUI's Transformers 5.20.0.

```sh
python -m pytest tests-unit/comfy_test/moss_audio_test.py -q
MOSS_REFERENCE_SOURCE=/absolute/pinned/upstream \
MOSS_REFERENCE_MODEL=/absolute/original/checkpoint \
  /absolute/reference/python -m pytest tests-unit/comfy_test/moss_audio_reference_test.py -q
```

The reference source directory must contain `src/modeling_moss_audio.py` and its
pinned sibling modules. Reference tests are skipped unless explicitly configured.
Full BF16 inference has been compared on an RTX 5090 against the pinned upstream
implementation. The captured prefill and first three cached-decode logits have
matching argmax tokens and cosine similarity above 0.99997; generated wording can
differ. BF16 intermediate-feature cosine similarities exceed 0.99993. Tiny FP32
fixtures establish the tighter numerical tolerance above; BF16 is not bit-exact
across ComfyUI and Transformers attention/kernel paths.

Identical greedy prompts were tested on male narration, solo trumpet, birds,
a 30-second mixed scene and a two-minute scene. Both implementations recognize
the main categories. Both incorrectly call the trumpet a harmonica and omit quiet
birds beneath speech in the short mixed scene. Both detect music beginning at
1:54 in the two-minute scene, alongside earlier speech and birds. This validates
tail coverage, not reliable instrument identification or exhaustive captioning.
The test audio derives from the librosa examples: LibriSpeech/Garth Comira
(CC BY 4.0), trumpet/Mihai Sorohan (CC BY 3.0), robin/InspectorJ (CC BY 3.0).
Original attribution files accompany local validation fixtures.

Cancellation, repeated generation, offload/reload, text-only generation, context
capacity errors and transition to Qwen3-VL-4B and back passed on the RTX 5090.
Greedy outputs were unchanged after reload and the model transition. Instrumented
runs confirmed one audio-encoder call per generation.

The installed service exposes `moss_audio` and completed the example through its
normal queue on 16 October 2026. The two-minute example returned speech, birds
and music; the sampled reasoning also demonstrated unreliable exact timestamps.
The editable workflow is `MOSS-Audio 8B Thinking.json` in the user workflow library,
with `moss_audio_demo.wav` in the instance input directory. Portable examples are
[editor JSON](workflows/moss_audio.json) and [API JSON](workflows/moss_audio_api.json);
select your own audio when using these elsewhere. Other accelerator backends
remain unverified.

## Measured performance

RTX 5090 32 GB, Linux, PyTorch 2.14.1+cu130, BF16, the same 14.84-second narration,
greedy decoding capped at 128 tokens. Three fresh processes per backend, alternating
reference/native; three additional warm executions in each first process. No
capture copies were enabled. The OS filesystem cache was **not cleared**: these
runs followed prior validation reads and are not cold-disk benchmarks. CPU kernel
initialization, page faults, allocator behavior and GPU transfer affect first use.

| Metric | Native ComfyUI | Upstream Transformers 4.57.1 |
|---|---:|---:|
| Checkpoint setup, three runs (s) | 0.836 / 0.796 / 0.548 | 0.680 / 0.593 / 0.635 |
| Placement call, three runs (s) | 0.135 / 0.163 / 0.117 | 37.069 / 18.528 / 18.903 |
| First execution to first token (s) | 10.785 / 12.575 / 10.387 | 6.082 / 3.702 / 3.857 |
| **Load start to first token (s)** | **11.756 / 13.534 / 11.052** | **43.831 / 22.823 / 23.395** |
| Warm execution to first token (s) | 0.090 / 0.078 / 0.087 | 0.051 / 0.048 / 0.046 |
| Warm decoding (tokens/s) | 71.8 / 72.8 / 72.5 | 70.0 / 71.6 / 66.1 |
| Peak process RAM, fresh runs (GB) | 2.72–2.74 | 6.19–6.25 |
| Peak process GPU memory (GB) | 19.11 | 19.37–19.39 |

Native dynamic-VRAM placement is lazy: its placement-call timing does **not** mean
all weights have reached the GPU. The combined load-to-first-token metric includes
that first-use work; its median improved from 23.4 to 11.8 seconds (about 2×) on
these runs. This is a BF16 comparison, not a measured comparison with GGUF.
Checkpoint parsing alone was not consistently faster. First-execution native
decoding was 50.7–61.5 tokens/s versus 64.4–67.5 upstream; warm decoding recovered
to comparable throughput. Warm native audio preprocessing took 2.5–3.2 ms and
encoding 12.9–15.5 ms, versus 8.8–11.6 ms and 7.0–8.2 ms upstream.

RAM is process peak RSS and excludes the OS file cache. GPU memory is driver
process usage sampled every 0.5 seconds, so brief peaks may be missed; PyTorch
allocator statistics alone omit ComfyUI's aimdo-managed weights. These are
measurements with available GPU capacity, not guarantees under competing work.

Reproduce with `tests-inference/benchmark_moss_audio.py --help`, coordinating GPU
ownership first. Use the serving Python for native and the isolated reference
Python for upstream, explicit local model/audio paths and a truthful
`--cache-condition`. Run three new processes per backend and use `--iterations 4`
for one fresh plus three warm executions. `--capture` is diagnostic and excluded
from timings. `--lifecycle --transition-model /absolute/other/native/checkpoint`
checks cancellation, context limits, offload/reload, text-only generation and
return from another native model. Reports include per-execution timings, token
IDs, captions and memory measurements; no data leaves the machine.
