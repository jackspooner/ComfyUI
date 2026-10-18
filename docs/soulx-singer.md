# Native SoulX-Singer

This fork implements local SoulX-Singer SVS (new lyrics with singer reference,
continuous F0 or note/score conditioning), transcription-free SVC, RMVPE pitch
extraction, karaoke vocal separation and dereverberation. Models use native
ComfyUI operations, selected attention, CoreModelPatcher, prefetch and the memory
compiler. Audio is the standard ComfyUI `AUDIO` dictionary. No upstream application,
singing subprocess, Transformers model or network request runs during inference.

The upstream [project and README](https://github.com/Soul-AILab/SoulX-Singer)
provide zero-shot SVS and SVC, an automatic transcription pipeline, examples and
an external MIDI editor. Its automatic alignment is explicitly imperfect. This
integration ports the released inference, not a promise of exact singer identity,
perfect lyrics or polyphonic pitch tracking. SoulX code/model cards use Apache-2.0;
the adapted Amphion/Vocos/Roformer portions retain MIT notices in
[`NOTICE`](../comfy/ldm/soulx_singer/NOTICE). CMU/G2P assets retain their own notices.
The complete ComfyUI distribution keeps its existing license.

## Pinned sources and files

| Source | Revision |
| --- | --- |
| Soul-AILab/SoulX-Singer code | `81aeb3ae772c70093c3de74dc23c92d983801ae4` |
| Soul-AILab/SoulX-Singer weights/config | `40493ad90286056c7a9095035164434a79daa8c9` |
| Soul-AILab/SoulX-Singer-Preprocess | `83dc50289d22a81b1e9998f5b9e111aef7c1fdcd` |
| openai/whisper-base (SVC content encoder) | `e37978b90ca9030d5170a5c07aadb050351a65bb` |

[`soulx-singer-sources.json`](soulx-singer-sources.json) records published sizes,
Git blob IDs and LFS SHA256 values. The retained Download Manager job IDs and
actual destinations are in [`soulx-singer-downloads.json`](soulx-singer-downloads.json).
The upstream SVC wrapper is not an `nn.Module`: its Whisper-base weights are
absent from `model-svc.pt` and must be acquired separately. Vocos is included in
each main checkpoint; there is no separate vocoder download or text tokenizer.
The exact phone inventory is taken from the pinned source snapshot.

Inventory existing local files first. Acquire missing files through Download
Manager 2, with the authorized scoped Firefox Hugging Face cookie import. Wait
for completion and verify hashes before conversion. Do not install upstream's
Torch 2.2/Transformers 4.41/NumPy<2 requirements into the serving environment.
ASR/ROSVOT/its second RMVPE are optional upstream preprocessing alternatives;
they are retained locally but are not dependencies of this native inference path.
Existing offline Faster-Whisper supplies transcription in the full example.

## Offline preparation

Use an existing prepared ComfyUI Python with torch, safetensors, NumPy, SciPy,
librosa and soundfile. Run `COMFYUI_PYTHON=/path/to/python python3
scripts/setup_worktree.py --soulx`. Optional Mandarin text conversion uses
preinstalled pypinyin. English pronunciation needs the two hash-verified wheel
archives below as **data**, without importing NLTK or installing packages.

```bash
python scripts/convert_soulx_singer.py \
  --source /absolute/SoulX-Singer \
  --preprocess /absolute/SoulX-Singer-Preprocess \
  --source-code /absolute/pinned-SoulX-source \
  --whisper /absolute/whisper-base \
  --destination /absolute/ComfyUI-model-root

python scripts/prepare_soulx_pronunciation.py \
  --source /absolute/wheels-containing-g2p_en-2.1.0-and-cmudict-1.1.3 \
  --destination /absolute/ComfyUI-model-root/text_encoders/soulx_singer
```

The converter verifies source hashes, preserves FP32 tensor precision and exact
config/phone/frontend metadata, splits components, and checks every saved tensor
for bit-exact round trips. Small shared rotary tensors are copied independently.
Whisper's unused decoder is omitted explicitly. Source checkpoints are preserved.
Existing outputs are refused; `--resume` verifies their complete tensors and
provenance before continuing without overwriting weights. A `.partial` file is
retained and refused after an interrupted write: inspect it and preserve the
diagnosis before removing only that conversion's unpublished partial output.
`soulx_singer.manifest.json` is written after all components pass.

The installed model root on this machine is `/home/jack/models`:

| Category | Files |
| --- | --- |
| `diffusion_models` | `soulx_singer_svs.safetensors`, `soulx_singer_svc.safetensors` |
| `vae` | `soulx_singer_svs_vocos.safetensors`, `soulx_singer_svc_vocos.safetensors` |
| `audio_encoders` | `soulx_singer_whisper_base.safetensors`, `soulx_singer_rmvpe.safetensors`, `soulx_singer_karaoke.safetensors`, `soulx_singer_dereverb.safetensors` |
| `text_encoders/soulx_singer` | `pronunciation.safetensors`, `cmudict.dict`, license files and manifest |

Converted model components occupy about 8.5 GB. Allow additional disk for the
original checkpoints, conversion outputs and retained audio. Filenames resolve
through normal ComfyUI category roots and containment checks; no hard-coded disk
root is used by runtime nodes. Missing or mismatched files fail locally.

## Nodes and workflow order

`SoulXSingerLoader` loads SVS or SVC plus the **matching** Vocos. SVC additionally
selects the Whisper-base file. Precision is explicit; the examples start at FP32.
CPU uses FP32. FP16/BF16 are compute choices; converted storage remains FP32.
`SoulXAudioModelLoader` loads karaoke/dereverb on the selected device or RMVPE on
CPU. Loader outputs are managed handles, not separate worker sessions.

`SoulXSeparateVocals` returns the vocal estimate and original-minus-estimate
residual at 44.1 kHz, using overlapping chunks (default five seconds). Karaoke
selects the vocal stem; dereverb selects dry vocals. Use the karaoke residual
for the original backing. Keep backing harmonies separately: the separator can
leak harmonies or old lyrics, and a scalar F0 contour represents one singer only.
The dereverb residual contains removed reverb, not a new instrumental stem.

`SoulXExtractPitch` takes isolated lead `AUDIO` and the CPU RMVPE handle. It returns
JSON containing `frame_rate:50`, duration, continuous Hz values (zero unvoiced)
and transparent median-filtered note estimates. These estimates are not ROSVOT,
MIDI ground truth or a polyphonic transcription. Keep the continuous contour when
expression matters. SoulX itself quantizes F0 to 20-cent bins.

`SoulXAlignLyrics` takes pitch JSON and timed regions (`start`, `end`, `text`, in
seconds). It also accepts Faster-Whisper segment JSON with nested `words`.
Empty replacement lyrics use original word times. New lyrics use one phrase per
region, separated by `|` or newlines, with new syllables allocated across retained
voiced timing. Each syllable keeps one phoneme event with its median note pitch;
vibrato and glides remain in F0 rather than fragmenting syllables into tiny events.
The finer `notes` in pitch JSON are inspection estimates, not forced syllable
boundaries. Different syllable counts create different allocations. No audio
time stretching occurs. The result is an **estimated** alignment, split into
bounded segments; inspect and correct it before treating timing as musical truth.
English uses CMU pronunciations and the offline G2P predictor for unknown words.
Homographs/names need review; this path does not use upstream's POS tagger.
Mandarin pypinyin is optional; explicit `zh_`/`yue_` tokens permit Mandarin and
Cantonese metadata without a new worker. Cantonese automatic G2P is not supplied.

Pronunciation overrides map a word to an ARPAbet array, for example
`{"Anjuna":["AH0","N","JH","UW1","N","AH0"]}`. Preserve stress digits.
Inspect the returned JSON's `rows`: each row owns start/end, phoneme, note pitch,
note type, word and syllable identifiers. Edit those rows in a multiline string
and connect the corrected string to synthesis. Rows are authoritative when present;
upstream array/string metadata without rows remains accepted. Upstream note types
and complete phoneme inventory remain supported. `time` uses milliseconds for
segment placement; row times are seconds relative to each segment.

`SoulXSynthesize` consumes isolated reference audio with its matching metadata,
target metadata, melody/score control, pitch shift, steps, guidance and seed. The
first reference metadata segment supplies identity; provide corresponding audio
from that same segment. Prefer a short, correctly transcribed reference clip; using
the whole target passage as its own identity reference reproduced the original
words in the retained song test. The full example exposes `TrimAudioDuration` and
separate reference pitch/ASR for this choice. Target segment times preserve gaps
and backing duration.
Default 32 midpoint steps and guidance 3 match upstream, including prompt removal
in the null branch and CFG standard-deviation rescaling. Zero pitch shift retains
the requested melody; automatic pitch shifting is intentionally not a hidden default.

`SoulXVoiceConvert` consumes singer/source audio plus both extracted F0 JSONs;
it preserves source lyrics rather than replacing them. The Whisper content encoder
has a 30-second window: reference uses its first 30 seconds, long targets use
overlapping windows with crossfades. Short externally supplied contours retain
their samples and receive an unvoiced tail, rather than being stretched. Extract
pitch from the same complete audio for normal operation. Output is mono 24 kHz.
Merge it with backing via standard `AudioMerge` (backing as `audio1`, since that
node owns final length), adjust level with standard audio nodes, then save.

## Examples and inspection

Versioned editor graphs and matching API prompts are under
[`examples/soulx_singer`](../examples/soulx_singer). Local editable copies are in
`/home/jack/comfyui/user/default/workflows/SoulX-Singer/`:

* `SoulX - Singing Voice Conversion.json`: extract both F0 inputs, then SVC.
* `SoulX - Melody or Score Synthesis.json`: editable upstream English metadata;
  switch the synthesis control between continuous F0 and score.
* `SoulX - Reference Song Lyric Replacement.json`: mixed song → karaoke/dereverb
  → pitch + installed offline ASR, a separate short singer clip and its metadata
  → original/replacement alignment → native SVS
  → original-backing mix. Preview nodes expose pitch and both alignments.

The first two use the separately prepared upstream example clips. The full
example's four replacement phrase regions are for the retained 12.9-second
reference excerpt; change the audio, phrase times and lyrics together for another
song. Its Faster-Whisper nodes require installed Cutlery-Nodes and local ASR
weights. CPU/int8 ASR avoids a second GPU owner. No model is fetched by the graph.
Its full-song transcript preview helps author phrase regions; the selected reference
clip is transcribed separately on its own time axis. Copy inspected alignment JSON
to an editable string before making corrections.
The full graph's `rows`/notes remain inspectable even when synthesis is disconnected.
Submit the API prompt through normal `/prompt`, keep its prompt ID, and follow
queue/history after a timeout before retrying. Do not submit editor JSON as a prompt.

## Verification and limits

```bash
OMP_NUM_THREADS=1 python -m pytest tests-unit/comfy_test/soulx_singer_test.py -q
python -m scripts.check_soulx_singer --models /absolute/models \
  --reference /absolute/upstream-run --output /absolute/new-native-run --device cpu
```

`scripts/soulx_singer_reference.py` runs the actual pinned upstream code in a
separate prepared validation environment. It requires upstream's Transformers
4.41.2 and dependencies, local assets, explicit output/device/precision/seed and
steps. `--task svc --whisper-source /absolute/whisper-base` redirects only upstream's
online loader to local files and makes placement explicit; inference math is
unchanged. The reference uses compatible Torch 2.14.1/TorchAudio 2.11, rather than
claiming its obsolete Torch 2.2 wheel supports this GPU. Keep reference dependencies
outside serving. Both reference and native scripts preserve requests, tensors and
floating-point audio, and refuse an existing output directory.

CPU checks establish real-model default-step numerical parity and preprocessing
fidelity; GPU queue checks establish execution/lifecycle on this RTX 5090. Their
compact retained summary is [`soulx-singer-validation.json`](soulx-singer-validation.json).
The isolated candidate service uses port 8189 and its own input/output directories;
the serving 8188 service is not restarted by setup, conversion or validation.
Source integration, candidate acceptance and serving activation are separate.

The 32-step CPU SVS and SVC checks pass the declared mel/waveform tolerances;
RMVPE and Roformer match actual upstream preprocessing. A two-step SVC diagnostic
fails the stricter pointwise waveform gate (maximum error 0.00205) despite close
mel agreement; its failed report is retained without relaxing tolerances. The GPU
lifecycle suite includes an uncached, bit-exact same-seed repeat. Short fixtures
used about 5.5 GiB driver-reported VRAM; the complete separation/synthesis excerpt
used about 8 GiB. These are measurements on this GPU, not minimum-capacity claims.

**Song quality acceptance remains incomplete.** On the 12.9-second read-only source
excerpt, default 32-step synthesis with a short reference gave median absolute
pitch error 8.8 cents and 94.7% of jointly voiced frames within 50 cents on the
shared 50 Hz time grid. A separately reported bounded DTW diagnostic permits
300 ms of warp and used at most 80 ms. These pitch results do not pass the lyric
gate: blind ASR misheard several requested replacement words. Guidance 5/64 steps
and guidance 8/32 steps did not resolve that gate. The complete 25-node example
(including both existing offline CPU ASR calls) also executes successfully: its
direct median pitch error is 8.95 cents, with 93.15% within 50 cents, but its
independent wording check still fails. Original full-passage reference
cases copied the old lyrics and remain retained. Edit phrase/phoneme/syllable rows
and pronunciation overrides, then re-evaluate; do not label these drafts as an
accepted replacement. Automatic ASR times are not forced phoneme alignment. No
human listening, validated singer-identity comparison or full-song join review
has been performed. Finite audio, exact placement and no sample clipping in a
short draft do not establish subjective mix quality.

Assess lyrics/phonemes, voiced-mask and temporally aligned pitch, syllable/note
timing, singer identity and mix artifacts separately. ASR is only a wording check.
Automatic pitch/timing scores are estimates, not human listening or proof of exact
identity. Retain failed gates and original outputs; do not hide an outlier by
changing the reference/seed/tolerance after evaluation. Long songs, other devices,
low-VRAM capacity, Cantonese G2P and subjective listening require separate checks.

Cancellation follows ComfyUI interruption points; managed models are reclaimable
through ComfyUI's existing memory owner. After an interrupted or failed prompt,
inspect its history and returned error, correct the cause, then submit deliberately.
Never clear someone else's queue or restart a busy serving instance. Native GPU
validation outside the serving queue must use the existing coordinated GPU hold
and yield when serving work arrives; release only the exact task-owned hold.
