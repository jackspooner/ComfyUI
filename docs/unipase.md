# Native UniPASE speech restoration

This fork implements the released [UniPASE pipeline](https://github.com/Xiaobin-Rong/unipase)
by Xiaobin Rong et al. for speech enhancement/restoration. It is generative: wording,
timbre and artifacts need listening review. General music enhancement and perfect
repair of synthetic audio are not promised. The complete URGENT winning hybrid is
not this standalone released pipeline. [Pinned sources](unipase-sources.json).

## Offline setup

Use the prepared ComfyUI Python (PyTorch, safetensors and existing audio dependencies).
Run `COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py`
in each development worktree. No additional inference dependency is required.
Download only the four named files from
[HF revision f0b4d4c](https://huggingface.co/Xiaobin-Rong/unipase/tree/f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0)
through Download Manager 2 into an explicit local source directory:
`DeWavLM-Omni.pt`, `Adapter.pt`, `Vocoder_DWO-L1.pt`, `PostNet.pt`.
The evaluation-only `Vocoder_WavLM-L24.pt` is unnecessary.

```sh
/path/to/python scripts/convert_unipase.py /absolute/source /configured/models/audio_models/unipase.safetensors
```

The converter requires the exact published SHA256s, restricted `weights_only`
loading, complete native tensor inventory, FP32 finite tensors and saved-value
round trips. It folds positional convolution weight normalization and maps feature
convolution keys to existing Wav2Vec2 layers. Originals remain intact; roughly
2.2 GB source plus 2.2 GB output and several GB RAM are required. A `.partial`
marks interrupted publication: inspect it and use a fresh destination. Never load
an incomplete file. A self-contained artifact is published before its audit manifest.

Code revision: `857b60ad05d37a2cf6d7a89883ec9fc4fc164b45` (MIT); checkpoint
card: Apache-2.0. Component notices for Microsoft WavLM, WavTokenizer, Cisco PASE
and ESPnet are retained in `comfy/ldm/unipase/licenses/`. No upstream training
code, model downloads or outbound requests run in ComfyUI.

## Nodes and behavior

Use **Load UniPASE → UniPASE Speech Restoration → Save Audio** with standard AUDIO.
The loader resolves its selected safetensors file within `audio_models`; all four
networks use native managed patchers. `output_sample_rate=0` preserves the source
rate; `enable_plc=true` detects near-zero 20 ms packets. These are the only model
controls. No prompt, seed, enhancement strength or implicit EQ is used.

Every channel/batch item is restored independently. This extends upstream's
mono/first-channel file behavior; stereo spatial fidelity is unverified. Input is
resampled to 16 kHz. Output below 16 kHz is resampled from the vocoder; 16 kHz skips
PostNet. Higher rates use PostNet at 48 kHz before final resampling. Sample count
is `floor(input_samples * output_rate / input_rate)`, preserving duration to a sample.
The input track's peak is restored once after stitching. Silent tracks remain silent.

Long recordings use upstream's eight-second segments, four-second hop and
mid-overlap boundary crops, with final short tails merged into the previous segment
(up to ten seconds). Tiny clips are padded enough for the encoder. Changed-rate
stitch offsets use the output rate; upstream's short-clip chunk failure and
input-rate offset bug are not copied. Input/output CPU memory still scales with
recording length; neural activations are segment-bounded. No persistent tensor
cache or model-owned residency manager is added. ComfyUI owns cancellation,
loading, offload and cleanup; PostNet's small native LSTM component loads in full.

## Verification and recovery

```sh
/path/to/python -m pytest tests-unit/comfy_test/unipase_test.py -q
/path/to/python scripts/check_unipase.py --upstream /pinned/unipase/source --sources /local/originals --model /models/audio_models/unipase.safetensors --output /existing/review/directory
```

The CPU harness checks imported upstream source hashes and original checkpoints,
then compares three official speech fixtures at 16/48 kHz and saves before/native/
upstream WAVs plus metrics. It replaces only ESPnet's abstract base and PReLU
lookup with their native PyTorch equivalents; upstream numerical code is unchanged.
Reference Python needs upstream import dependencies (TorchAudio and OmegaConf);
serving does not. [Acceptance and measured limits](unipase-acceptance.md).

Native queue failures keep their prompt ID. Inspect history before retrying an
uncertain submit; never clear another job. Missing nodes mean the process has not
loaded the integration. Only use the supported idle activation workflow. Model
loading or fixture tests alone do not certify GPU, low-VRAM or perceptual quality.

RouterAudio's `unipase` profile and MediaTools use the existing durable
`audio_generate` → `audio_job` flow with `audioReferences.source`. Their required
prompt is a descriptive job label, without model text conditioning. Discover the
selected live card and controls; source integration does not establish activation.
