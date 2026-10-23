# Video component extraction

This fork extends upstream **Get Video Components** without changing its node ID,
five output sockets, or default colour conversion. PyAV already uses FFmpeg
libraries and threaded decoding; this change retains that backend. It introduces
no executable, Python dependency, GPU allocation, download or network request.

## Workflow and controls

Use **Load Video → Trim Video → Get Video Components**. Trimming a file-backed
VIDEO before extraction lets the reader seek and stop at the requested interval.
The new inputs are optional advanced controls; older saved graphs and direct
`GetVideoComponents.execute(video)` calls retain their existing behavior.

| Input | Default | Behavior |
|---|---|---|
| `components` | `Images and audio` | Also offers `Images only`, `Audio only`, and `Metadata only`. File inputs demux/decode only requested streams. Disabled images produce an empty float32 batch of shape `(0, 0, 0, 3)`; disabled or absent audio produces `None`. |
| `frame_stride` | `1` | Keep the first frame in the active trim, then every Nth frame. Returned image FPS is source average FPS divided by N. Audio is unchanged. Audio-only and metadata-only calls report source FPS. |
| `max_size` | `0` | Bound the longest image edge, preserving aspect ratio without enlargement. Zero keeps dimensions. Bilinear antialiased resizing occurs after rotation, colour conversion and crop, before file frames are accumulated. |
| `decode_mode` | `Preserve precision` | `8-bit SDR` allows packed RGB8/RGBA8 conversion for 8-bit inputs whose stream transfer is SDR or unspecified. It may change rounding, clipping of out-of-range SDR values and chroma interpolation. High-bit-depth, HDR and floating inputs retain their precision path. |

Unconnected output sockets do not automatically suppress extraction. Select the
appropriate component mode explicitly. Bit depth and colour space outputs still
describe the source; the image output remains float32. An audio-only or
metadata-only result has no image frames and must not be fed to an image consumer.
The nullable Cutlery wrapper benefits from default buffer improvements but does
not expose these new controls.

## Python contract

Existing `VideoInput.get_components()` signatures remain unchanged. A new method
is available on the base interface:

```python
components = video.get_components_selected(
    include_images=True,
    include_audio=False,
    frame_stride=2,
    max_size=640,
    compact=False,
)
```

Both inclusion flags default to true, stride to 1, size to 0 and compact to false.
Stride below 1 or a negative size raises `ValueError`. The return type remains
`VideoComponents`, including alpha and metadata where available. Audio stays at
the original trimmed rate and duration. `VideoFromFile` implements stream
selection and conversion skipping. Tensor inputs, concatenated videos and other
`VideoInput` subclasses use a base fallback that first calls their existing
`get_components()`, then selects/resizes; it does not provide the file reader's
decode savings. A custom implementation can override the new method for its own
efficient selection. Default node execution calls the original method directly.

## Allocation and correctness

The reader fills CPU float32 Torch storage through NumPy views. RGB8
intermediates are normalized directly into the destination instead
of keeping uint8, converted-float and division-result batches simultaneously.
The existing float/HDR plane reading, alignment padding, display rotation, crop,
audio selection, trim and metadata handling are retained.

Container frame counts and trim-duration estimates are allocation hints only.
Missing or underestimated counts spill into additional buffers; overestimated
counts are copied down to the actual number of frames. Initial reservations from
hints are limited to 2 GiB per image/alpha buffer (at least one frame); unknown
counts use blocks of at most 64 MiB (at least one frame). If the actual output
fills one buffer exactly, it is returned without a final full-size copy. Otherwise
the reader allocates the exact output and releases each source buffer as it is
copied. No buffer persists on the video object or across executions.

The final IMAGE tensor still holds every selected frame. For example, 300 RGB
float32 frames at 1920×1080 require about 6.95 GiB, excluding audio, decoder,
temporary buffers and framework overhead. Buffer consolidation can require
additional memory; this is not streaming output or a fixed memory ceiling.
Temporal subsampling avoids RGB conversion/storage for discarded frames, but
inter-frame codecs may still decode them. Resizing saves accumulated/output
memory rather than full-resolution decode and colour-conversion work. Selection
uses decoded frame order and average FPS; it does not resample variable-rate
timestamps to a constant rate. Retaining the last partial stride can slightly
extend image-derived duration; audio is not stretched to compensate.

## Verification and benchmarking

Use the repository's prepared runtime (Python >=3.10, Torch, NumPy and PyAV from
its requirements). Developer setup is
`COMFYUI_PYTHON=/path/to/python python3 scripts/setup_worktree.py`.

```bash
"$COMFYUI_PYTHON" -m pytest -q tests-unit/comfy_api_test/video_extraction_test.py tests-unit/comfy_api_test/video_types_test.py tests-unit/comfy_api_test/video_bit_depth_test.py tests-unit/comfy_api_test/video_accumulation_test.py tests-unit/comfy_extras_test/test_video_preview_nodes.py
PYTHONPATH=. "$COMFYUI_PYTHON" scripts/benchmark_video_extraction.py /tmp/video-extraction.mp4 --create
PYTHONPATH=. "$COMFYUI_PYTHON" scripts/benchmark_video_extraction.py /tmp/video-extraction.mp4
PYTHONPATH=. "$COMFYUI_PYTHON" scripts/benchmark_video_extraction.py /tmp/video-extraction.mp4 --mode compact
PYTHONPATH=. "$COMFYUI_PYTHON" scripts/benchmark_video_extraction.py /tmp/video-extraction.mp4 --mode selected
```

Fixture creation writes the requested path. Benchmark runs read it, use CPU-only
decoding, and print wall time, process peak RSS, tensor size/shape and FPS. Optional
`--profile /tmp/extraction.prof` writes cProfile data. Run each comparison in a
fresh process with `PYTHONPATH` pointing to the desired checkout and the same
fixture/runtime. Process RSS includes framework imports and decoder memory;
timings vary with other workloads. The synthetic fixture is H.264 SDR, 120 frames,
1280×720 at 30 FPS; it does not establish every codec's speed or GPU performance.

On 2026-10-23, three alternating fresh-process trials on the local Ryzen 7 9700X
with Python 3.13.16, Torch 2.14.1 and the same prepared PyAV runtime gave:

| Reader | Median seconds (range) | Median peak process RSS, MiB |
|---|---|---|
| Baseline, committed `0a35d6421` | 1.256 (0.850–1.612) | 3515 |
| Direct-buffer default, same precision | 0.718 (0.681–0.749) | 2260 |
| Optional compact 8-bit SDR | 0.370 (0.360–0.661) | 2228 |

The default reduced median latency by about 43% and peak process RSS by about
36% on this fixture; compact conversion was about 3.4× faster than baseline.
All three returned the same tensor shape, frame count and FPS. Compact pixels
are intentionally not required to match the precision path. Other active host
work contributed to timing variation. These figures do not promise a speedup
for every codec, image size or host; exact-default pixel parity is covered by
the regression suite. Stride 2 plus a 640px limit produces a 158 MiB output
instead of 1266 MiB, with 60 frames at 15 FPS instead of 120 at 30 FPS.

If an input fails, retain the original media and check it independently with
PyAV/FFmpeg. Retry with `Preserve precision` for compact-mode issues, and reproduce
with defaults to distinguish selection from source-decoder failures. Do not
restart an active ComfyUI queue or replay an uncertain queued job. Source changes
require an idle service restart before live node discovery exposes the controls.
