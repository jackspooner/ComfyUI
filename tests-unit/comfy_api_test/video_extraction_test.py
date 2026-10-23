import io
from fractions import Fraction

import av
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from comfy_api.latest._input_impl.video_types import VideoFromComponents, VideoFromFile, VideoFromList, _VideoFrameBuffer
from comfy_api.latest._util.video_types import VideoComponents
from comfy_extras.nodes_video import GetVideoComponents


@pytest.fixture
def source(tmp_path):
    images = torch.linspace(0, 1, 7 * 18 * 34 * 3).reshape(7, 18, 34, 3)
    audio = {"waveform": torch.sin(torch.arange(7000).float() / 10)[None, None], "sample_rate": 8000}
    path = str(tmp_path / "source.mp4")
    VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(8), audio=audio)).save_to(path)
    return VideoFromFile(path)


def test_default_pixels_match_previous_conversion(source):
    with av.open(source.get_stream_source()) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Match the original reader's alignment workaround on this 34px-wide video.
        graph = av.filter.Graph()
        src = graph.add_buffer(template=stream)
        pad = graph.add("pad", "64:32:0:0")
        fill = graph.add("fillborders", "left=0:right=30:top=0:bottom=14:mode=smear")
        sink = graph.add("buffersink")
        src.link_to(pad)
        pad.link_to(fill)
        fill.link_to(sink)
        graph.configure()
        expected = []
        for frame in container.decode(stream):
            src.push(frame)
            expected.append(torch.from_numpy(sink.pull().to_ndarray(format="gbrpf32le")[:18, :34].copy()))
    torch.testing.assert_close(source.get_components().images, torch.stack(expected), rtol=0, atol=0)


@pytest.mark.parametrize("hint", [0, 1, 7, 1000000])
def test_frame_count_is_only_a_hint(hint):
    pixels = np.arange(18 * 34 * 3, dtype=np.float32).reshape(18, 34, 3)
    buffer = _VideoFrameBuffer(hint)
    for index in range(7):
        buffer.append(pixels + index)
    result = buffer.finish(3)
    torch.testing.assert_close(result, torch.from_numpy(np.stack([pixels + index for index in range(7)])))
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()
    assert not buffer.blocks


def test_variable_dimensions_do_not_broadcast():
    buffer = _VideoFrameBuffer(2)
    buffer.append(np.zeros((18, 34, 3)))
    with pytest.raises(ValueError, match="incompatible dimensions"):
        buffer.append(np.zeros((18, 1, 3)))


def test_accurate_count_returns_filled_storage_without_copy():
    buffer = _VideoFrameBuffer(2)
    for _ in range(2):
        buffer.append(np.zeros((18, 34, 3)))
    storage = buffer.blocks[0]
    assert buffer.finish(3).data_ptr() == storage.ctypes.data
    assert not buffer.blocks


def test_uint8_normalization_is_exact():
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
    buffer = _VideoFrameBuffer(1)
    buffer.append(pixels)
    torch.testing.assert_close(buffer.finish(1), torch.from_numpy(pixels)[None].float() / 255, rtol=0, atol=0)


def test_selection_before_accumulation_preserves_trim_crop_and_audio(source):
    trimmed = source.as_trimmed(0.125, 0.5).as_cropped(2, 2, 28, 14)
    complete = trimmed.get_components()
    selected = trimmed.get_components_selected(frame_stride=2, max_size=14)
    expected = F.interpolate(complete.images[::2].movedim(-1, 1), size=(7, 14), mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
    torch.testing.assert_close(selected.images, expected)
    assert selected.frame_rate == Fraction(4)
    assert selected.audio["sample_rate"] == complete.audio["sample_rate"]
    torch.testing.assert_close(selected.audio["waveform"], complete.audio["waveform"], rtol=0, atol=0)


def test_selection_resizes_after_rotation(source, monkeypatch):
    complete = source.get_components().images
    monkeypatch.setattr("comfy_api.latest._input_impl.video_types._rotation_quadrant", lambda frame: 1)
    actual = source.get_components_selected(frame_stride=2, max_size=17)
    rotated = torch.rot90(complete[::2], 1, (1, 2))
    expected = F.interpolate(rotated.movedim(-1, 1), size=(17, 9), mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
    torch.testing.assert_close(actual.images, expected)


def test_float_source_keeps_out_of_range_values_with_compact_resize(tmp_path):
    path = str(tmp_path / "float.exr")
    pixels = np.linspace(-2, 12, 16 * 32 * 3, dtype=np.float32).reshape(16, 32, 3)
    codec = av.CodecContext.create("exr", "w")
    codec.width, codec.height, codec.pix_fmt = 32, 16, "gbrpf32le"
    codec.time_base = Fraction(1)
    codec.options = {"format": "float", "compression": "zip16"}
    frame = av.VideoFrame.from_ndarray(pixels, format="gbrpf32le")
    frame.pts, frame.time_base = 0, Fraction(1)
    with open(path, "wb") as file:
        file.write(b"".join(bytes(packet) for packet in list(codec.encode(frame)) + list(codec.encode(None))))
    expected = F.interpolate(torch.from_numpy(pixels)[None].movedim(-1, 1), size=(8, 16), mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
    actual = VideoFromFile(path).get_components_selected(compact=True, max_size=16)
    torch.testing.assert_close(actual.images, expected, rtol=0, atol=0)
    assert actual.images.min() < 0 and actual.images.max() > 1


@pytest.mark.parametrize("images,audio,streams", [(True, False, ["video"]), (False, True, ["audio"]), (False, False, [])])
def test_unused_streams_are_not_demuxed(source, images, audio, streams):
    with av.open(source.get_stream_source()) as container:
        requested = []

        class Container:
            def __getattr__(self, name):
                return getattr(container, name)

            def demux(self, *selected):
                requested.extend(stream.type for stream in selected)
                return container.demux(*selected)

        result = source.get_components_internal(Container(), include_images=images, include_audio=audio)
    assert requested == streams
    assert bool(result.images.shape[0]) == images
    assert (result.audio is not None) == audio


def test_compact_8bit_matches_rgb24(source):
    with av.open(source.get_stream_source()) as container:
        expected = torch.stack([torch.from_numpy(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)]).float() / 255
    actual = source.get_components_selected(compact=True)
    torch.testing.assert_close(actual.images, expected, rtol=0, atol=0)


@pytest.mark.parametrize("depth,color_space", [(10, "sRGB"), (8, "HDR"), (10, "HDR PQ")])
def test_compact_preserves_high_depth_and_hdr(tmp_path, depth, color_space):
    path = str(tmp_path / "hdr.mp4")
    images = torch.linspace(0.25, 0.30, 64).reshape(1, 1, 64, 1).expand(2, 32, 64, 3).contiguous()
    VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(8)), bit_depth=depth, color_space=color_space).save_to(path)
    video = VideoFromFile(path)
    torch.testing.assert_close(video.get_components_selected(compact=True).images, video.get_components().images, rtol=0, atol=0)


def test_alpha_selected_and_resized(tmp_path):
    path = str(tmp_path / "alpha.mov")
    pixels = np.arange(4 * 18 * 34 * 4, dtype=np.uint8).reshape(4, 18, 34, 4)
    with av.open(path, "w") as container:
        stream = container.add_stream("png", rate=8)
        stream.width, stream.height, stream.pix_fmt = 34, 18, "rgba"
        for image in pixels:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgba")))
        container.mux(stream.encode(None))
    video = VideoFromFile(path)
    complete = video.get_components()
    selected = video.get_components_selected(frame_stride=2, max_size=17, compact=True)
    expected = F.interpolate(torch.from_numpy(pixels[::2].copy()).float().movedim(-1, 1) / 255, size=(9, 17), mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
    torch.testing.assert_close(selected.images, expected[..., :3])
    torch.testing.assert_close(selected.alpha, expected[..., 3:])
    torch.testing.assert_close(complete.alpha, torch.from_numpy(pixels[..., 3:].copy()).float() / 255, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["tensor", "list"])
def test_nonfile_selection_fallback(kind):
    images = torch.linspace(-2, 12, 8 * 16 * 32 * 3).reshape(8, 16, 32, 3)
    source = VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(8)))
    video = source if kind == "tensor" else VideoFromList([source, source])
    expected = video.get_components().images[::3]
    actual = video.get_components_selected(frame_stride=3, max_size=8, include_audio=False)
    expected = F.interpolate(expected.movedim(-1, 1), size=(4, 8), mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
    torch.testing.assert_close(actual.images, expected)
    assert actual.frame_rate == Fraction(8, 3)
    assert actual.audio is None


def test_bytesio_and_empty_trim(source):
    with open(source.get_stream_source(), "rb") as file:
        video = VideoFromFile(io.BytesIO(file.read()))
    assert video.get_components_selected(frame_stride=3, max_size=100).images.shape == (3, 18, 34, 3)
    empty = VideoFromFile(source.get_stream_source(), start_time=10, duration=1).get_components()
    assert empty.images.shape == (0, 0, 0, 3)


def test_node_defaults_and_metadata_selection(source):
    default = GetVideoComponents.execute(source).result
    metadata = GetVideoComponents.execute(source, components="Metadata only").result
    assert metadata[0].shape == (0, 0, 0, 3)
    assert metadata[1] is None
    assert metadata[2:] == default[2:]
    schema = GetVideoComponents.define_schema()
    assert all(input.optional and input.advanced and input.tooltip for input in schema.inputs[1:])
    assert [output.display_name for output in schema.outputs] == ["images", "audio", "fps", "bit_depth", "color_space"]


def test_default_call_preserves_existing_reader_override(source):
    class Reader(VideoFromFile):
        def get_components_internal(self, container):
            return super().get_components_internal(container)

    actual = GetVideoComponents.execute(Reader(source.get_stream_source())).result
    torch.testing.assert_close(actual[0], source.get_components().images, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"components": "unknown"}, {"decode_mode": "unknown"}])
def test_node_rejects_unknown_modes(source, kwargs):
    with pytest.raises(ValueError, match="Unknown video"):
        GetVideoComponents.execute(source, **kwargs)


@pytest.mark.parametrize("kwargs", [{"frame_stride": 0}, {"max_size": -1}])
def test_invalid_selection(source, kwargs):
    with pytest.raises(ValueError, match="frame_stride must be"):
        source.get_components_selected(**kwargs)
