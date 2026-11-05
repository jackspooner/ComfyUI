from fractions import Fraction
from threading import Lock
from types import SimpleNamespace

import av
import numpy as np
from PIL import Image
import pytest
import torch

import nodes
from comfy import execution_metrics
from comfy_api.latest._input_impl.video_types import VideoFromFile
from comfy_api.latest._input_impl.video_metrics import VideoDecodeMetrics
from app.run_timing import RunTimingStore
from app.run_timing_metrics import attach_runtime_metrics
from comfy_execution.utils import CurrentNodeContext


@pytest.fixture(autouse=True)
def cpu_outputs(monkeypatch):
    execution_metrics.set_observer(None)
    monkeypatch.setattr(nodes.comfy.model_management, 'intermediate_device', lambda: 'cpu')
    monkeypatch.setattr(nodes.comfy.model_management, 'intermediate_dtype', lambda: torch.float32)
    yield
    execution_metrics.set_observer(None)


def png(tmp_path, monkeypatch, alpha):
    array = np.arange(10 * 12 * 4, dtype=np.uint16).reshape(10, 12, 4).astype(np.uint8)
    path = tmp_path / 'private-source.png'
    Image.fromarray(array if alpha else array[:, :, :3]).save(path)
    monkeypatch.setattr(nodes, 'resolve_image_path', lambda _: str(path))
    return array


@pytest.mark.parametrize('alpha', [False, True])
@pytest.mark.parametrize('cls', [nodes.LoadImage, nodes.LoadImageOutput])
def test_real_png_decode_alpha_mask_and_pixels_survive_observation_and_failure(tmp_path, monkeypatch, alpha, cls):
    array = png(tmp_path, monkeypatch, alpha)
    expected = cls().load_image('PRIVATE')
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = cls().load_image('PRIVATE')
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], torch.from_numpy(array[:, :, :3].copy()).float().div(255).unsqueeze(0))
    if alpha:
        torch.testing.assert_close(actual[1], 1 - torch.from_numpy(array[:, :, 3].copy()).float().div(255).unsqueeze(0))
    else:
        assert actual[1].shape == (1, 64, 64) and torch.count_nonzero(actual[1]) == 0
    summary = dict(events)['video_decode_work']
    assert summary['complete'] and summary['fullDecode'] and summary['videoCodec'] == 'png'
    assert summary['processedFrames'] == 1 and summary['processedPixels'] == 120
    assert summary['alphaFrames'] == int(alpha) and summary['audioFrames'] == 0
    assert len(events) == 4
    assert 'PRIVATE' not in str(events) and str(tmp_path) not in str(events)
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer offline')))
    for before, after in zip(expected, cls().load_image('PRIVATE')):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize('channel,effective,index', [('alpha', 'alpha', None), ('red', 'red', 0), ('green', 'green', 1),
                                                  ('blue', 'blue', 2), ('other', 'red', 0)])
def test_mask_loader_observes_actual_selection_without_changing_legacy_channel_fallback(tmp_path, monkeypatch, channel, effective, index):
    array = png(tmp_path, monkeypatch, True)
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    mask, = nodes.LoadImageMask().load_image_mask('PRIVATE', channel)
    expected = torch.from_numpy(array[:, :, 3 if index is None else index].copy()).float().div(255).unsqueeze(0)
    torch.testing.assert_close(mask, 1 - expected if index is None else expected)
    selected = dict(events)['image_mask_selected']
    assert selected == {'shape': (1, 10, 12), 'channel': effective, 'mode': 'alpha_mask' if index is None else 'channel_clone'}


def test_pillow_animation_fallback_preserves_rgb_and_missing_alpha_masks(tmp_path, monkeypatch):
    path = tmp_path / 'animation.webp'
    frames = [Image.new('RGB', (12, 10), color) for color in ('red', 'blue')]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, lossless=True)
    monkeypatch.setattr(nodes, 'resolve_image_path', lambda _: str(path))
    monkeypatch.setattr(nodes.InputImpl, 'VideoFromFile', lambda _: SimpleNamespace(get_components=lambda: SimpleNamespace(images=torch.zeros(0, 0, 0, 3))))
    expected = nodes.LoadImage().load_image('PRIVATE')
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = nodes.LoadImage().load_image('PRIVATE')
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    work = dict(events)['image_pillow_work']
    assert work['frames'] == work['selectedFrames'] == 2 and work['decodedPixels'] == 240
    assert work['alphaFrames'] == 0 and work['consistent']
    assert actual[0].shape == (2, 10, 12, 3) and actual[1].shape == (2, 64, 64)
    execution_metrics.set_observer(lambda *_: (_ for _ in ()).throw(OSError('observer offline')))
    for before, after in zip(expected, nodes.LoadImage().load_image('PRIVATE')):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def media_file(tmp_path, frames):
    path = tmp_path / 'tiny.mkv'
    with av.open(str(path), 'w') as container:
        video = container.add_stream('h264', rate=4)
        video.width, video.height, video.pix_fmt = 48, 40, 'yuv420p'
        audio = container.add_stream('pcm_s16le', rate=8000)
        audio.layout = 'mono'
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((40, 48, 3), index, dtype=np.uint8), format='rgb24')
            frame.pts = index
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 800), dtype=np.int16), format='s16', layout='mono')
        frame.sample_rate, frame.pts, frame.time_base = 8000, 0, Fraction(1, 8000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return str(path)


@pytest.mark.parametrize('frames', [2, 16])
def test_decoder_reports_audio_work_even_when_image_loader_discards_the_audio(tmp_path, monkeypatch, frames):
    path = media_file(tmp_path, frames)
    expected = VideoFromFile(path).get_components()
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    actual = VideoFromFile(path).get_components()
    torch.testing.assert_close(actual.images, expected.images, rtol=0, atol=0)
    torch.testing.assert_close(actual.audio['waveform'], expected.audio['waveform'], rtol=0, atol=0)
    assert len(events) == 1
    work = events[0][1]
    assert work['complete'] and work['videoFrames'] == work['processedFrames'] == frames
    assert work['alignmentPixels'] == frames * 64 * 64
    assert work['audioStream'] and work['audioFrames'] > 0 and work['audioChannels'] == 1
    assert work['audioSamples'] == work['resampledAudioSamples'] == work['retainedAudioSamples'] == 800
    assert work['audioRate'] == 8000 and work['audioCodec'] == 'pcm_s16le'
    monkeypatch.setattr(nodes, 'resolve_image_path', lambda _: path)
    events.clear()
    images, masks = nodes.LoadImage().load_image('PRIVATE')
    assert images.shape[0] == masks.shape[0] == frames
    assert len(events) == 4 and dict(events)['video_decode_work']['retainedAudioSamples'] == 800


@pytest.mark.parametrize('node_class', [nodes.LoadImage, nodes.LoadImageOutput, nodes.LoadImageMask])
def test_real_native_image_work_flows_into_the_existing_stage_owner(tmp_path, monkeypatch, node_class):
    png(tmp_path, monkeypatch, True)
    store = RunTimingStore(str(tmp_path / 'timing.sqlite'), host_id='fixture-host', gpu_uuid='GPU-fixture')
    item = (1, 'fixture-prompt', {'image': {'class_type': node_class.__name__, 'inputs': {'image': 'PRIVATE'}}}, {}, ['image'])
    store.event('accepted', item); store.event('started', item)
    attach_runtime_metrics(SimpleNamespace(timing=store, mutex=Lock(), currently_running={0: item}))
    with CurrentNodeContext('fixture-prompt', 'image'):
        with execution_metrics.stage_scope('image', node_class.__name__):
            if node_class is nodes.LoadImageMask:
                node_class().load_image_mask('PRIVATE', 'green')
            else:
                node_class().load_image('PRIVATE')
    stage = store.page('0', 100, [item], [])['active']['records'][0]['stages'][0]
    assert stage['phase'] == 'succeeded' and stage['workload']['complete']
    assert stage['workload']['resolved']['imageDecode.videoCodec']['value'] == 'png'
    assert stage['workload']['measurements']['image_source_decode.pyav.seconds']['value'] >= 0
    assert 'PRIVATE' not in str(stage)


def test_variable_or_missing_native_metadata_stays_incomplete_without_affecting_decoder_results():
    events = []
    execution_metrics.set_observer(lambda event, facts: events.append((event, facts)))
    work = VideoDecodeMetrics(True, False)
    frame = SimpleNamespace(width=12, height=8, format=SimpleNamespace(name='rgb24', components=[SimpleNamespace(bits=8)]))
    work.video([frame, SimpleNamespace(width=14, height=8, format=frame.format)])
    work.processed(frame, (8, 12, 3), 'rgb24', False, 0, False, False)
    work.finish(torch.zeros(1, 8, 12, 3), None, None)
    assert events[0][1]['complete'] is False and events[0][1]['videoFrames'] == 2
    work = VideoDecodeMetrics(True, False)
    work.video([object()]); work.audio([object()]); work.resampled(object())
    work.finish(torch.zeros(0, 0, 0, 3), None, None)
    assert events[-1][1]['complete'] is False
