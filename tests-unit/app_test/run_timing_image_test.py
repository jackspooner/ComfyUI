import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema
import pytest

from app.run_timing import RunTimingStore, digest
from app.run_timing_image import VIDEO_COUNTS, observe_image_metrics
from app.run_timing_workload import describe


def video_work(**changes):
    values = dict.fromkeys(VIDEO_COUNTS, 0)
    values.update(version=1, complete=True, fullDecode=True, audioStream=False, videoFrames=1, videoPixels=120,
                  processedFrames=1, processedPixels=120, convertedPixels=120, videoWidth=12, videoHeight=10,
                  videoBitDepth=8, videoFormat='rgb24', conversionFormat='rgb24', videoCodec='png', decoderVersion='fixture')
    values.update(changes)
    return values


def loaded(**changes):
    values = dict(shape=(1, 10, 12, 3), mask_shape=(1, 64, 64), pipeline_version=1, loader='pyav',
                  alpha=False, precision='torch.float32', device='cpu')
    values.update(changes)
    return values


def graph(kind='LoadImage'):
    return {'image': {'class_type': kind, 'inputs': {'image': '/PRIVATE/image.png'}},
            'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'weight'}},
            'encode': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['image', 0], 'vae': ['vae', 0]}}}


def store(tmp_path, kind='LoadImage'):
    now = [0.0]
    value = RunTimingStore(str(tmp_path / 'timing.sqlite'), host_id='host', gpu_uuid='GPU-fixture', now=lambda: now[0])
    item = (1, 'prompt', graph(kind), {}, ['encode'])
    with patch('app.run_timing.selected_file_identity', return_value='selected-weight'):
        value.event('accepted', item)
        value.event('started', item)
    return value, item, now


def stage(value, item, name='image'):
    record = value.page('0', 100, [item], [])['active']['records'][0]
    return next(stage for stage in record['stages'] if stage['id'] == name), record


def image(value, **changes):
    value.safe_stage('prompt', 'image', 'started')
    value.safe_metric('prompt', 'image', 'video_decode_work', video_work())
    value.safe_metric('prompt', 'image', 'image_loaded', loaded(**changes))


@pytest.mark.parametrize('kind', ['LoadImage', 'LoadImageOutput', 'LoadImageMask'])
def test_explicit_loader_declarations_exclude_paths_and_require_native_processing(kind):
    resource = {'hostId': 'host', 'gpuUuid': 'GPU-fixture', 'instanceId': 'boot', 'laneId': 'comfyui'}
    evidence, stages = describe(graph(kind), {'profileRevision': 'fixture'}, resource, 'now', lambda *_: 'weight', digest)
    current = next(stage for stage in stages if stage['id'] == 'image')
    assert current['kind'] == 'preparation' and current['workload']['profile'] == 'vision'
    assert current['workload']['resolved']['imageLoader.maskSelectionRequired']['value'] is (kind == 'LoadImageMask')
    assert 'reference_processing_unmeasured' in current['workload']['reasons']
    assert '/PRIVATE' not in json.dumps([evidence, stages])
    encode = next(stage for stage in stages if stage['id'] == 'encode')
    assert encode['workload']['components'][0]['features']['sourceStage']['value'] == 'image'
    assert 'image_encode_workload_unmeasured' in encode['workload']['reasons']


def test_complete_native_input_clears_only_its_own_gap_and_keeps_missing_duration_separate(tmp_path):
    value, item, _ = store(tmp_path)
    image(value)
    value.safe_metric('prompt', 'image', 'image_output_prepare', {'seconds': -1, 'succeeded': True})
    value.safe_stage('prompt', 'image', 'finished')
    current, record = stage(value, item)
    assert current['workload']['complete']
    assert 'image_preparation_measurement_unavailable' in current['reasons']
    assert 'image_preparation_measurement_unavailable' not in current['workload']['reasons']
    component = current['workload']['components'][0]
    assert component['features']['sourceStage']['value'] == 'image'
    assert component['features']['decode.videoCodec']['value'] == 'png'
    assert component['features']['output.maskHeight']['value'] == 64
    assert not record['workload']['evidence']['complete']
    assert 'image_encode_workload_unmeasured' in record['workload']['evidence']['reasons']
    assert '/PRIVATE' not in json.dumps(record)
    schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
    jsonschema.validate(value.page('0', 100, [item], []), schema['$defs']['TimingObservationPage'])


@pytest.mark.parametrize('change', [dict(version=2), dict(complete=False), dict(fullDecode=False), dict(processedPixels=99),
                                   dict(videoFrames='unknown'), dict(decodeErrors=1), dict(videoCodec=None)])
def test_partial_or_malformed_native_work_never_becomes_complete(tmp_path, change):
    value, item, _ = store(tmp_path)
    value.safe_stage('prompt', 'image', 'started')
    value.safe_metric('prompt', 'image', 'video_decode_work', video_work(**change))
    value.safe_metric('prompt', 'image', 'image_loaded', loaded())
    current, _ = stage(value, item)
    assert 'reference_processing_unmeasured' in current['workload']['reasons']


def test_shape_only_legacy_evidence_and_repeated_calls_keep_the_gap(tmp_path):
    value, item, _ = store(tmp_path)
    value.safe_stage('prompt', 'image', 'started')
    value.safe_metric('prompt', 'image', 'image_loaded', {'shape': (1, 10, 12, 3)})
    assert not stage(value, item)[0]['workload']['complete']
    value.safe_metric('prompt', 'image', 'video_decode_work', video_work())
    value.safe_metric('prompt', 'image', 'image_loaded', loaded())
    value.safe_stage('prompt', 'image', 'finished')
    assert not stage(value, item)[0]['workload']['complete']


def test_mask_selection_and_pillow_path_require_their_own_work(tmp_path):
    value, item, _ = store(tmp_path, 'LoadImageMask')
    value.safe_stage('prompt', 'image', 'started')
    empty = video_work(videoFrames=0, videoPixels=0, processedFrames=0, processedPixels=0, convertedPixels=0)
    value.safe_metric('prompt', 'image', 'video_decode_work', empty)
    value.safe_metric('prompt', 'image', 'image_pillow_work', dict(version=1, frames=1, decodedPixels=120, selectedFrames=1,
                      alphaFrames=0, sourceWidth=12, sourceHeight=10, format='WEBP', mode='RGB', consistent=True))
    value.safe_metric('prompt', 'image', 'image_loaded', loaded(loader='pillow'))
    assert not stage(value, item)[0]['workload']['complete']
    value.safe_metric('prompt', 'image', 'image_mask_selected', {'shape': (1, 10, 12), 'channel': 'red', 'mode': 'channel_clone'})
    current, _ = stage(value, item)
    assert current['workload']['complete']
    assert current['workload']['resolved']['imageMask.channel']['value'] == 'red'


@pytest.mark.parametrize('facts', [dict(shape=(1, 9, 12), channel='red', mode='channel_clone'),
                                  dict(shape=(1, 10, 12), channel='alpha', mode='channel_clone'),
                                  dict(shape=(1, 10, 12), channel='alpha', mode='alpha_mask')])
def test_mask_selection_must_match_actual_loaded_pixels_or_alpha_mask(tmp_path, facts):
    value, item, _ = store(tmp_path, 'LoadImageMask')
    image(value)
    value.safe_metric('prompt', 'image', 'image_mask_selected', facts)
    assert not stage(value, item)[0]['workload']['complete']


@pytest.mark.parametrize('event', ['video_decode_work', 'image_loaded', 'image_mask_selected'])
def test_repeated_image_work_immediately_invalidates_live_coverage_once(tmp_path, event):
    value, item, _ = store(tmp_path, 'LoadImageMask')
    image(value)
    selected = dict(shape=(1, 10, 12), channel='red', mode='channel_clone')
    value.safe_metric('prompt', 'image', 'image_mask_selected', selected)
    assert stage(value, item)[0]['workload']['complete']
    facts = {'video_decode_work': video_work(), 'image_loaded': loaded(), 'image_mask_selected': selected}[event]
    value.safe_metric('prompt', 'image', event, facts)
    assert not stage(value, item)[0]['workload']['complete']
    changes = value.db.total_changes
    value.safe_metric('prompt', 'image', event, facts)
    assert value.db.total_changes == changes


def test_malformed_later_output_immediately_invalidates_live_coverage(tmp_path):
    value, item, _ = store(tmp_path)
    image(value)
    assert stage(value, item)[0]['workload']['complete']
    value.safe_metric('prompt', 'image', 'image_loaded', {'shape': None})
    assert not stage(value, item)[0]['workload']['complete']
    changes = value.db.total_changes
    value.safe_metric('prompt', 'image', 'image_loaded', {'shape': None})
    assert value.db.total_changes == changes


@pytest.mark.parametrize('cached_event', ['cache', 'stage'])
def test_verified_reuse_skips_source_processing_without_manufacturing_measurements(tmp_path, cached_event):
    value, item, _ = store(tmp_path)
    if cached_event == 'cache':
        value.safe_cache('prompt', ['image', 'encode'])
    else:
        value.safe_stage('prompt', 'image', 'reused')
        value.safe_stage('prompt', 'encode', 'reused')
    current, _ = stage(value, item)
    assert current['reused'] is True and current['durationSeconds'] == 0
    assert current['workload']['complete'] and not current['workload']['measurements']
    assert current['workload']['resolved']['imageLoader.sourceReused']['value'] is True
    assert stage(value, item, 'encode')[0]['workload']['resolved']['imageEncode.sourceReused']['value'] is True


def encode(value, tiled=False, forwards=2, complete=True):
    value.safe_stage('prompt', 'encode', 'started')
    value.safe_inputs('prompt', 'encode', 'VAEEncode', {'pixels': SimpleNamespace(shape=(2, 10, 12, 3))})
    value.safe_metric('prompt', 'encode', 'vae_processed_input', {'shape': (2, 3, 8, 12), 'tiled': tiled, 'tile_count_complete': complete})
    plan = dict(shape=(2, 3, 8, 12), tiled=tiled, forwards=forwards, batch_size=1, complete=complete,
                precision='torch.float32', load_device='cpu', output_device='cpu')
    if tiled:
        plan.update(tile_x=8, tile_y=8, overlap=2)
    value.safe_metric('prompt', 'encode', 'vae_encode_plan', plan)
    return plan


@pytest.mark.parametrize('tiled', [False, True])
def test_native_2d_encode_plan_preserves_source_relationship_and_actual_forwards(tmp_path, tiled):
    value, item, _ = store(tmp_path)
    encode(value, tiled=tiled)
    current, _ = stage(value, item, 'encode')
    assert current['workload']['complete']
    assert current['workload']['resolved']['executionPrecision']['source'] == 'native_vae'
    assert current['workload']['resolved']['executionPrecision']['availableAt'] == 'live'
    assert current['workload']['components'][0]['features']['sourceStage']['value'] == 'image'
    before = value.db.total_changes
    for _ in range(2):
        value.safe_metric('prompt', 'encode', 'vae_forward_attempt', {'shape': (1, 3, 8, 12), 'tiled': tiled})
        value.safe_metric('prompt', 'encode', 'vae_forward_completed', {'tiled': tiled})
    assert value.db.total_changes == before
    value.safe_stage('prompt', 'encode', 'finished')
    assert stage(value, item, 'encode')[0]['workload']['complete']


@pytest.mark.parametrize('failure', ['fallback', 'repeated', 'missing_forward', 'internal_tiles'])
def test_encoder_fallback_unknown_internal_tiles_and_unmatched_forwards_reinstate_gaps(tmp_path, failure):
    value, item, _ = store(tmp_path)
    plan = encode(value, complete=failure != 'internal_tiles')
    if failure == 'fallback':
        value.safe_metric('prompt', 'encode', 'vae_tiled_fallback', {'tile_count_complete': True})
    elif failure == 'repeated':
        value.safe_metric('prompt', 'encode', 'vae_encode_plan', plan)
    value.safe_stage('prompt', 'encode', 'finished')
    current, _ = stage(value, item, 'encode')
    assert not current['workload']['complete']
    assert 'image_encode_forwards_unresolved' in current['workload']['reasons'] or 'image_encode_workload_unmeasured' in current['workload']['reasons']


@pytest.mark.parametrize('event', ['vae_encode_plan', 'vae_processed_input'])
def test_repeated_vae_input_or_plan_immediately_invalidates_live_coverage(tmp_path, event):
    value, item, _ = store(tmp_path)
    plan = encode(value)
    assert stage(value, item, 'encode')[0]['workload']['complete']
    facts = plan if event == 'vae_encode_plan' else dict(shape=(2, 3, 8, 12), tiled=False, tile_count_complete=True)
    value.safe_metric('prompt', 'encode', event, facts)
    assert not stage(value, item, 'encode')[0]['workload']['complete']
