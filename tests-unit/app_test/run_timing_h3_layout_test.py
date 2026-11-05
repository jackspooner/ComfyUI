import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import torch

from app import run_timing
from app.run_timing_h3_layout import h3_layout_complete, observe_h3_layout_metrics
from comfy import execution_metrics, latent_formats, samplers
from comfy.ldm.minimax.model import PackedLayout


def state():
    return {'resolved': {'batch': 1, 'latentChannels': 24, 'latentFrames': 3, 'latentHeight': 3, 'latentWidth': 5,
                         'jointAV.audio.latentChannels': 32, 'jointAV.audio.audioLatentChannels': 2,
                         'jointAV.audio.audioLatentLength': 5}, 'measurements': {}, 'reasons': []}


def facts():
    return {'native': True, 'signature': (4, 3, 4, 6, 5), 'tokens': 44,
            'segment_count': 4, 'segments': (('text', 4), ('ref_img', 12), ('audio', 10), ('video', 18))}


def test_layout_requires_native_plan_and_correlates_padded_target_shape():
    value = state()
    assert not h3_layout_complete(value)
    assert observe_h3_layout_metrics(value, 'h3_packed_layout', facts())
    assert h3_layout_complete(value)
    assert value['resolved']['h3Layout.ref_img.tokens'] == 12
    assert value['availability']['h3Layout.tokens'] == 'live'
    assert value['measurement_availability']['h3Layout.builds'] == 'live'
    observe_h3_layout_metrics(value, 'h3_packed_layout', facts())
    assert h3_layout_complete(value) and value['measurements']['h3Layout.builds'] == 2
    for key, changed in [('batch', 2), ('latentFrames', 4), ('latentWidth', 7), ('jointAV.audio.audioLatentLength', 6)]:
        variant = copy.deepcopy(value)
        variant['resolved'][key] = changed
        assert not h3_layout_complete(variant)
    changed = facts()
    changed.update(signature=(5, 3, 4, 6, 5), tokens=45, segments=(('text', 5), ('ref_img', 12), ('audio', 10), ('video', 18)))
    observe_h3_layout_metrics(value, 'h3_packed_layout', changed)
    assert not h3_layout_complete(value) and 'h3_packed_layout_variants_unresolved' in value['reasons']


@pytest.mark.parametrize('change', [
    {'native': False}, {'signature': (4, 3, 4, 6)}, {'signature': (4, 3, 4, True, 5)},
    {'tokens': 45}, {'segment_count': 129}, {'segments': (('text', 4), ('audio', 10), ('custom', 18))},
    {'segments': (('text', 4), ('audio', 11), ('video', 18)), 'segment_count': 3, 'tokens': 33},
])
def test_unknown_truncated_and_inconsistent_layouts_remain_incomplete(change):
    value = state()
    observe_h3_layout_metrics(value, 'h3_packed_layout', {**facts(), **change})
    assert not h3_layout_complete(value)
    assert value['reasons'] == ['h3_packed_layout_metadata_unresolved']


def test_native_h3_plan_never_publishes_complete_work_before_packing_and_bounds_snapshots(tmp_path):
    store = run_timing.RunTimingStore(str(tmp_path / 'timing.sqlite'), now=lambda: 0.)
    graph = {'sample': {'class_type': 'SamplerCustomAdvanced', 'inputs': {}}}
    item = (1, 'fixture', graph, {}, ['sample'])
    try:
        store.event('accepted', item)
        store.event('started', item)
        store.safe_stage('fixture', 'sample', 'started')
        guider = samplers.CFGGuider.__new__(samplers.CFGGuider)
        guider.model_patcher = SimpleNamespace(model=SimpleNamespace(latent_format=latent_formats.MiniMaxH3AV()),
            model_dtype=lambda: torch.float32, is_dynamic=lambda: False, load_device='cpu', offload_device='cpu')
        guider.cfg, guider.model_options, guider.original_conds = 1, {}, {'positive': [{}]}
        execution_metrics.set_observer(lambda event, facts: store.safe_metric('fixture', 'sample', event, facts))
        samplers.sampling_metrics_callback(guider, [(1, 24, 3, 3, 5), (1, 32, 2, 5)],
            samplers.ksampler('er_sde'), torch.zeros(3), None)
        def current():
            return store.page('0', 100, [], [])['observations'][-1]
        pending = current()
        assert not pending['stages'][0]['workload']['complete']
        assert 'h3_packed_layout_unmeasured' in pending['stages'][0]['reasons']
        assert pending['stages'][0]['workload']['resolved']['h3Layout.required']['availableAt'] == 'execution_start'
        PackedLayout(4, 3, 4, 6, 5, refs=[{'kind': 'image', 'latent_h': 4, 'latent_w': 12}])
        complete = current()
        assert complete['stages'][0]['workload']['complete']
        assert complete['stages'][0]['workload']['resolved']['h3Layout.tokens']['value'] == 44
        assert complete['stages'][0]['workload']['resolved']['h3Layout.tokens']['availableAt'] == 'live'
        assert not pending['stages'][0]['workload']['complete']  # Saved pre-layout evidence is immutable.
        writes = store.db.total_changes
        for _ in range(10):
            PackedLayout(4, 3, 4, 6, 5, refs=[{'kind': 'image', 'latent_h': 4, 'latent_w': 12}])
        assert store.db.total_changes == writes
        PackedLayout(4, 3, 4, 6, 5)
        assert not current()['stages'][0]['workload']['complete']
        assert store.db.total_changes > writes
        schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
        jsonschema.validate(store.page('0', 100, [], []), schema['$defs']['TimingObservationPage'])
    finally:
        execution_metrics.set_observer(None)
        store.db.close()
