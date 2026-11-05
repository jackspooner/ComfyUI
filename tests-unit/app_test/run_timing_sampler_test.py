import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import torch

from app import run_timing
from comfy import execution_metrics, samplers
from comfy_extras.nodes_custom_sampler import SamplerEulerAncestral


@pytest.mark.parametrize('eta', [0., .5, 1.])
def test_native_ancestral_node_options_reach_start_facts_without_outcome_leakage(tmp_path, eta):
    graph = {
        'sampler': {'class_type': 'SamplerEulerAncestral', 'inputs': {'eta': eta, 's_noise': .5}},
        'sample': {'class_type': 'SamplerCustomAdvanced', 'inputs': {'sampler': ['sampler', 0]}},
    }
    store = run_timing.RunTimingStore(str(tmp_path / 'timing.sqlite'), now=lambda: 0.)
    item = (1, 'fixture', graph, {}, ['sample'])
    try:
        store.event('accepted', item)
        store.event('started', item)
        store.safe_stage('fixture', 'sample', 'started')
        guider = samplers.CFGGuider.__new__(samplers.CFGGuider)
        guider.model_patcher = SimpleNamespace(model_dtype=lambda: torch.float32, is_dynamic=lambda: False,
                                               load_device='cpu', offload_device='cpu')
        guider.cfg, guider.model_options, guider.original_conds = 1, {}, {'positive': [{}]}
        execution_metrics.set_observer(lambda event, facts: store.safe_metric('fixture', 'sample', event, facts))
        sampler = SamplerEulerAncestral.execute(eta, .5).result[0]
        samplers.sampling_metrics_callback(guider, [(1, 4, 2, 2)], sampler, torch.tensor([1., .5, 0.]), None)
        # This focused sampler fixture has no model execution or managed weights.
        execution_metrics.emit('managed_load_plan', operation=object(), models=0, force_patch_weights=False,
                               force_full_load=False, requested_bytes=0, minimum_bytes=0)
        page = store.page('0', 100, [], [])
        record = page['observations'][-1]
        stage = next(value for value in record['stages'] if value['id'] == 'sample')
        prepared = next(value for value in record['stages'] if value['id'] == 'sampler')
        assert prepared['kind'] == 'preparation'
        assert prepared['workload']['resolved']['eta']['value'] == eta
        assert 'unknown_node_adapter:SamplerEulerAncestral' not in record['workload']['evidence']['reasons']
        assert stage['workload']['complete'] is True
        assert stage['workload']['resolved']['modelEvaluations']['value'] == 2
        assert stage['workload']['resolved']['nativeSampler']['value'] == 'euler_ancestral'
        facts = stage['workload']['resolved']
        assert facts['samplerOption.eta']['value'] == eta
        assert facts['samplerOption.s_noise']['value'] == .5
        assert all(facts[key]['availableAt'] == 'execution_start' for key in facts if key.startswith('samplerOption.'))
        assert 'noiseEvaluations' not in stage['workload']['measurements']
        schema = json.loads((Path(__file__).resolve().parents[2] / 'app/run_timing.schema.json').read_text())
        jsonschema.validate(page, schema['$defs']['TimingObservationPage'])
    finally:
        execution_metrics.set_observer(None)
        store.db.close()
