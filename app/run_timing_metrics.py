"""Bind native scalar observations to the existing queue/context owner."""
from comfy import execution_metrics
from comfy_execution.utils import get_executing_context


def apply_metrics(record, stage, state, observed_at, completed=False):
    def fact(value, available, key):
        return {'value': value, 'availableAt': available, 'observedAt': observed_at,
                'source': state.get('sources', {}).get(key, 'native_sampler')}
    workload = stage.get('workload')
    evidence = record['workload'].get('evidence')
    if workload is None or evidence is None:
        return
    for key, value in state.get('resolved', {}).items():
        workload['resolved'][key] = fact(value, state.get('availability', {}).get(key, 'execution_start'), key)
        evidence['resolved'][stage['id'] + '.' + key] = workload['resolved'][key]
    for key, value in state.get('measurements', {}).items():
        workload['measurements'][key] = fact(value, 'completion' if completed else 'live', key)
    live = {key: {'value': value, 'availableAt': 'live', 'observedAt': observed_at, 'source': 'native_lora'}
            for key, value in state.get('live', {}).items()}
    if live:
        workload['resolved'].update(live)
        evidence['resolved'].update({stage['id'] + '.' + key: value for key, value in live.items()})
        order = next(index for index, item in enumerate(record['stages']) if item['id'] == stage['id'])
        for component in workload['components'] + [component for component in evidence['components'] if component['order'] == order]:
            if component['kind'] == 'lora':
                component['features'].update(live)
    if state.get('reference_features'):
        order = next(index for index, item in enumerate(record['stages']) if item['id'] == stage['id'])
        components = workload['components'] + [item for item in evidence['components'] if item['order'] == order]
        for component in components:
            if component['kind'] != 'reference':
                continue
            role = component['role'] if component['role'] == 'image_input' else component['role'].removeprefix(stage['id'] + ':')
            for key, value in state['reference_features'].get(role, {}).items():
                component['features'][key] = {**value, 'observedAt': observed_at}
    if state.get('control_features'):
        order = next(index for index, item in enumerate(record['stages']) if item['id'] == stage['id'])
        for prefix, values in state['control_features'].items():
            role = stage['id'] + ':' + prefix
            origin = next((item for item in record['stages'] if item['id'] == values['sourceStage']), None)
            original = next((item for item in origin['workload']['components'] if item['kind'] == 'controlnet'), None) if origin and origin.get('workload') else None
            features = {key: {'value': value, 'availableAt': 'live', 'observedAt': observed_at, 'source': 'native_controlnet'}
                        for key, value in values.items()}
            component = {'kind': 'controlnet', 'identity': original['identity'] if original else None,
                         'role': role, 'order': order, 'features': features}
            for target in (workload, evidence):
                target['components'] = [item for item in target['components'] if item['role'] != role] + [component]
    if state.get('precision'):
        record['workload']['precision'] = state['precision']
    reasons = set(stage['reasons'])
    if 'modelEvaluations' in state.get('resolved', {}) and 'cfgBranches' in state['resolved']:
        reasons.discard('model_evaluations_unmeasured')
    if all(key in state.get('resolved', {}) for key in ('latentWidth', 'latentHeight', 'latentFrames')):
        reasons.discard('latent_shape_unresolved')
    if state.get('text_plans') == 1 and completed and state.get('measurements', {}).get('prefillForwardsCompleted') == 1:
        generated = state['measurements'].get('generatedTokens')
        if type(generated) is int and state['measurements'].get('decodeForwardsCompleted', 0) == max(0, generated - 1):
            reasons.discard('text_loop_coverage_unresolved')
    reasons.update(state.get('reasons', []))
    if completed and state.get('measurements', {}).get('noiseEvaluations') != state.get('resolved', {}).get('modelEvaluations'):
        reasons.add('sampling_evaluation_count_unresolved')
    if completed and state.get('resolved', {}).get('cfgBranches') is not None:
        expected = state['resolved'].get('modelEvaluations')
        if expected is not None and state.get('measurements', {}).get('cfgBranchEvaluations') != expected * state['resolved']['cfgBranches']:
            reasons.add('cfg_behavior_unmeasured')
    stage['reasons'] = sorted(reasons)
    workload.update(complete=not reasons, reasons=stage['reasons'])
    remaining = set(evidence['reasons']) - {'model_evaluations_unmeasured', 'latent_shape_unresolved', 'text_loop_coverage_unresolved'}
    remaining.update(reason for item in record.get('stages', []) for reason in item['reasons'])
    evidence.update(complete=not remaining, reasons=sorted(remaining))
    if state.get('progress') and stage['kind'] in ('sampling', 'inference'):
        stage['progress'] = dict(state['progress'])


def attach_runtime_metrics(queue):
    def observe(event, facts):
        context = get_executing_context()
        if context is not None:
            node_id = execution_metrics.current_stage() or context.node_id
            with queue.mutex:
                if event == 'stage_started':
                    item = next((item for item in queue.currently_running.values() if item[1] == context.prompt_id), None)
                    if item is not None:
                        queue.timing.safe_resolve_started(context.prompt_id, item[2])
                if event == 'stage_inputs':
                    queue.timing.safe_inputs(context.prompt_id, node_id, facts['class_type'], facts['inputs'])
                elif event.startswith('stage_'):
                    action = {'stage_started': 'started', 'stage_finished': 'finished',
                              'stage_failed': 'failed', 'stage_reused': 'reused'}.get(event)
                    if action:
                        queue.timing.safe_stage(context.prompt_id, node_id, action)
                else:
                    queue.timing.safe_metric(context.prompt_id, node_id, event, facts)
    execution_metrics.set_observer(observe)
