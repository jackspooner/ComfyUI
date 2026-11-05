"""Bind native scalar observations to the existing queue/context owner."""
from comfy import execution_metrics
from comfy_execution.utils import get_executing_context


def apply_metrics(record, stage, state, observed_at, completed=False):
    def fact(value, available):
        return {'value': value, 'availableAt': available, 'observedAt': observed_at, 'source': 'native_sampler'}
    workload = stage.get('workload')
    evidence = record['workload'].get('evidence')
    if workload is None or evidence is None:
        return
    for key, value in state.get('resolved', {}).items():
        workload['resolved'][key] = fact(value, 'execution_start')
        evidence['resolved'][stage['id'] + '.' + key] = workload['resolved'][key]
    for key, value in state.get('measurements', {}).items():
        workload['measurements'][key] = fact(value, 'completion' if completed else 'live')
    if state.get('precision'):
        record['workload']['precision'] = state['precision']
    reasons = set(stage['reasons'])
    if 'modelEvaluations' in state.get('resolved', {}) and 'cfgBranches' in state['resolved']:
        reasons.discard('model_evaluations_unmeasured')
    if all(key in state.get('resolved', {}) for key in ('latentWidth', 'latentHeight', 'latentFrames')):
        reasons.discard('latent_shape_unresolved')
    reasons.update(state.get('reasons', []))
    if completed and state.get('measurements', {}).get('noiseEvaluations') != state.get('resolved', {}).get('modelEvaluations'):
        reasons.add('sampling_evaluation_count_unresolved')
    stage['reasons'] = sorted(reasons)
    workload.update(complete=not reasons, reasons=stage['reasons'])
    remaining = set(evidence['reasons']) - {'model_evaluations_unmeasured', 'latent_shape_unresolved'}
    remaining.update(reason for item in record.get('stages', []) for reason in item['reasons'])
    evidence.update(complete=not remaining, reasons=sorted(remaining))
    if state.get('progress') and stage['kind'] == 'sampling':
        stage['progress'] = dict(state['progress'])


def attach_runtime_metrics(queue):
    def observe(event, facts):
        context = get_executing_context()
        if context is not None:
            with queue.mutex:
                queue.timing.safe_metric(context.prompt_id, context.node_id, event, facts)
    execution_metrics.set_observer(observe)
