"""Reduce native control events without persisting process-local handles."""

CONTROL_EVENTS = {'control_interval_skipped', 'control_hint_prepared', 'control_forward_attempt',
                  'control_applied', 'control_features_encoded', 'control_features_reused'}


def observe_control_metrics(state, event, facts, origin):
    if event not in CONTROL_EVENTS:
        return False
    binding = facts.get('binding')
    if type(binding) is not object:
        return True
    controls = state.setdefault('controls', {})
    control = controls.setdefault(binding, {'index': len(controls)})
    prefix = 'control.' + str(control['index']) + '.'
    features = state.setdefault('control_features', {}).setdefault(prefix, {})
    features.update(sourceStage=origin, adapter=facts.get('adapter'))
    reasons = state['reasons']

    def reason(value):
        if value not in reasons:
            reasons.append(value)

    def count(key):
        name = prefix + key
        measurements = state['measurements']
        measurements[name] = measurements.get(name, 0) + 1
        state.setdefault('sources', {})[name] = 'native_controlnet'

    if origin is None:
        reason('control_application_binding_unresolved')
    shape = facts.get('shape')
    if isinstance(shape, tuple) and 1 <= len(shape) <= 8 and all(type(size) is int and size > 0 for size in shape):
        previous = control.setdefault('hint_shape', shape)
        if previous != shape:
            reason('control_hint_shape_varied')
        else:
            features.update({'hint.axis.' + str(axis): size for axis, size in enumerate(shape)})
    names = {'control_hint_prepared': 'hintPreparations', 'control_forward_attempt': 'forwardAttempts',
             'control_applied': 'activeCalls', 'control_features_encoded': 'featureEncodes',
             'control_features_reused': 'featureReuseHits'}
    if event in names:
        count(names[event])
    if event in ('control_applied', 'control_interval_skipped'):
        evaluation = facts.get('evaluation')
        if evaluation is None:
            reason('control_evaluation_correlation_unavailable')
        else:
            key = 'activeEvaluations' if event == 'control_applied' else 'skippedEvaluations'
            # CFG batches can call the same control more than once per evaluation.
            if control.get(key) is not evaluation:
                count(key)
                control[key] = evaluation
            if event == 'control_interval_skipped':
                name = prefix + 'activeEvaluations'
                state['measurements'].setdefault(name, 0)
                state.setdefault('sources', {})[name] = 'native_controlnet'
    return True
