"""Scalar CLIP vision reference work, scoped to its actual native boundary."""
import math

EVENTS = {'clip_vision_source', 'clip_vision_plan', 'clip_vision_output',
          'clip_vision_preprocess', 'clip_vision_forward', 'clip_vision_metadata_unavailable'}
MODEL_TYPES = {'clip_vision_model', 'siglip_vision_model', 'siglip2_vision_model', 'dinov2', 'dinov3'}


def clip_vision_plan_complete(state):
    source, processed = state.get('clip_vision_source_shape'), state.get('clip_vision_processed_shape')
    return (state.get('clip_vision_sources') == state.get('clip_vision_plans') == 1
            and source is not None and processed is not None and source[0] == processed[0]
            and state.get('clip_vision_source_crop') == state.get('clip_vision_plan_crop'))


def observe_clip_vision_metrics(state, event, facts):
    if event not in EVENTS:
        return False
    source = 'native_clip_vision'
    measured, resolved = state['measurements'], state['resolved']
    sources = state.setdefault('sources', {})

    def reason(value):
        if value not in state['reasons']:
            state['reasons'].append(value)

    def count(key, value=1):
        measured[key] = measured.get(key, 0) + value
        sources[key] = source

    def reference(values, available):
        state.setdefault('reference_features', {}).setdefault('image', {}).update({
            key: {'value': value, 'source': source, 'availableAt': available} for key, value in values.items()})

    if event == 'clip_vision_metadata_unavailable':
        reason('clip_vision_metadata_unresolved')
        return True
    if event in ('clip_vision_preprocess', 'clip_vision_forward'):
        prefix = 'visionPreprocess' if event == 'clip_vision_preprocess' else 'visionEncodeForward'
        count(prefix + ('Completed' if facts.get('succeeded') is True else 'Failed'))
        seconds = facts.get('seconds')
        if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
            count(prefix + 'Seconds', seconds)
        return True

    counter = {'clip_vision_source': 'clip_vision_sources', 'clip_vision_plan': 'clip_vision_plans',
               'clip_vision_output': 'clip_vision_outputs'}[event]
    state[counter] = state.get(counter, 0) + 1
    if state[counter] != 1:
        reason('multiple_clip_vision_invocations')
        return True
    shape = facts.get('shape')
    rank = 3 if event == 'clip_vision_output' else 4
    valid = isinstance(shape, (tuple, list)) and len(shape) == rank and all(type(size) is int and size > 0 for size in shape)
    if event == 'clip_vision_source':
        valid = valid and shape[-1] >= 3 and type(facts.get('crop')) is bool
    elif event == 'clip_vision_plan':
        valid = valid and shape[1] == 3 and type(facts.get('crop')) is bool
        valid = valid and isinstance(facts.get('model_type'), str) and facts['model_type'] in MODEL_TYPES
        valid = valid and facts.get('memory_strategy') in ('dynamic', 'standard')
        valid = valid and all(isinstance(facts.get(key), str) and 0 < len(facts[key]) <= 128
                              for key in ('precision', 'load_device', 'offload_device'))
    if not valid:
        reason('clip_vision_metadata_unresolved')
        return True
    if event == 'clip_vision_output':
        if shape[0] != (state.get('clip_vision_processed_shape') or (None,))[0]:
            reason('clip_vision_metadata_unresolved')
        count('visionEncodedBatch', shape[0])
        count('visionEncodedPositions', shape[0] * shape[1])
        count('visionEncodedWidth', shape[2])
        state.setdefault('measurement_availability', {}).update({key: 'completion' for key in
            ('visionEncodedBatch', 'visionEncodedPositions', 'visionEncodedWidth')})
        reference({'encodedBatch': shape[0], 'encodedPositions': shape[1], 'encodedWidth': shape[2]}, 'completion')
        return True
    prefix = 'visionEncode.source' if event == 'clip_vision_source' else 'visionEncode.processed'
    values = {prefix + '.axis.' + str(axis): size for axis, size in enumerate(shape)}
    values[prefix + '.crop'] = facts['crop']
    available = 'execution_start' if event == 'clip_vision_source' else 'live'
    if event == 'clip_vision_source':
        state['clip_vision_source_shape'], state['clip_vision_source_crop'] = tuple(shape), facts['crop']
    else:
        state['clip_vision_processed_shape'], state['clip_vision_plan_crop'] = tuple(shape), facts['crop']
        values.update({'visionEncode.' + key: facts[key] for key in ('model_type', 'load_device', 'offload_device', 'memory_strategy')})
        values['visionEncode.forwards'] = 1
        state['precision'] = facts['precision']
        state.setdefault('availability', {})['executionPrecision'] = available
        sources['executionPrecision'] = source
    resolved.update(values)
    sources.update({key: source for key in values})
    state.setdefault('availability', {}).update({key: available for key in values})
    reference(values, available)
    return True
