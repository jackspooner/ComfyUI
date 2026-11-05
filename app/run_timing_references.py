"""Scalar reference-boundary observations; no tensor reads, I/O or prediction."""
import math
from app.run_timing_image import observe_image_metrics


SHAPE_EVENTS = {'vae_processed_input', 'vae_forward_attempt', 'conditioning_encoded',
                'vision_forward', 'vision_feature_reused'}
REFERENCE_EVENTS = SHAPE_EVENTS | {'vae_forward_completed', 'vae_encode_plan', 'vae_tiled_fallback', 'vae_decode_tiled_fallback', 'tensor_shape_unavailable'}


def observe_reference_metrics(state, event, facts):
    if observe_image_metrics(state, event, facts):
        return True
    if event not in REFERENCE_EVENTS:
        return False
    measurements = state['measurements']
    sources = state.setdefault('sources', {})

    def count(key, value=1, source='native_vae'):
        measurements[key] = measurements.get(key, 0) + value
        sources[key] = source

    def reason(value):
        if value not in state['reasons']:
            state['reasons'].append(value)

    def reference(role, values, source, available):
        target = state.setdefault('reference_features', {}).setdefault(role, {})
        target.update({key: {'value': value, 'source': source, 'availableAt': available} for key, value in values.items()})

    if event == 'tensor_shape_unavailable':
        reason('audio_output_metadata_unavailable' if facts.get('operation') == 'audio_decoded' else 'reference_shape_unavailable')
        return True
    if event == 'vae_encode_plan':
        state['image_encode_plans'] = state.get('image_encode_plans', 0) + 1
        shape = facts.get('shape')
        valid = isinstance(shape, (tuple, list)) and len(shape) == 4 and all(type(value) is int and value > 0 for value in shape)
        valid = valid and all(type(facts.get(key)) is int and facts[key] > 0 for key in ('forwards', 'batch_size'))
        valid = valid and type(facts.get('tiled')) is bool and all(isinstance(facts.get(key), str) and 0 < len(facts[key]) <= 128
                                                                for key in ('precision', 'load_device', 'output_device'))
        if facts.get('tiled') is True:
            valid = valid and all(type(facts.get(key)) is int and facts[key] > 0 for key in ('tile_x', 'tile_y'))
            valid = valid and type(facts.get('overlap')) is int and facts['overlap'] >= 0
        state['image_encode_plan_complete'] = valid and facts.get('complete') is True and state['image_encode_plans'] == 1
        if valid and state['image_encode_plans'] == 1:
            values = {'vaeEncode.plan.axis.' + str(axis): size for axis, size in enumerate(shape)}
            values.update({'vaeEncode.plan.' + key: value for key, value in facts.items()
                           if key in ('forwards', 'batch_size', 'tiled', 'tile_x', 'tile_y', 'overlap', 'precision', 'load_device', 'output_device')})
            state['resolved'].update(values)
            state.setdefault('availability', {}).update({key: 'live' for key in values})
            sources.update({key: 'native_vae' for key in values})
            reference('pixels', values, 'native_vae', 'live')
            state['precision'] = facts['precision']
            sources['executionPrecision'] = 'native_vae'
            state['availability']['executionPrecision'] = 'live'
    elif event == 'vae_decode_tiled_fallback':
        count('vaeDecodeTiledFallbacks')
        key = 'vaeDecode.tiledFallback'
        state['resolved'][key] = True
        sources[key] = 'native_vae'
        state.setdefault('availability', {})[key] = 'live'
        reason('vae_decode_tile_work_unmeasured')
    elif event == 'vae_tiled_fallback':
        count('vaeTiledFallbacks')
        index = state.get('vae_inputs', 0) - 1
        if index >= 0:
            key = 'vaeEncode.' + str(index) + '.tiled'
            state['resolved'][key] = True
            sources[key] = 'native_vae'
            state.setdefault('availability', {})[key] = 'live'
            reference('pixels', {key: True}, 'native_vae', 'live')
    elif event == 'vae_forward_completed':
        count('vaeForwardCompleted')
        if facts.get('tiled') is True:
            count('vaeTileForwardCompleted')
    else:
        shape = facts.get('shape')
        if not isinstance(shape, (tuple, list)) or not 1 <= len(shape) <= 8 or not all(type(value) is int and value > 0 for value in shape):
            reason('reference_shape_unavailable')
            return True
        if event == 'vae_processed_input':
            index = state.get('vae_inputs', 0)
            state['vae_inputs'] = index + 1
            values = {'vaeEncode.' + str(index) + '.axis.' + str(axis): size for axis, size in enumerate(shape)}
            values['vaeEncode.' + str(index) + '.tiled'] = facts.get('tiled') is True
            state['resolved'].update(values)
            sources.update({key: 'native_vae' for key in values})
            reference('pixels', values, 'native_vae', 'execution_start')
        elif event == 'vae_forward_attempt':
            count('vaeForwardAttempts')
            count('vaeForwardInputElements', math.prod(shape))
            if facts.get('tiled') is True:
                count('vaeTileForwardAttempts')
        elif event == 'conditioning_encoded':
            count('conditioningEncodeCalls', source='native_text_encoder')
            count('conditioningElements', math.prod(shape), 'native_text_encoder')
            if len(shape) == 3:
                count('conditioningEncodedPositions', shape[0] * shape[1], 'native_text_encoder')
        elif event == 'vision_forward':
            count('visionForwardAttempts', source='native_vision_encoder')
            count('visionForwardInputElements', math.prod(shape), 'native_vision_encoder')
        elif event == 'vision_feature_reused':
            count('visionFeatureReuseHits', source='native_vision_encoder')
    if event in ('vae_processed_input', 'vae_tiled_fallback'):
        measurements['vaeTileCountingComplete'] = measurements.get('vaeTileCountingComplete', True) and facts.get('tile_count_complete') is True
        sources['vaeTileCountingComplete'] = 'native_vae'
    return True
