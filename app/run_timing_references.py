"""Scalar reference-boundary observations; no tensor reads, I/O or prediction."""
import math


SHAPE_EVENTS = {'image_loaded', 'vae_processed_input', 'vae_forward_attempt', 'conditioning_encoded',
                'vision_forward', 'vision_feature_reused'}
REFERENCE_EVENTS = SHAPE_EVENTS | {'vae_forward_completed', 'vae_tiled_fallback', 'tensor_shape_unavailable'}


def observe_reference_metrics(state, event, facts):
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
        reason('reference_shape_unavailable')
        return True
    if event == 'vae_tiled_fallback':
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
        if event == 'image_loaded':
            if len(shape) != 4:
                reason('reference_shape_unavailable')
                return True
            values = dict(batch=shape[0], height=shape[1], width=shape[2], channels=shape[3])
            reference('image_input', values, 'native_image_loader', 'live')
        elif event == 'vae_processed_input':
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
