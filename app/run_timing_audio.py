"""Native audio workload facts from metadata only; never read waveform values."""
import math


def waveform_features(shape, sample_rate):
    if (not isinstance(shape, (tuple, list)) or len(shape) != 3
            or not all(type(value) is int and value > 0 for value in shape)
            or type(sample_rate) is not int or sample_rate <= 0):
        return None
    batch, channels, samples = shape
    duration = samples / sample_rate
    if not math.isfinite(duration):
        return None
    return {'batch': batch, 'channels': channels, 'samples': samples,
            'sampleRate': sample_rate, 'durationSeconds': duration}


def observe_audio_metrics(state, event, facts):
    events = {'audio_loaded': ('audioSource', 'live'),
              'audio_vae_encode_input': ('audioProcessed', 'live'),
              'audio_vae_decode_input': ('audioDecode', 'execution_start'),
              'audio_decoded': ('audioOutput', 'completion')}
    if event not in events:
        return False
    prefix, availability = events[event]
    reasons = state['reasons']
    def reason(value):
        if value not in reasons:
            reasons.append(value)
    if event == 'audio_vae_decode_input':
        shape = facts.get('shape')
        if (not isinstance(shape, (tuple, list)) or len(shape) not in (3, 4)
                or not all(type(value) is int and value > 0 for value in shape)):
            reason('audio_latent_shape_unavailable')
            return True
        values = {'latent.axis.' + str(axis): size for axis, size in enumerate(shape)}
        for key in ('tiled', 'normalize'):
            if type(facts.get(key)) is bool:
                values[key] = facts[key]
            else:
                reason('audio_decode_options_unavailable')
        if facts.get('tiled') is True:
            reason('audio_tile_work_unmeasured')
            for source, target in (('tile_size', 'tileSize'), ('overlap', 'overlap')):
                if type(facts.get(source)) is int and facts[source] >= 0:
                    values[target] = facts[source]
    else:
        values = waveform_features(facts.get('shape'), facts.get('sample_rate'))
        if values is None:
            reason('audio_output_metadata_unavailable' if event == 'audio_decoded' else 'audio_waveform_metadata_unavailable')
            return True
        if event == 'audio_vae_encode_input':
            source_rate = facts.get('source_sample_rate')
            if type(source_rate) is int and source_rate > 0:
                values['sourceSampleRate'] = source_rate
            else:
                reason('audio_source_rate_unavailable')
    count_key = prefix + '_count'
    index = state.get(count_key, 0)
    state[count_key] = index + 1
    if index:
        reason('multiple_audio_invocations')
    if index >= 16:
        reason('audio_invocation_limit')
        return True
    # Output dimensions are outcomes, even when the surrounding node is still
    # running. Never promote them into resolved input/comparison features.
    target = state['measurements'] if availability == 'completion' else state['resolved']
    for key, value in values.items():
        name = prefix + '.' + str(index) + '.' + key
        target[name] = value
        state.setdefault('sources', {})[name] = 'native_audio'
        if availability == 'completion':
            state.setdefault('measurement_availability', {})[name] = 'completion'
        else:
            state.setdefault('availability', {})[name] = 'live' if index else availability
    if event == 'audio_loaded' and index == 0:
        state.setdefault('reference_features', {})['audio_input'] = {
            key: {'value': value, 'source': 'native_audio', 'availableAt': 'live'} for key, value in values.items()}
    return True
