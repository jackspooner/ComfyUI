"""Native audio workload facts from metadata only; never read waveform values."""
import math


def audio_latent_features(shape, layout=None):
    if (not isinstance(shape, (tuple, list)) or len(shape) not in (3, 4)
            or not all(type(value) is int and value > 0 for value in shape)):
        return None
    if len(shape) == 3 and layout in (None, 'bct'):
        values = {'batch': shape[0], 'latentChannels': shape[1], 'audioLatentLength': shape[2], 'audioLatentChannels': 1}
        if layout is not None:
            values['audioLatentLayout'] = layout
        return values
    if len(shape) == 4 and layout == 'bctf':
        return {'batch': shape[0], 'latentChannels': shape[1], 'audioLatentLength': shape[2],
                'audioLatentFrequencyBins': shape[3], 'audioLatentLayout': layout}
    if len(shape) == 4 and layout == 'bcat' and shape[2] == 2:
        return {'batch': shape[0], 'latentChannels': shape[1], 'audioLatentChannels': shape[2],
                'audioLatentLength': shape[3], 'audioLatentLayout': layout}
    return None


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
              'audio_latent_prepared': ('audioPrepared', 'live'),
              'audio_decoded': ('audioOutput', 'completion')}
    if event not in events:
        return False
    prefix, availability = events[event]
    reasons = state['reasons']
    def reason(value):
        if value not in reasons:
            reasons.append(value)
    if event == 'audio_latent_prepared':
        values = audio_latent_features(facts.get('shape'), facts.get('layout'))
        if values is None:
            reason('audio_latent_layout_unresolved')
            return True
        precision, device = facts.get('precision'), facts.get('device')
        if not all(isinstance(value, str) and 0 < len(value) <= 128 for value in (precision, device)):
            reason('audio_latent_metadata_unavailable')
            return True
        values['device'] = device
        state['precision'] = precision
        state.setdefault('sources', {})['executionPrecision'] = 'native_audio'
        state.setdefault('availability', {})['executionPrecision'] = 'live'
    elif event == 'audio_vae_decode_input':
        shape = facts.get('shape')
        if (not isinstance(shape, (tuple, list)) or len(shape) not in (3, 4)
                or not all(type(value) is int and value > 0 for value in shape)):
            reason('audio_latent_shape_unavailable')
            return True
        values = {'latent.axis.' + str(axis): size for axis, size in enumerate(shape)}
        if facts.get('layout') is not None:
            declared = audio_latent_features(shape, facts['layout'])
            if declared is None:
                reason('audio_latent_layout_unresolved')
            else:
                values.update(declared)
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
    if event == 'audio_vae_encode_input' and index == 0:
        state.setdefault('reference_features', {}).setdefault('audio', {}).update({
            'processed.' + key: {'value': value, 'source': 'native_audio', 'availableAt': 'live'} for key, value in values.items()})
    return True
