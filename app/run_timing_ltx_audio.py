"""LTX audio preprocessing and decoder plans, without waveform/tensor reads."""
import math
from app.run_timing_audio import waveform_features, audio_latent_features

EVENTS = {'ltx_audio_resampled', 'ltx_audio_mel', 'ltx_audio_decode_plan', 'ltx_audio_latent_output',
          'ltx_audio_vocoder_input', 'ltx_audio_encode_forward', 'ltx_audio_decode_forward', 'ltx_audio_vocoder_forward'}


def ltx_audio_encode_complete(state):
    source, mel = state.get('ltx_audio_resampled_shape'), state.get('ltx_audio_mel_shape')
    return (state.get('ltx_audio_resampled_count') == state.get('ltx_audio_mel_count') == 1
            and source is not None and mel is not None and source[:2] == mel[:2]
            and mel[-1] == state['resolved'].get('ltxAudio.resampled.mel_bins'))


def ltx_audio_decode_complete(state):
    shape = state.get('ltx_audio_decode_plan_shape')
    return (state.get('ltx_audio_decode_plan_count') == 1 and shape is not None
            and all(state['resolved'].get('audioDecode.0.latent.axis.' + str(axis)) == size
                    for axis, size in enumerate(shape)))


def observe_ltx_audio_metrics(state, event, facts):
    if event not in EVENTS:
        return False
    resolved, measured = state['resolved'], state['measurements']
    sources = state.setdefault('sources', {})

    def reason(value):
        if value not in state['reasons']:
            state['reasons'].append(value)

    def measurement(key, value):
        measured[key] = measured.get(key, 0) + value
        sources[key] = 'native_ltx_audio'
        state.setdefault('measurement_availability', {})[key] = 'completion'

    if event.endswith('_forward'):
        prefix = {'ltx_audio_encode_forward': 'ltxAudio.encode', 'ltx_audio_decode_forward': 'ltxAudio.decode',
                  'ltx_audio_vocoder_forward': 'ltxAudio.vocoder'}[event]
        measurement(prefix + ('.completed' if facts.get('succeeded') is True else '.failed'), 1)
        if type(facts.get('seconds')) in (int, float) and math.isfinite(facts['seconds']) and facts['seconds'] >= 0:
            measurement(prefix + '.seconds', facts['seconds'])
        return True

    counter = event + '_count'
    state[counter] = state.get(counter, 0) + 1
    if state[counter] != 1:
        reason('multiple_ltx_audio_invocations')
        return True
    shape = facts.get('shape')
    rank = 3 if event == 'ltx_audio_resampled' else None if event == 'ltx_audio_vocoder_input' else 4
    valid = isinstance(shape, (tuple, list)) and len(shape) in ((3, 4) if rank is None else (rank,))
    valid = valid and all(type(size) is int and size > 0 for size in shape)
    if not valid:
        reason('ltx_audio_metadata_unavailable')
        return True
    values = {}
    if event == 'ltx_audio_resampled':
        waveform = waveform_features(shape, facts.get('sample_rate'))
        if waveform is None or not all(type(facts.get(key)) is int and facts[key] > 0
                                      for key in ('source_sample_rate', 'mel_bins', 'hop_length', 'n_fft')):
            reason('ltx_audio_metadata_unavailable')
            return True
        values.update(waveform)
        values.update({key: facts[key] for key in ('source_sample_rate', 'mel_bins', 'hop_length', 'n_fft')})
        prefix = 'ltxAudio.resampled.'
    elif event in ('ltx_audio_mel', 'ltx_audio_decode_plan'):
        if not all(isinstance(facts.get(key), str) and 0 < len(facts[key]) <= 128 for key in ('precision', 'device')):
            reason('ltx_audio_metadata_unavailable')
            return True
        values.update({'axis.' + str(axis): size for axis, size in enumerate(shape)})
        values['device'] = facts['device']
        if event == 'ltx_audio_decode_plan':
            target = facts.get('target_shape')
            if (audio_latent_features(shape, 'bctf') is None or not isinstance(target, (tuple, list)) or len(target) != 4
                    or not all(type(size) is int and size > 0 for size in target) or target[0] != shape[0] or target[1] not in (1, 2)):
                reason('ltx_audio_metadata_unavailable')
                return True
            values.update({'target.axis.' + str(axis): size for axis, size in enumerate(target)})
            state['ltx_audio_expected_vocoder_shape'] = ((target[0], target[3], target[2]) if target[1] == 1
                                                        else (target[0], target[1], target[3], target[2]))
        state['precision'] = facts['precision']
        state.setdefault('availability', {})['executionPrecision'] = 'live'
        sources['executionPrecision'] = 'native_ltx_audio'
        prefix = 'ltxAudio.mel.' if event == 'ltx_audio_mel' else 'ltxAudio.decode.'
    else:
        if event == 'ltx_audio_vocoder_input' and tuple(shape) != state.get('ltx_audio_expected_vocoder_shape'):
            reason('ltx_audio_vocoder_plan_mismatch')
        for axis, size in enumerate(shape):
            measurement(event + '.axis.' + str(axis), size)
        return True
    state[event + '_shape'] = tuple(shape)
    values = {prefix + key: value for key, value in values.items()}
    resolved.update(values)
    sources.update({key: 'native_ltx_audio' for key in values})
    state.setdefault('availability', {}).update({key: 'live' for key in values})
    marker = 'ltxAudio.encodePlanRequired' if event in ('ltx_audio_resampled', 'ltx_audio_mel') else 'ltxAudio.decodePlanRequired'
    resolved[marker] = True
    sources[marker] = 'native_ltx_audio'
    state.setdefault('availability', {})[marker] = 'live'
    if event in ('ltx_audio_resampled', 'ltx_audio_mel'):
        state.setdefault('reference_features', {}).setdefault('audio', {}).update({
            key: {'value': value, 'source': 'native_ltx_audio', 'availableAt': 'live'} for key, value in values.items()})
    return True
