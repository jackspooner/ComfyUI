"""Observe the actual native H3 packed token plan, never reference contents."""

ROLES = ('text', 'cond', 'cond_audio', 'ref_img', 'ref_audio', 'audio', 'video')


def h3_layout_complete(state):
    signature = state.get('h3_layout_signature')
    values = state['resolved']
    height, width = values.get('latentHeight'), values.get('latentWidth')
    return (signature is not None and height is not None and width is not None
            and signature[1:] == (values.get('latentFrames'), (height + 1) // 2 * 2, (width + 1) // 2 * 2,
                                   values.get('jointAV.audio.audioLatentLength'))
            and values.get('batch') == 1 and values.get('latentChannels') == 24
            and values.get('jointAV.audio.latentChannels') == 32
            and values.get('jointAV.audio.audioLatentChannels') == 2
            and not any(reason.startswith('h3_packed_layout_') for reason in state['reasons']))


def observe_h3_layout_metrics(state, event, facts):
    if event not in ('h3_packed_layout', 'h3_packed_layout_required'):
        return False

    def missing(reason):
        if reason not in state['reasons']:
            state['reasons'].append(reason)

    def retain(key, value, target='resolved'):
        state[target][key] = value
        state.setdefault('sources', {})[key] = 'native_h3_packed_layout'
        state.setdefault('measurement_availability' if target == 'measurements' else 'availability', {})[key] = 'execution_start' if event == 'h3_packed_layout_required' else 'live'

    if event == 'h3_packed_layout_required':
        retain('h3Layout.required', True)
        return True
    builds = state['measurements'].get('h3Layout.builds', 0) + 1
    retain('h3Layout.builds', builds, 'measurements')
    signature, segments = facts.get('signature'), facts.get('segments')
    if (facts.get('native') is not True or not isinstance(signature, (tuple, list)) or len(signature) != 5
            or any(type(size) is not int or size < (0 if axis == 0 else 1) for axis, size in enumerate(signature))
            or not isinstance(segments, (tuple, list)) or not 3 <= len(segments) <= 128
            or facts.get('segment_count') != len(segments)
            or any(not isinstance(segment, (tuple, list)) or len(segment) != 2 or segment[0] not in ROLES
                   or type(segment[1]) is not int or segment[1] < (0 if segment[0] == 'text' else 1) for segment in segments)):
        missing('h3_packed_layout_metadata_unresolved')
        return True
    signature, segments = tuple(signature), tuple(tuple(segment) for segment in segments)
    text, video_t, height, width, audio_t = signature
    if (segments[0] != ('text', text) or segments[-2:] != (('audio', audio_t * 2), ('video', video_t * ((height + 1) // 2) * ((width + 1) // 2)))
            or any(role not in ('cond', 'cond_audio', 'ref_img', 'ref_audio') for role, _ in segments[1:-2])
            or type(facts.get('tokens')) is not int or facts['tokens'] != sum(size for _, size in segments)):
        missing('h3_packed_layout_metadata_unresolved')
        return True
    if state.get('h3_layout_signature') is not None:
        if signature != state['h3_layout_signature'] or segments != state['h3_layout_segments']:
            missing('h3_packed_layout_variants_unresolved')
        return True
    state.update(h3_layout_signature=signature, h3_layout_segments=segments)
    retain('h3Layout.tokens', facts['tokens'])
    retain('h3Layout.segments', len(segments))
    for role in ROLES:
        retain('h3Layout.' + role + '.tokens', sum(size for kind, size in segments if kind == role))
        retain('h3Layout.' + role + '.segments', sum(kind == role for kind, _ in segments))
    for axis, size in enumerate(signature):
        retain('h3Layout.signature.' + str(axis), size)
    for index, (role, size) in enumerate(segments):
        retain('h3Layout.segment.' + str(index) + '.role', role)
        retain('h3Layout.segment.' + str(index) + '.tokens', size)
    return True
