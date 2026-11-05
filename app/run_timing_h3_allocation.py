"""Native H3 allocation facts; requested frames never stand in for the computed plan."""
import math

EVENTS = {'h3_av_temporal_plan', 'h3_av_allocation_plan', 'h3_av_allocate', 'h3_av_allocation_output'}


def h3_allocation_complete(state):
    temporal, planned = state.get('h3_temporal'), state.get('h3_allocation_shapes')
    return (state.get('h3_temporal_count') == state.get('h3_plan_count') == state.get('h3_output_count') == 1
            and temporal is not None and planned is not None and state.get('h3_output_shapes') == planned
            and temporal['video_latents'] == planned['video'][2] and temporal['audio_latents'] == planned['audio'][3])


def observe_h3_allocation_metrics(state, event, facts):
    probe_failed = event == 'tensor_shape_unavailable' and facts.get('operation') == 'h3_av_allocation_output'
    if event not in EVENTS and not probe_failed:
        return False

    def missing(reason):
        if reason not in state['reasons']:
            state['reasons'].append(reason)

    if probe_failed:
        missing('h3_allocation_metadata_unresolved')
        return True
    sources = state.setdefault('sources', {})
    def retain(key, value, target='resolved', available='execution_start'):
        state[target][key] = value
        sources[key] = 'native_h3_allocation'
        state.setdefault('measurement_availability' if target == 'measurements' else 'availability', {})[key] = available

    if event == 'h3_av_allocate':
        suffix = 'completed' if facts.get('succeeded') is True else 'failed'
        key = 'jointAllocation.' + suffix
        retain(key, state['measurements'].get(key, 0) + 1, 'measurements', 'completion')
        if (suffix == 'failed' or state['measurements'].get('jointAllocation.completed', 0)
                + state['measurements'].get('jointAllocation.failed', 0) != 1):
            missing('h3_allocation_execution_unresolved')
        seconds = facts.get('seconds')
        if type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0:
            retain('jointAllocation.seconds', state['measurements'].get('jointAllocation.seconds', 0) + seconds,
                   'measurements', 'completion')
        return True

    counter = {'h3_av_temporal_plan': 'h3_temporal_count', 'h3_av_allocation_plan': 'h3_plan_count',
               'h3_av_allocation_output': 'h3_output_count'}[event]
    state[counter] = state.get(counter, 0) + 1
    if state[counter] != 1:
        missing('multiple_h3_allocation_invocations')
        return True
    if event == 'h3_av_temporal_plan':
        keys = ('frames', 'video_latents', 'audio_latents', 'frame_rate', 'audio_latent_rate')
        if not all(type(facts.get(key)) is int and facts[key] > 0 for key in keys):
            missing('h3_allocation_metadata_unresolved')
            return True
        state['h3_temporal'] = {key: facts[key] for key in keys}
        for key in keys:
            retain('jointAllocation.' + key, facts[key])
        return True

    output = event == 'h3_av_allocation_output'
    shapes = facts.get('shapes') if output else {'video': facts.get('video_shape'), 'audio': facts.get('audio_shape')}
    if (not isinstance(shapes, dict) or shapes.keys() != {'video', 'audio'}
            or any(not isinstance(shapes[role], (tuple, list)) or len(shapes[role]) != rank
                   or any(type(size) is not int or size <= 0 for size in shapes[role])
                   for role, rank in (('video', 5), ('audio', 4)))):
        missing('h3_allocation_metadata_unresolved')
        return True
    shapes = {role: tuple(shape) for role, shape in shapes.items()}
    if shapes['video'][1] != 24 or shapes['audio'][1:3] != (32, 2) or shapes['video'][0] != shapes['audio'][0]:
        missing('h3_allocation_metadata_unresolved')
        return True
    if not output:
        state['h3_allocation_shapes'] = shapes
        retain('jointAllocation.kind', 'h3_zeros')
        for role, shape in shapes.items():
            for axis, size in enumerate(shape):
                retain('jointAllocation.plan.' + role + '.axis.' + str(axis), size)
        return True

    precisions, devices = facts.get('precisions'), facts.get('devices')
    if any(not isinstance(values, dict) or values.keys() != shapes.keys()
           or any(not isinstance(value, str) or not 0 < len(value) <= 128 for value in values.values())
           for values in (precisions, devices)):
        missing('h3_allocation_metadata_unresolved')
        return True
    state['h3_output_shapes'] = shapes
    if shapes != state.get('h3_allocation_shapes'):
        missing('h3_allocation_output_mismatch')
    for role, shape in shapes.items():
        for axis, size in enumerate(shape):
            retain('jointAllocation.output.' + role + '.axis.' + str(axis), size, 'measurements', 'completion')
        for suffix, value in (('precision', precisions[role]), ('device', devices[role])):
            retain('jointAllocation.' + role + '.' + suffix, value, available='live')
            retain('jointAllocation.output.' + role + '.' + suffix, value, 'measurements', 'completion')
    state['precision'] = precisions['video']
    sources['executionPrecision'] = 'native_h3_allocation'
    state.setdefault('availability', {})['executionPrecision'] = 'live'
    return True
