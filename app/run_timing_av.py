"""Native AV preparation work. Shapes are scalar evidence, never tensor values."""
import math

PLAN_EVENTS = {'av_concat_plan': 'concat', 'av_separate_plan': 'separate'}
OUTPUT_EVENTS = {'av_concat_output': 'concat', 'av_separate_output': 'separate'}


def av_preparation_complete(state, operation):
    return (state.get('av_plan_count') == 1 and state.get('av_operation') == operation
            and state.get('av_expected_output') is not None)


def av_preparation_execution_complete(state):
    return (state.get('av_output_count') == 1 and state.get('av_actual_output') == state.get('av_expected_output')
            and (state.get('av_input_streams') != 2 or state.get('av_operation') != 'concat'
                 or state['measurements'].get('jointPreparation.fit.completed') == 1
                 and not state['measurements'].get('jointPreparation.fit.failed', 0)))


def observe_av_metrics(state, event, facts):
    if event == 'tensor_shape_unavailable' and facts.get('operation') in PLAN_EVENTS.keys() | OUTPUT_EVENTS.keys():
        reason = 'joint_preparation_metadata_unresolved'
    elif event in PLAN_EVENTS or event in OUTPUT_EVENTS or event == 'av_concat_fit':
        reason = None
    else:
        return False

    def missing(value):
        if value not in state['reasons']:
            state['reasons'].append(value)

    if reason:
        missing(reason)
        return True
    sources = state.setdefault('sources', {})
    if event == 'av_concat_fit':
        key = 'jointPreparation.fit.' + ('completed' if facts.get('succeeded') is True else 'failed')
        measured = state['measurements']
        measured[key] = measured.get(key, 0) + 1
        sources[key] = 'native_av_preparation'
        if (facts.get('succeeded') is not True or state.get('av_operation') != 'concat'
                or state.get('av_input_streams') != 2
                or measured.get('jointPreparation.fit.completed', 0) + measured.get('jointPreparation.fit.failed', 0) != 1):
            missing('joint_preparation_fit_unresolved')
        seconds = facts.get('seconds')
        if type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0:
            key = 'jointPreparation.fit.seconds'
            measured[key] = measured.get(key, 0) + seconds
            sources[key] = 'native_av_preparation'
        return True

    output = event in OUTPUT_EVENTS
    key = 'av_output_count' if output else 'av_plan_count'
    state[key] = state.get(key, 0) + 1
    if state[key] != 1:
        missing('multiple_joint_preparation_invocations')
        return True
    shapes = facts.get('shapes')
    if (not isinstance(shapes, dict) or not {'video', 'audio'} <= shapes.keys()
            or shapes.keys() - {'video', 'audio', 'target_audio', 'video_mask', 'audio_mask'}
            or any(not isinstance(shape, (tuple, list)) or not 1 <= len(shape) <= 5
                   or any(type(size) is not int or size <= 0 for size in shape) for shape in shapes.values())):
        missing('joint_preparation_metadata_unresolved')
        return True
    shapes = {key: tuple(shape) for key, shape in shapes.items()}
    precisions, devices = facts.get('precisions'), facts.get('devices')
    if any(not isinstance(values, dict) or values.keys() != shapes.keys()
           or any(not isinstance(value, str) or not 0 < len(value) <= 128 for value in values.values())
           for values in (precisions, devices)):
        missing('joint_preparation_metadata_unresolved')
        return True
    video, audio = shapes['video'], shapes['audio']
    if len(video) != 5 or len(audio) not in (3, 4) or video[0] != audio[0]:
        missing('joint_preparation_metadata_unresolved')
        return True
    if output:
        if OUTPUT_EVENTS[event] != state.get('av_operation'):
            missing('joint_preparation_operation_mismatch')
            return True
        state['av_actual_output'] = shapes
        for role, shape in shapes.items():
            for suffix, value in (('precision', precisions[role]), ('device', devices[role])):
                key = 'jointPreparation.output.' + role + '.' + suffix
                state['measurements'][key] = value
                sources[key] = 'native_av_preparation'
                state.setdefault('measurement_availability', {})[key] = 'completion'
            for axis, size in enumerate(shape):
                key = 'jointPreparation.output.' + role + '.axis.' + str(axis)
                state['measurements'][key] = size
                sources[key] = 'native_av_preparation'
                state.setdefault('measurement_availability', {})[key] = 'completion'
        if (shapes != state.get('av_expected_output')
                or any(precisions[role] != state['av_input_precisions'][role]
                       or devices[role] != state['av_input_devices'][role] for role in ('video', 'audio'))):
            missing('joint_preparation_output_mismatch')
        return True

    operation, streams = PLAN_EVENTS[event], facts.get('input_streams')
    target = shapes.get('target_audio')
    if (type(streams) is not int or streams not in (1, 2)
            or operation == 'separate' and (streams != 2 or target is not None)
            or operation == 'concat' and (target is not None) != (streams == 2)
            or operation == 'separate' and ('video_mask' in shapes) != ('audio_mask' in shapes)):
        missing('joint_preparation_metadata_unresolved')
        return True
    mode, fit_axis = 'none', None
    if target is not None:
        if len(target) != len(audio):
            missing('joint_preparation_metadata_unresolved')
            return True
        axes = [axis for axis in range(len(audio)) if audio[axis] != target[axis]]
        if len(axes) > 1 or axes and axes[0] < 2:
            missing('joint_preparation_metadata_unresolved')
            return True
        if axes:
            fit_axis = axes[0]
            mode = 'trim' if audio[fit_axis] > target[fit_axis] else 'pad'
    actual_audio = target or audio
    expected = {'video': video, 'audio': actual_audio}
    if operation == 'separate':
        expected.update({key: shape for key, shape in shapes.items() if key.endswith('_mask')})
    elif 'video_mask' in shapes or 'audio_mask' in shapes:
        expected['video_mask'] = shapes.get('video_mask', video)
        expected['audio_mask'] = shapes['audio_mask'] if 'audio_mask' in shapes and fit_axis is None else actual_audio
    state.update(av_operation=operation, av_input_streams=streams, av_expected_output=expected,
                 av_input_precisions=dict(precisions), av_input_devices=dict(devices))
    values = {'jointPreparation.operation': operation, 'jointPreparation.inputStreams': streams,
              'jointPreparation.fitMode': mode, 'jointPreparation.fitAxis': fit_axis,
              'jointPreparation.videoMaskPresent': 'video_mask' in shapes,
              'jointPreparation.audioMaskPresent': 'audio_mask' in shapes}
    for role, shape in shapes.items():
        values.update({'jointPreparation.input.' + role + '.axis.' + str(axis): size for axis, size in enumerate(shape)})
        values['jointPreparation.input.' + role + '.precision'] = precisions[role]
        values['jointPreparation.input.' + role + '.device'] = devices[role]
    state['precision'] = precisions['video']
    sources['executionPrecision'] = 'native_av_preparation'
    state.setdefault('availability', {})['executionPrecision'] = 'execution_start' if operation == 'concat' else 'live'
    for key, value in values.items():
        state['resolved'][key] = value
        sources[key] = 'native_av_preparation'
        state.setdefault('availability', {})[key] = 'execution_start' if operation == 'concat' else 'live'
    roles = {'video_latent': {'video', 'target_audio', 'video_mask'}, 'audio_latent': {'audio', 'audio_mask'}} if operation == 'concat' else {'av_latent': set(shapes)}
    for role, names in roles.items():
        state.setdefault('reference_features', {})[role] = {
            name + '.axis.' + str(axis): {'value': size, 'source': 'native_av_preparation',
                                        'availableAt': 'execution_start' if operation == 'concat' else 'live'}
            for name in names if name in shapes for axis, size in enumerate(shapes[name])}
    return True
