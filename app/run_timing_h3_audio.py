"""Correlated H3 reference resampling, wrapper preparation and native audio encode."""
import math

MAX_AUDIO_OPERATIONS = 32
PHASES = ('resample', 'encoder', 'projection', 'mean')
EVENTS = {'h3_audio_' + phase for phase in PHASES} | {'h3_audio_source', 'h3_audio_prepared',
    'h3_audio_wrapper', 'h3_audio_plan', 'h3_audio_encoder_input', 'h3_audio_projection_input',
    'h3_audio_mean_input', 'h3_audio_output', 'h3_audio_result', 'h3_audio_metadata_unavailable'}


def h3_audio_complete(state):
    operations = state.get('h3_audio_operations', {})
    return (bool(operations) and not any(reason.startswith('h3_audio_') for reason in state['reasons'])
            and all(all(op.get(key) for key in ('prepared', 'wrapper', 'plan', 'encoder', 'projection', 'mean', 'output', 'result'))
                    and all(op.get(phase + '_completed', 0) == (int(op['rate'] != op['target_rate']) if phase == 'resample' else 1)
                            for phase in PHASES) for op in operations.values()))


def h3_reference_audio_complete(state):
    entries = state.get('h3_reference_entries', [])
    expected = {index for index, entry in enumerate(entries) if entry['audio_encode']}
    operations = list(state.get('h3_audio_operations', {}).values())
    if not expected:
        return not operations
    if not h3_audio_complete(state) or len(operations) != len(expected) or {op['reference_index'] for op in operations} != expected:
        return False
    for op in operations:
        index = op['reference_index']
        entry = entries[index]
        prefix = 'h3References.entry.' + str(index) + '.'
        if (not entry['output'] or op['source'] != entry['shapes'].get('waveform')
                or op['rate'] != state['resolved'].get(prefix + 'audio.sampleRate')
                or op['result'] != tuple(state['measurements'].get(prefix + 'output.audio.axis.' + str(axis)) for axis in range(4))):
            return False
    return True


def scoped_h3_audio_replaces_legacy_limit(state):
    """Only fully proved reference calls replace the generic repeated-call gap."""
    count = len(state.get('h3_audio_operations', {}))
    return (count > 0 and h3_reference_audio_complete(state) and state.get('audioProcessed_count') == count
            and all(state.get(prefix + '_count', 0) <= 1 for prefix in ('audioSource', 'audioDecode', 'audioPrepared', 'audioOutput')))


def observe_h3_audio_metrics(state, event, facts):
    if event == 'tensor_shape_unavailable' and facts.get('operation') in EVENTS:
        event = 'h3_audio_metadata_unavailable'
    if event not in EVENTS:
        return False

    def missing():
        if 'h3_audio_metadata_unresolved' not in state['reasons']:
            state['reasons'].append('h3_audio_metadata_unresolved')

    def retain(key, value, measured=False, available='live'):
        state['measurements' if measured else 'resolved'][key] = value
        state.setdefault('sources', {})[key] = 'native_h3_audio_vae'
        state.setdefault('measurement_availability' if measured else 'availability', {})[key] = available

    def metadata(name, rank):
        shapes, precisions, devices = (facts.get(key) for key in ('shapes', 'precisions', 'devices'))
        if any(not isinstance(value, dict) or set(value) != {name} for value in (shapes, precisions, devices)):
            return None
        shape = shapes[name]
        if (not isinstance(shape, (tuple, list)) or len(shape) != rank
                or any(type(size) is not int or size <= 0 for size in shape)
                or precisions[name] not in ('torch.float32', 'torch.float16', 'torch.bfloat16')
                or not isinstance(devices[name], str) or not 0 < len(devices[name]) <= 128):
            return None
        return tuple(shape)

    def shape_facts(prefix, name, shape, measured=False):
        for axis, size in enumerate(shape):
            retain(prefix + '.axis.' + str(axis), size, measured, 'completion' if measured else 'live')
        for key in ('precisions', 'devices'):
            retain(prefix + '.' + key, facts[key][name], measured, 'completion' if measured else 'live')

    retain('h3AudioVae.required', True)
    retain('managedLoading.required', True)
    binding = facts.get('binding')
    if event == 'h3_audio_metadata_unavailable' or type(binding) is not object:
        missing()
        return True
    operations = state.setdefault('h3_audio_operations', {})
    if event == 'h3_audio_source':
        shape = metadata('waveform', 3)
        reference = facts.get('reference_index')
        if (binding in operations or len(operations) >= MAX_AUDIO_OPERATIONS or shape is None
                or any(type(facts.get(key)) is not int or facts[key] <= 0 for key in ('sample_rate', 'target_rate'))
                or reference is not None and (type(reference) is not int or not 0 <= reference < MAX_AUDIO_OPERATIONS)):
            missing()
            return True
        op = dict(index=len(operations), source=shape, rate=facts['sample_rate'], target_rate=facts['target_rate'], reference_index=reference)
        operations[binding] = op
        prefix = 'h3AudioVae.' + str(op['index'])
        for key in ('rate', 'target_rate', 'reference_index'):
            retain(prefix + '.' + key, op[key])
        # Full batches are resampled before selecting the first item for the wrapper.
        retain(prefix + '.resample.batch', shape[0], measured=op['rate'] == op['target_rate'])
        retain(prefix + '.source.channels', shape[1])
        retain(prefix + '.source.samples', shape[2])
        retain(prefix + '.source.precision', facts['precisions']['waveform'])
        retain(prefix + '.source.device', facts['devices']['waveform'])
        retain('h3AudioVae.calls', len(operations))
        return True
    op = operations.get(binding)
    if op is None or op.get('result') or 'reference_index' in facts and facts['reference_index'] != op['reference_index']:
        missing()
        return True
    prefix = 'h3AudioVae.' + str(op['index'])
    phase = event.removeprefix('h3_audio_')
    if phase in PHASES:
        seconds, succeeded = facts.get('seconds'), facts.get('succeeded')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0 or type(succeeded) is not bool:
            missing()
            return True
        key = prefix + '.' + phase + '.cpuSeconds'
        retain(key, state['measurements'].get(key, 0) + seconds, measured=True)
        key = prefix + '.' + phase + ('.completed' if succeeded else '.failed')
        retain(key, state['measurements'].get(key, 0) + 1, measured=True)
        op[phase + '_completed'] = op.get(phase + '_completed', 0) + int(succeeded)
        if (not succeeded or op[phase + '_completed'] > 1
                or phase == 'resample' and (op['rate'] == op['target_rate'] or op.get('prepared'))
                or phase != 'resample' and not op.get(phase)):
            missing()
        return True
    if op.get(phase):
        missing()
        return True
    if event == 'h3_audio_prepared':
        shape = metadata('pixels', 3)
        if (shape is None or shape[0] != 1 or shape[2] != op['source'][1]
                or op.get('resample_completed', 0) != int(op['rate'] != op['target_rate'])
                or op['rate'] == op['target_rate'] and shape[1] != op['source'][2]):
            missing()
            return True
        op['prepared'] = shape
        shape_facts(prefix + '.prepared', 'pixels', shape)
    elif event == 'h3_audio_wrapper':
        shape, prepared = metadata('pixels', 3), op.get('prepared')
        crop, ratio = facts.get('crop_input'), facts.get('crop_ratio')
        if (prepared is None or shape is None or facts.get('native') is not True or type(crop) is not bool
                or crop and (type(ratio) is not int or ratio <= 0)
                or not crop and ratio is not None or type(facts.get('output_channels')) is not int or facts['output_channels'] != 2
                or prepared[2] < 2 and facts.get('pad_channel_value') != 'replicate'):
            missing()
            return True
        length = prepared[1] // ratio * ratio if crop else prepared[1]
        if shape != (1, length, 2):
            missing()
            return True
        op['wrapper'] = shape
        shape_facts(prefix + '.wrapper', 'pixels', shape)
        retain(prefix + '.wrapper.crop', crop)
        retain(prefix + '.wrapper.channelWork', 'replicate' if prepared[2] < 2 else 'truncate' if prepared[2] > 2 else 'identity')
        retain(prefix + '.wrapper.cropRatio', ratio, measured=True)
    elif event == 'h3_audio_plan':
        shape, wrapper = metadata('waveform', 3), op.get('wrapper')
        config = ('hop_length', 'sample_rate', 'latent_width', 'output_channels', 'heads')
        if (wrapper is None or shape is None or shape != (wrapper[0], wrapper[2], wrapper[1])
                or facts.get('native') is not True or facts.get('sample_rate') != op['target_rate']
                or any(type(facts.get(key)) is not int or facts[key] <= 0 for key in config)
                or facts['output_channels'] != 32 or facts['latent_width'] % facts['heads']
                or type(facts.get('right_pad')) is not int or facts['right_pad'] != (-shape[2]) % facts['hop_length']
                or any(type(facts.get(key)) is not bool for key in ('training', 'grad_enabled', 'inference_mode'))):
            missing()
            return True
        op['plan'], op['frames'] = shape, (shape[2] + facts['right_pad']) // facts['hop_length']
        op.update({key: facts[key] for key in config})
        shape_facts(prefix + '.native', 'waveform', shape)
        for key in (*config, 'right_pad', 'training', 'grad_enabled', 'inference_mode'):
            retain(prefix + '.' + key, facts[key])
    elif event.endswith('_input'):
        phase = phase.removesuffix('_input')
        shape, plan = metadata('input', 3), op.get('plan')
        if plan is None:
            missing()
            return True
        batch = plan[0] * plan[1]
        expected = {'encoder': (batch, 1, op['frames'] * op['hop_length']),
                    'projection': (batch, op['frames'], op['latent_width']),
                    'mean': (batch, op['output_channels'], op['frames'])}
        previous = {'projection': 'encoder', 'mean': 'projection'}.get(phase)
        if (phase not in expected or op.get(phase) or shape != expected[phase]
                or previous and op.get(previous + '_completed') != 1):
            missing()
            return True
        op[phase] = shape
        shape_facts(prefix + '.' + phase, 'input', shape)
    elif event in ('h3_audio_output', 'h3_audio_result'):
        shape, plan = metadata('latents', 4), op.get('plan')
        if (plan is None or shape != (plan[0], op['output_channels'], plan[1], op['frames'])
                or any(op.get(part + '_completed') != 1 for part in ('encoder', 'projection', 'mean'))
                or phase == 'result' and shape != op.get('output')):
            missing()
            return True
        op[phase] = shape
        shape_facts(prefix + '.' + phase, 'latents', shape, measured=True)
    return True
