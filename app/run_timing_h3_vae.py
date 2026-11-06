"""Bounded native H3 video-VAE encode work, correlated without model references."""
import json
import math

MAX_OPERATIONS, MAX_CLIPS, MAX_FORWARDS = 32, 128, 512
EVENTS = {'h3_vae_begin', 'h3_vae_clip', 'h3_vae_tiles', 'h3_vae_forward_input',
          'h3_vae_forward', 'h3_vae_encode', 'h3_vae_output', 'h3_vae_metadata_unavailable'}


def h3_video_vae_complete(state):
    operations = state.get('h3_vae_operations', {})
    return (bool(operations) and not any(reason.startswith('h3_video_vae_') for reason in state['reasons'])
            and all(op.get('output') and op.get('encode_completed') == 1
                    and len(op['clips']) == op['expected_clips']
                    and op.get('forwards', 0) > 0 and op.get('forward_completed', 0) == op.get('forwards', 0)
                    and all(clip.get('tiles') and clip.get('forwards', 0) == len(clip['tiles']) for clip in op['clips'])
                    for op in operations.values()))


def h3_reference_video_vae_complete(state):
    entries = state.get('h3_reference_entries', [])
    expected = {index for index, entry in enumerate(entries) if entry['visual_encode']}
    if not expected:
        return not state.get('h3_vae_operations')
    if not h3_video_vae_complete(state):
        return False
    operations = list(state['h3_vae_operations'].values())
    if len(operations) != len(expected) or {op['reference_index'] for op in operations} != expected:
        return False
    for op in operations:
        entry = entries[op['reference_index']]
        shape = entry.get('visual_shape')
        if not entry['output'] or shape is None or op['source'] != (1, 3, shape[0], shape[1], shape[2]):
            return False
        prefix = 'h3References.entry.' + str(op['reference_index']) + '.output.visual.axis.'
        if op['output'] != tuple(state['measurements'].get(prefix + str(axis)) for axis in range(5)):
            return False
    return True


def observe_h3_vae_metrics(state, event, facts):
    if event == 'tensor_shape_unavailable' and facts.get('operation') in EVENTS:
        event = 'h3_vae_metadata_unavailable'
    if event not in EVENTS:
        return False

    def missing():
        if 'h3_video_vae_metadata_unresolved' not in state['reasons']:
            state['reasons'].append('h3_video_vae_metadata_unresolved')

    def retain(key, value, measured=False, available='live'):
        state['measurements' if measured else 'resolved'][key] = value
        state.setdefault('sources', {})[key] = 'native_h3_video_vae'
        state.setdefault('measurement_availability' if measured else 'availability', {})[key] = available

    def metadata(name):
        shapes, precisions, devices = (facts.get(key) for key in ('shapes', 'precisions', 'devices'))
        if any(not isinstance(value, dict) or set(value) != {name} for value in (shapes, precisions, devices)):
            return None
        shape = shapes[name]
        if (not isinstance(shape, (tuple, list)) or len(shape) != 5
                or any(type(size) is not int or size <= 0 for size in shape)
                or any(not isinstance(value, str) or not 0 < len(value) <= 128 for value in (precisions[name], devices[name]))):
            return None
        return tuple(shape)

    retain('h3VideoVae.required', True)
    retain('managedLoading.required', True)
    binding = facts.get('binding')
    if event == 'h3_vae_metadata_unavailable' or type(binding) is not object:
        missing()
        return True
    operations = state.setdefault('h3_vae_operations', {})
    if event == 'h3_vae_begin':
        source = metadata('pixels')
        positive = ('tile_size', 'clip_length', 'spatial_ratio', 'temporal_ratio', 'tokens_chunk_size')
        nonnegative = ('tile_overlap_min', 'token_drop')
        reference = facts.get('reference_index')
        if (binding in operations or len(operations) >= MAX_OPERATIONS or facts.get('native') is not True
                or source is None or source[1] != 3 or type(facts.get('tiling')) is not bool
                or any(type(facts.get(key)) is not int or facts[key] <= 0 for key in positive)
                or any(type(facts.get(key)) is not int or facts[key] < 0 for key in nonnegative)
                or facts['tokens_chunk_size'] != math.ceil(facts['clip_length'] / facts['temporal_ratio'])
                or facts['token_drop'] >= facts['tokens_chunk_size']
                or facts['precisions']['pixels'] not in ('torch.float32', 'torch.float16', 'torch.bfloat16')
                or any(type(facts.get(key)) is not bool for key in ('training', 'grad_enabled', 'inference_mode'))
                or any(size % facts['spatial_ratio'] for size in source[-2:])
                or reference is not None and (type(reference) is not int or not 0 <= reference < MAX_OPERATIONS)
                or not isinstance(facts.get('device'), str) or not 0 < len(facts['device']) <= 128):
            missing()
            return True
        op = dict(index=len(operations), source=source, input_precision=facts['precisions']['pixels'], clips=[], reference_index=reference,
                  expected_clips=1 if source[2] == 1 else math.ceil(source[2] / facts['clip_length']),
                  **{key: facts[key] for key in (*positive, *nonnegative, 'tiling', 'device', 'training', 'grad_enabled', 'inference_mode')})
        operations[binding] = op
        prefix = 'h3VideoVae.' + str(op['index']) + '.'
        for axis, size in enumerate(source):
            retain(prefix + 'input.axis.' + str(axis), size)
        for key in ('tiling', 'device', 'spatial_ratio', 'temporal_ratio', 'reference_index', 'training', 'grad_enabled', 'inference_mode'):
            retain(prefix + key, op[key])
        for key in ('clip_length', 'tokens_chunk_size', 'token_drop'):
            retain(prefix + key, op[key], measured=source[2] == 1)
        for key in ('tile_size', 'tile_overlap_min'):
            retain(prefix + key, op[key], measured=True)
        retain(prefix + 'input.precision', facts['precisions']['pixels'])
        retain(prefix + 'input.device', facts['devices']['pixels'])
        retain('h3VideoVae.calls', len(operations))
        return True
    op = operations.get(binding)
    if op is None or op.get('output'):
        missing()
        return True
    prefix = 'h3VideoVae.' + str(op['index']) + '.'
    if event in ('h3_vae_forward', 'h3_vae_encode'):
        seconds, succeeded = facts.get('seconds'), facts.get('succeeded')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0 or type(succeeded) is not bool:
            missing()
            return True
        phase = event.removeprefix('h3_vae_')
        key = prefix + phase + '.cpuSeconds'
        retain(key, state['measurements'].get(key, 0) + seconds, measured=True)
        key = prefix + phase + ('.completed' if succeeded else '.failed')
        retain(key, state['measurements'].get(key, 0) + 1, measured=True)
        op[phase + '_completed'] = op.get(phase + '_completed', 0) + int(succeeded)
        if not succeeded or phase == 'forward' and op['forward_completed'] > op.get('forwards', 0) or phase == 'encode' and op['encode_completed'] > 1:
            missing()
        return True
    if event == 'h3_vae_clip':
        source = op['source']
        shape = metadata('pixels')
        expected = (source[0], source[1], 1 if source[2] == 1 else op['clip_length'], *source[-2:])
        if (shape != expected or facts['precisions']['pixels'] != op['input_precision']
                or sum(len(item['clips']) for item in operations.values()) >= MAX_CLIPS
                or len(op['clips']) >= op['expected_clips']
                or op['clips'] and op['clips'][-1].get('forwards', 0) != len(op['clips'][-1].get('tiles', []))):
            missing()
            return True
        clip = {'shape': shape, 'precision': facts['precisions']['pixels'], 'device': facts['devices']['pixels']}
        if not op['tiling']:
            clip['tiles'] = [shape]
        op['clips'].append(clip)
        clip_prefix = prefix + 'clip.' + str(len(op['clips']) - 1) + '.'
        for axis, size in enumerate(shape):
            retain(clip_prefix + 'axis.' + str(axis), size)
        retain(clip_prefix + 'precision', facts['precisions']['pixels'])
        retain(clip_prefix + 'device', facts['devices']['pixels'])
        return True
    if not op['clips']:
        missing()
        return True
    clip = op['clips'][-1]
    clip_prefix = prefix + 'clip.' + str(len(op['clips']) - 1) + '.'
    if event == 'h3_vae_tiles':
        if not op['tiling'] or 'tiles' in clip:
            missing()
            return True
        axes = {}
        for axis, size in (('y', clip['shape'][-2]), ('x', clip['shape'][-1])):
            starts, lengths, overlaps = (facts.get(axis + '_' + key) for key in ('starts', 'lengths', 'overlaps'))
            if (any(not isinstance(value, (tuple, list)) or len(value) > 64 for value in (starts, lengths, overlaps))
                    or not starts or len(starts) != len(lengths) or len(overlaps) != len(starts) - 1
                    or any(type(value) is not int for values in (starts, lengths, overlaps) for value in values)
                    or starts[0] != 0 or starts[-1] + lengths[-1] != size
                    or any(value <= 0 or value % op['spatial_ratio'] for value in lengths)
                    or any(value < 0 or value % op['spatial_ratio'] for value in starts)
                    or any(not 0 < overlap < min(lengths[index:index + 2]) or overlap % op['spatial_ratio']
                           or starts[index + 1] != starts[index] + lengths[index] - overlap for index, overlap in enumerate(overlaps))):
                missing()
                return True
            axes[axis] = lengths
            for name, values in (('starts', starts), ('lengths', lengths), ('overlaps', overlaps)):
                encoded = json.dumps(values, separators=(',', ':'))
                if len(encoded) > 512:
                    missing()
                    return True
                retain(clip_prefix + axis + '.' + name, encoded)
        planned = sum(len(item.get('tiles', [])) for operation in operations.values() for item in operation['clips'])
        if planned + len(axes['x']) * len(axes['y']) > MAX_FORWARDS:
            missing()
            return True
        clip['tiles'] = [(*clip['shape'][:-2], h, w) for h in axes['y'] for w in axes['x']]
    elif event == 'h3_vae_forward_input':
        shape, index = metadata('pixels'), clip.get('forwards', 0)
        if (sum(item.get('forwards', 0) for item in operations.values()) >= MAX_FORWARDS
                or index >= len(clip.get('tiles', [])) or shape != clip['tiles'][index]
                or facts['precisions']['pixels'] != clip['precision'] or facts['devices']['pixels'] != clip['device']):
            missing()
            return True
        clip['forwards'], op['forwards'] = index + 1, op.get('forwards', 0) + 1
        forward_prefix = prefix + 'forward.' + str(op['forwards'] - 1) + '.'
        retain(forward_prefix + 'shape', ','.join(map(str, shape)))
        retain(forward_prefix + 'precision', facts['precisions']['pixels'])
        retain(forward_prefix + 'device', facts['devices']['pixels'])
    elif event == 'h3_vae_output':
        shape = metadata('latents')
        source = op['source']
        frames = 1 if source[2] == 1 else op['expected_clips'] * op['tokens_chunk_size'] - op['token_drop']
        expected = (source[0], 24, frames, source[3] // op['spatial_ratio'], source[4] // op['spatial_ratio'])
        if op.get('output') or shape != expected:
            missing()
            return True
        op['output'] = shape
        for axis, size in enumerate(shape):
            retain(prefix + 'output.axis.' + str(axis), size, measured=True, available='completion')
        retain(prefix + 'output.precision', facts['precisions']['latents'], measured=True, available='completion')
        retain(prefix + 'output.device', facts['devices']['latents'], measured=True, available='completion')
    return True
