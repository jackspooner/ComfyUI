"""Correlated H3 vision geometry/cache work; bindings never enter owner records."""
import math

MAX_VISION_CALLS = 128
EVENTS = {'h3_vision_begin', 'h3_vision_geometry', 'h3_vision_execution', 'h3_vision_output',
          'h3_vision_preprocess', 'h3_vision_forward', 'h3_vision_reuse', 'h3_vision_metadata_unavailable'}


def h3_vision_complete(state):
    values = state['resolved']
    image_positions = values.get('h3Conditioning.imagePositions')
    entries = state.get('h3_vision_operations', {}).values()
    return (type(image_positions) is int and values.get('h3Conditioning.audioPositions') == 0
            and all(entry.get('geometry') and entry.get('execution') and entry.get('output')
                    and entry['output_width'] == values.get('h3Conditioning.width') for entry in entries)
            and sum(entry.get('tokens', 0) for entry in entries) == image_positions
            and not any(reason.startswith('h3_vision_') for reason in state['reasons']))


def observe_h3_vision_metrics(state, event, facts):
    if event == 'tensor_shape_unavailable' and facts.get('operation') in EVENTS:
        event = 'h3_vision_metadata_unavailable'
    if event not in EVENTS:
        return False

    def missing(reason='h3_vision_metadata_unresolved'):
        if reason not in state['reasons']:
            state['reasons'].append(reason)

    def retain(key, value, measured=False, available='live'):
        state['measurements' if measured else 'resolved'][key] = value
        state.setdefault('sources', {})[key] = 'native_h3_vision'
        state.setdefault('measurement_availability' if measured else 'availability', {})[key] = available

    def metadata(names, rank):
        shapes, precisions, devices = (facts.get(key) for key in ('shapes', 'precisions', 'devices'))
        if (any(not isinstance(value, dict) or set(value) != names for value in (shapes, precisions, devices))
                or any(not isinstance(shape, (list, tuple)) or len(shape) != rank
                       or any(type(size) is not int or size <= 0 for size in shape) for shape in shapes.values())
                or any(not isinstance(value, str) or not 0 < len(value) <= 128
                       for values in (precisions, devices) for value in values.values())):
            return None
        return {key: tuple(shape) for key, shape in shapes.items()}

    retain('h3Vision.required', True)
    retain('h3Conditioning.required', True)
    retain('managedLoading.required', True)
    if event == 'h3_vision_metadata_unavailable':
        missing()
        return True
    binding = facts.get('binding')
    if type(binding) is not object:
        missing('h3_vision_correlation_unresolved')
        return True
    operations = state.setdefault('h3_vision_operations', {})
    if event == 'h3_vision_begin':
        if binding in operations:
            missing('h3_vision_multiple_invocations')
            return True
        if len(operations) >= MAX_VISION_CALLS:
            missing('h3_vision_call_limit')
            return True
        config = ('layers', 'hidden_size', 'heads', 'patch_size', 'merge_size', 'temporal_patch_size', 'output_width')
        deepstack = facts.get('deepstack')
        shapes = metadata({'source'}, 4)
        if (facts.get('native') is not True or facts.get('mode') not in ('image', 'video') or shapes is None
                or shapes['source'][3] < 3
                or any(type(facts.get(key)) is not int or facts[key] <= 0 for key in config)
                or not isinstance(deepstack, (tuple, list)) or len(deepstack) > 7
                or any(type(value) is not int or not 0 <= value < facts['layers'] for value in deepstack)
                or len(set(deepstack)) != len(deepstack)
                or facts.get('compute_precision') not in ('torch.float32', 'torch.float16', 'torch.bfloat16')
                or any(type(facts.get(key)) is not bool for key in ('training', 'grad_enabled', 'inference_mode'))):
            missing()
            return True
        entry = {'index': len(operations), 'source': shapes['source'], 'mode': facts['mode'],
                 'deepstack': tuple(deepstack), **{key: facts[key] for key in config}}
        operations[binding] = entry
        prefix = 'h3Vision.' + str(entry['index']) + '.'
        for key in (*config, 'mode', 'compute_precision', 'training', 'grad_enabled', 'inference_mode'):
            retain(prefix + key, facts[key])
        retain(prefix + 'deepstack.count', len(deepstack))
        for index, layer in enumerate(deepstack):
            retain(prefix + 'deepstack.' + str(index), layer)
        retain(prefix + 'source.precision', facts['precisions']['source'])
        retain(prefix + 'source.device', facts['devices']['source'])
        for axis, size in enumerate(shapes['source']):
            retain(prefix + 'source.axis.' + str(axis), size, measured=True)
        retain('h3Vision.calls', len(operations))
        return True
    entry = operations.get(binding)
    if entry is None:
        missing('h3_vision_correlation_unresolved')
        return True
    prefix = 'h3Vision.' + str(entry['index']) + '.'
    if event in ('h3_vision_preprocess', 'h3_vision_forward', 'h3_vision_reuse'):
        seconds, succeeded = facts.get('seconds'), facts.get('succeeded')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0 or type(succeeded) is not bool:
            missing()
            return True
        timer = prefix + event.removeprefix('h3_vision_')
        for suffix, value in (('.cpuSeconds', seconds), ('.completed' if succeeded else '.failed', 1)):
            retain(timer + suffix, state['measurements'].get(timer + suffix, 0) + value, measured=True)
        return True
    if event == 'h3_vision_geometry':
        shapes = metadata({'patches'}, 2)
        integers = ('source_frames', 'source_height', 'source_width', 'height', 'width', 'grid_t', 'grid_h', 'grid_w',
                    'patch_size', 'temporal_patch_size', 'merge_size')
        if (entry.get('geometry') or shapes is None
                or any(type(facts.get(key)) is not int or facts[key] <= 0 for key in integers)
                or facts.get('processor') != ('qwen_image' if entry['mode'] == 'image' else 'h3_video')
                or facts.get('interpolation') != 'bilinear'
                or facts['grid_t'] != 1
                or any(facts[key] != entry[key] for key in ('patch_size', 'temporal_patch_size', 'merge_size'))
                or (facts['source_height'], facts['source_width']) != entry['source'][1:3]
                or facts['source_frames'] != (1 if entry['mode'] == 'image' else 2)
                or entry['mode'] == 'video' and entry['source'][0] != 2
                or facts['height'] != facts['grid_h'] * facts['patch_size'] or facts['width'] != facts['grid_w'] * facts['patch_size']
                or facts['grid_h'] % facts['merge_size'] or facts['grid_w'] % facts['merge_size']
                or shapes['patches'] != (facts['grid_t'] * facts['grid_h'] * facts['grid_w'], 3 * facts['temporal_patch_size'] * facts['patch_size'] ** 2)):
            missing('h3_vision_geometry_unresolved')
            return True
        entry['geometry'], entry['patches'] = True, shapes['patches']
        entry['tokens'] = shapes['patches'][0] // facts['merge_size'] ** 2
        for key in (*integers, 'processor', 'interpolation'):
            retain(prefix + key, facts[key])
        retain(prefix + 'tokens', entry['tokens'])
        for key in ('precisions', 'devices'):
            retain(prefix + 'patches.' + key, facts[key]['patches'])
    elif event == 'h3_vision_execution':
        shapes = metadata({'patches'}, 2)
        if (entry.get('execution') or not entry.get('geometry') or shapes is None or shapes['patches'] != entry['patches']
                or facts.get('native') is not True or any(type(facts.get(key)) is not bool for key in ('reused', 'cache_eligible'))
                or facts['reused'] and not facts['cache_eligible']
                or entry['mode'] == 'video' and (facts['reused'] or facts['cache_eligible'])
                or not isinstance(facts.get('device'), str) or not 0 < len(facts['device']) <= 128):
            missing('h3_vision_execution_unresolved')
            return True
        entry['execution'] = True
        for key in ('reused', 'cache_eligible', 'device'):
            retain(prefix + key, facts[key])
    elif event == 'h3_vision_output':
        names = {'merged'} | {'deepstack.' + str(index) for index in range(len(entry['deepstack']))}
        shapes = metadata(names, 2)
        if (entry.get('output') or not entry.get('execution') or shapes is None
                or any(shape != (entry['tokens'], entry['output_width']) for shape in shapes.values())):
            missing('h3_vision_output_unresolved')
            return True
        entry['output'] = True
        for name, shape in shapes.items():
            for axis, size in enumerate(shape):
                retain(prefix + 'output.' + name + '.axis.' + str(axis), size, measured=True, available='completion')
            for key in ('precisions', 'devices'):
                retain(prefix + 'output.' + name + '.' + key, facts[key][name], measured=True, available='completion')
    return True
