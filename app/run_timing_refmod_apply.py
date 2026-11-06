"""Actual RefMod Apply branches, bounded per-reference work and CPU measurements."""
import math

MAX_REFERENCES = 256
EVENTS = {'begin', 'settings', 'selection', 'reference_begin', 'reference_plan', 'blur_plan',
          'reference_output', 'debug_plan', 'debug_output', 'attached', 'duration', 'end',
          'limit', 'metadata_unavailable'}
PRECISIONS = {'torch.float16', 'torch.bfloat16', 'torch.float32', 'torch.float64',
              'torch.int8', 'torch.int16', 'torch.int32', 'torch.int64', 'torch.uint8', 'torch.bool'}


def refmod_apply_complete(state):
    operation = state.get('refmod_apply')
    return bool(operation and operation.get('ended') and not any(
        reason.startswith('refmod_apply_') for reason in state['reasons']))


def observe_refmod_apply_metrics(state, event, facts):
    if not event.startswith('refmod_apply_') or event.removeprefix('refmod_apply_') not in EVENTS:
        return False
    event = event.removeprefix('refmod_apply_')

    def missing(reason='metadata_unresolved'):
        name = 'refmod_apply_' + reason
        if name not in state['reasons']:
            state['reasons'].append(name)

    def retain(key, value, measured=False):
        key = 'refmodApply.' + key
        state['measurements' if measured else 'resolved'][key] = value
        state.setdefault('sources', {})[key] = 'native_refmod_apply_v1'
        state.setdefault('measurement_availability' if measured else 'availability', {})[key] = 'live'

    def tensor():
        shape = facts.get('shape')
        if (not isinstance(shape, (tuple, list)) or not 1 <= len(shape) <= 5
                or any(type(size) is not int or size <= 0 for size in shape)
                or not isinstance(facts.get('precision'), str) or facts['precision'] not in PRECISIONS
                or not isinstance(facts.get('device'), str) or not 0 < len(facts['device']) <= 64
                or any(type(facts.get(key)) is not bool for key in ('requires_grad', 'grad_enabled', 'inference_mode'))
                or facts.get('default_precision') not in ('torch.float32', 'torch.float64')):
            return None
        return tuple(shape)

    def retain_tensor(prefix, shape, measured=False):
        for axis, size in enumerate(shape):
            retain(prefix + 'axis.' + str(axis), size, measured)
        for key in ('precision', 'device', 'requires_grad', 'default_precision', 'grad_enabled', 'inference_mode'):
            retain(prefix + key, facts[key], measured)

    retain('required', True)
    binding = facts.get('binding')
    if type(binding) is not object:
        missing('correlation_unresolved')
        return True
    if event == 'begin':
        if state.get('refmod_apply') or facts.get('version') != 1 or facts.get('index') is not None:
            missing('invocation_unresolved')
        else:
            state['refmod_apply'] = {'binding': binding, 'references': {}, 'spans': {}}
        return True
    operation = state.get('refmod_apply')
    if not operation or operation['binding'] is not binding or operation.get('ended'):
        missing('correlation_unresolved')
        return True
    if event in ('limit', 'metadata_unavailable'):
        missing('reference_limit' if event == 'limit' else 'metadata_unresolved')
        return True
    index = facts.get('index')
    references = operation['references']
    reference_event = event in ('reference_begin', 'reference_plan', 'blur_plan', 'reference_output')
    if index is not None and (type(index) is not int or not 0 <= index < MAX_REFERENCES) or reference_event and index is None:
        missing('correlation_unresolved')
        return True
    if event == 'duration':
        phase, seconds, succeeded = (facts.get(key) for key in ('phase', 'seconds', 'succeeded'))
        local = phase in ('reference', 'blur')
        if (phase not in ('reference', 'blur', 'total', 'preset_read', 'preset_save', 'debug_render', 'debug_conversion')
                or (index is not None) != local or local and index not in references
                or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                or type(succeeded) is not bool):
            missing()
            return True
        key = (index, phase)
        operation['spans'][key] = operation['spans'].get(key, 0) + 1
        if operation['spans'][key] != 1 or not succeeded:
            missing('execution_unresolved')
        prefix = ('reference.' + str(index) + '.' if local else '') + phase
        retain(prefix + '.cpuSeconds', seconds, True)
        retain(prefix + '.succeeded', succeeded, True)
        return True
    if not reference_event and index is not None:
        missing('correlation_unresolved')
        return True
    if event == 'settings':
        if (operation.get('settings') or facts.get('conditioning') not in ('native', 'pack')
                or any(type(facts.get(key)) is not bool for key in ('preset_read', 'preset_save'))):
            missing()
        else:
            operation['settings'] = {key: facts[key] for key in ('conditioning', 'preset_read', 'preset_save')}
            for key, value in operation['settings'].items():
                retain(key, value)
        return True
    if event == 'selection':
        if ('selection' in operation or not operation.get('settings')
                or any(type(facts.get(key)) is not int for key in ('inputs', 'selected'))
                or not 0 <= facts['selected'] <= facts['inputs'] or type(facts.get('scrambled')) is not bool):
            missing('selection_unresolved')
        else:
            operation['selection'] = facts['selected']
            operation['inputs'] = facts['inputs']
            if facts['selected'] > MAX_REFERENCES:
                missing('reference_limit')
            retain('inputs', facts['inputs'])
            retain('selected', facts['selected'])
            retain('scrambled', facts['scrambled'])
        return True
    if event == 'debug_plan':
        if (operation.get('debug') or any(type(facts.get(key)) is not int or not 1 <= facts[key] <= 4096
                                         for key in ('width', 'height', 'samples'))):
            missing('debug_unresolved')
        else:
            operation['debug'] = (1, facts['height'], facts['width'], 3)
            for key in ('width', 'height', 'samples'):
                retain('debug.' + key, facts[key])
        return True
    if event == 'debug_output':
        shape = tensor()
        if (operation.get('debug_output') or shape != operation.get('debug') or shape is None
                or facts.get('device') != 'cpu' or facts.get('precision') != 'torch.float32'):
            missing('debug_unresolved')
        else:
            operation['debug_output'] = True
            retain_tensor('debug.output.', shape, True)
        return True
    if event == 'attached':
        blocks = sum(entry.get('branch') != 'drop' for entry in references.values())
        if (operation.get('attached') or 'selection' not in operation
                or any(type(facts.get(key)) is not int or facts[key] < 0 for key in ('entries', 'blocks'))
                or facts['blocks'] != blocks or len(references) != operation['selection']
                or operation.get('settings', {}).get('conditioning') == 'pack' and facts['entries'] != 1):
            missing('attachment_unresolved')
        else:
            operation['attached'] = True
            retain('conditioningEntries', facts['entries'])
            retain('attachedBlocks', facts['blocks'])
        return True
    if event == 'end':
        spans, settings = operation['spans'], operation.get('settings')
        if (not settings or 'selection' not in operation or not operation.get('attached')
                or not operation.get('debug_output')
                or any(spans.get((None, phase)) != 1 for phase in ('total', 'debug_render', 'debug_conversion'))
                or any(spans.get((None, phase), 0) != int(settings[key])
                       for phase, key in (('preset_read', 'preset_read'), ('preset_save', 'preset_save')))
                or len(references) != operation['selection']
                or any(not entry.get('plan') or spans.get((i, 'reference')) != 1
                       or (entry['branch'] != 'drop') != bool(entry.get('output'))
                       or spans.get((i, 'blur'), 0) != int(entry['branch'] in ('frame_mix', 'scalar_mix'))
                       or bool(entry.get('blur')) != (entry['branch'] in ('frame_mix', 'scalar_mix'))
                       for i, entry in references.items())):
            missing('execution_unresolved')
        else:
            operation['ended'] = True
            # Debug conversion and native blur use FP32. Keep the observed
            # reference storage precisions distinct without inventing a model
            # dtype or using completed output tensors as forecast inputs.
            precisions = {'torch.float32'} | {entry['precision'] for entry in references.values() if entry['branch'] != 'drop'}
            precision = next(iter(precisions)) if len(precisions) == 1 else 'mixed:' + '+'.join(sorted(precisions))
            state['resolved']['executionPrecision'] = precision
            state.setdefault('sources', {})['executionPrecision'] = 'native_refmod_apply_v1'
            state.setdefault('availability', {})['executionPrecision'] = 'live'
        return True
    prefix = 'reference.' + str(index) + '.'
    if event == 'reference_begin':
        if (index != len(references) or 'selection' not in operation or index >= operation['selection']
                or type(facts.get('source')) is not int or not 0 <= facts['source'] < operation['inputs']):
            missing('selection_unresolved')
        else:
            references[index] = {}
            retain(prefix + 'sourceOrdinal', facts['source'], True)
        return True
    entry = references.get(index)
    if entry is None:
        missing('correlation_unresolved')
        return True
    shape = tensor()
    if shape is None:
        missing()
        return True
    if event == 'reference_plan':
        kind, frames, branch, weights, strength = (facts.get(key) for key in ('kind', 'frames', 'branch', 'weights', 'strength'))
        audio = kind == 'audio'
        if (entry.get('plan') or facts.get('native') is not True or kind not in ('image', 'video', 'audio')
                or len(shape) != (4 if audio else 5)
                or shape[:3 if audio else 2] != ((1, 32, 2) if audio else (1, 24))
                or type(frames) is not int or frames != shape[-1 if audio else 2]
                or not audio and (any(size % 2 for size in shape[-2:]) or kind == 'image' and frames != 1)
                or branch not in ('drop', 'reuse', 'frame_mix', 'scalar_mix')
                or type(weights) is not int or weights not in (0, frames)
                or branch == 'frame_mix' and weights != frames or branch in ('drop', 'scalar_mix') and weights != 0
                or type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1
                or (branch == 'drop') != (strength == 0) or branch == 'scalar_mix' and strength >= 1):
            missing()
        else:
            entry.update(plan=True, shape=shape, branch=branch, precision=facts['precision'], device=facts['device'])
            for key in ('kind', 'branch', 'frames', 'weights'):
                retain(prefix + key, facts[key], branch == 'drop' and key != 'branch')
            retain(prefix + 'strength', strength, True)
            retain_tensor(prefix + 'input.', shape, branch == 'drop')
        return True
    if not entry.get('plan'):
        missing('correlation_unresolved')
        return True
    if event == 'blur_plan':
        backend = facts.get('backend')
        if (entry.get('blur') or entry['branch'] not in ('frame_mix', 'scalar_mix') or shape != entry['shape']
                or facts['precision'] != entry['precision'] or facts['device'] != entry['device']
                or type(facts.get('factor')) is not int or facts['factor'] != 8
                or backend not in ('cpu', 'cuda') or facts['device'].split(':')[0] != backend):
            missing('blur_unresolved')
        else:
            entry['blur'] = True
            retain(prefix + 'blur.factor', facts['factor'])
            retain(prefix + 'blur.backend', backend)
            retain_tensor(prefix + 'blur.input.', shape)
        return True
    if event == 'reference_output':
        if (entry.get('output') or entry['branch'] == 'drop' or shape != entry['shape']
                or facts['device'] != entry['device']):
            missing('output_unresolved')
        else:
            entry['output'] = True
            retain_tensor(prefix + 'output.', shape, True)
    return True
