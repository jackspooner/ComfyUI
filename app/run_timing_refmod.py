"""Bounded RefMod loader evidence; no file reads, tensor values or inferred backbone."""
import math
import re

REFMOD_LOADERS = {'MiniMaxH3RefModsLoader': 'slots', 'MiniMaxH3RefModsAxis': 'axis'}
MAX_MEMBERS = 256
EVENTS = {'begin', 'slot', 'loaded', 'selection', 'duration', 'end', 'metadata_unavailable'}


def refmod_loader_complete(state):
    operation = state.get('refmod_loader')
    return bool(operation and operation.get('ended') and not any(
        reason.startswith('refmod_loader_') for reason in state['reasons']))


def observe_refmod_loader_metrics(state, event, facts):
    if not event.startswith('refmod_loader_') or event.removeprefix('refmod_loader_') not in EVENTS:
        return False
    event = event.removeprefix('refmod_loader_')

    def missing(reason='metadata_unresolved'):
        name = 'refmod_loader_' + reason
        if name not in state['reasons']:
            state['reasons'].append(name)

    def retain(key, value, measured=False):
        key = 'refmodLoader.' + key
        state['measurements' if measured else 'resolved'][key] = value
        state.setdefault('sources', {})[key] = 'native_refmod_loader_v1'
        state.setdefault('measurement_availability' if measured else 'availability', {})[key] = 'live'

    retain('required', True)
    binding = facts.get('binding')
    if type(binding) is not object:
        missing('correlation_unresolved')
        return True
    if event == 'begin':
        if (state.get('refmod_loader') or facts.get('version') != 1 or facts.get('slots') != 8
                or facts.get('mode') not in ('slots', 'axis')):
            missing('invocation_unresolved')
            return True
        state['refmod_loader'] = {'binding': binding, 'slots': {}, 'members': 0, 'mode': facts['mode'], 'spans': {}}
        retain('mode', facts['mode'])
        return True
    operation = state.get('refmod_loader')
    if operation is None or operation['binding'] is not binding or operation.get('ended'):
        missing('correlation_unresolved')
        return True
    if event == 'metadata_unavailable':
        missing()
        return True
    if event == 'end':
        slots = operation['slots']
        expected = sum(len(slot['members']) * slot['copies'] for slot in slots.values())
        if (set(slots) != set(range(1, 9)) or type(facts.get('references')) is not int
                or facts['references'] != expected or operation['spans'].get(('total', None)) != 1
                or any(slot['active'] and not slot.get('selected') for slot in slots.values())):
            missing('selection_unresolved')
        else:
            operation['ended'] = True
            retain('references', expected)
        return True
    index = facts.get('slot')
    if event == 'duration':
        phase, seconds, succeeded = (facts.get(key) for key in ('phase', 'seconds', 'succeeded'))
        if (phase not in ('metadata_read', 'standalone_read', 'bundle_read', 'total')
                or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                or type(succeeded) is not bool
                or (phase == 'total' and index is not None)
                or (phase != 'total' and (type(index) is not int or index not in operation['slots']
                                         or not operation['slots'][index]['active']))):
            missing()
            return True
        key = (phase, index)
        operation['spans'][key] = operation['spans'].get(key, 0) + 1
        if operation['spans'][key] != 1 or not succeeded:
            missing('execution_unresolved')
        prefix = ('slot.' + str(index) + '.' if index is not None else '') + phase
        retain(prefix + '.cpuSeconds', seconds, True)
        retain(prefix + '.succeeded', succeeded, True)
        return True
    if type(index) is not int or not 1 <= index <= 8:
        missing()
        return True
    prefix = 'slot.' + str(index) + '.'
    if event == 'slot':
        if (index in operation['slots'] or type(facts.get('active')) is not bool
                or facts.get('side') not in (('single',) if operation['mode'] == 'slots' else ('a', 'b', 'none'))
                or type(facts.get('copies')) is not int or not 1 <= facts['copies'] <= 10
                or operation['mode'] == 'axis' and facts['copies'] != 1
                or facts.get('selection') not in ('All', 'Visual', 'Audio')
                or any(type(facts.get(key)) not in (int, float) or not math.isfinite(facts[key])
                       or not 0 <= facts[key] <= 1 for key in ('strength', 'visual_strength', 'audio_strength'))
                or facts['active'] and (facts['strength'] <= 0 or facts['side'] == 'none')):
            missing()
            return True
        operation['slots'][index] = {key: facts[key] for key in ('active', 'copies', 'selection', 'strength', 'visual_strength', 'audio_strength')}
        operation['slots'][index]['members'] = []
        retain(prefix + 'active', facts['active'])
        # Skipped controls and A/B direction do not change file-loading work.
        if facts['active']:
            for key in ('copies', 'selection', 'strength', 'visual_strength', 'audio_strength'):
                retain(prefix + key, facts[key])
        retain(prefix + 'side', facts['side'], True)
        return True
    slot = operation['slots'].get(index)
    if not slot or not slot['active'] or slot.get('selected'):
        missing('selection_unresolved')
        return True
    container = facts.get('container')
    if container not in ('standalone', 'bundle') or operation['spans'].get(('metadata_read', index)) != 1:
        missing('selection_unresolved')
        return True
    if event == 'selection':
        if (type(facts.get('members')) is not int or facts['members'] != len(slot['members'])
                or container == 'standalone' and facts['members'] > 1
                or container == 'bundle' and operation['spans'].get(('bundle_read', index)) != 1
                or any(member['container'] != container for member in slot['members'])):
            missing('selection_unresolved')
        else:
            slot['selected'] = True
            retain(prefix + 'container', container)
            retain(prefix + 'members', facts['members'])
        return True
    if operation['members'] >= MAX_MEMBERS:
        missing('member_limit')
        return True
    shape, kind, reused = (facts.get(key) for key in ('shape', 'kind', 'reused'))
    member, stamp, assurance = (facts.get(key) for key in ('member', 'stamp_identity', 'stamp_assurance'))
    audio = kind == 'audio'
    if (kind not in ('image', 'video', 'audio') or not isinstance(shape, (tuple, list))
            or len(shape) != (4 if audio else 5) or any(type(size) is not int or size <= 0 for size in shape)
            or tuple(shape[:3] if audio else shape[:2]) != ((1, 32, 2) if audio else (1, 24))
            or not audio and (any(size % 2 for size in shape[-2:]) or kind == 'image' and shape[2] != 1)
            or type(reused) is not bool or facts.get('device') != 'cpu'
            or facts.get('precision') not in ('torch.float16', 'torch.bfloat16', 'torch.float32', 'torch.float64',
                                             'torch.int8', 'torch.int16', 'torch.int32', 'torch.int64', 'torch.uint8', 'torch.bool')
            or type(facts.get('tokens')) is not int
            or facts['tokens'] != (2 * shape[-1] if audio else shape[2] * (shape[3] // 2) * (shape[4] // 2))
            or container == 'bundle' and (reused or type(member) is not int or not 0 <= member < 256
                                         or any(item['member'] == member for item in slot['members'])
                                         or operation['spans'].get(('bundle_read', index)) != 1)
            or container == 'standalone' and (member is not None or slot['members']
                                             or not reused and operation['spans'].get(('standalone_read', index)) != 1)
            or not (stamp is None and assurance == 'unavailable'
                    or isinstance(stamp, str) and re.fullmatch('[a-f0-9]{64}', stamp) and assurance == 'size_mtime_ctime')):
        missing()
        return True
    modality = 'audio' if audio else 'visual'
    if slot[modality + '_strength'] <= 0 or slot['selection'] != 'All' and slot['selection'].lower() != modality:
        missing('selection_unresolved')
        return True
    number = len(slot['members'])
    slot['members'].append({'container': container, 'member': member})
    operation['members'] += 1
    prefix += 'member.' + str(number) + '.'
    for key in ('container', 'kind', 'reused', 'precision', 'device', 'tokens'):
        retain(prefix + key, facts[key])
    for axis, size in enumerate(shape):
        retain(prefix + 'axis.' + str(axis), size)
    for key in ('member', 'stamp_identity', 'stamp_assurance'):
        retain(prefix + key, facts[key], True)
    return True
