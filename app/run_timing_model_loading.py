"""Bounded model-manager plans and CPU boundary measurements, without model retention."""
import math

PLAN_EVENTS = {'managed_load_plan', 'managed_load_selection', 'managed_load_clones',
               'managed_load_requirement', 'managed_load_entry_plan', 'managed_load_usage'}
MEASURE_EVENTS = {'managed_model_load': 'load', 'managed_model_clone_detach': 'cloneDetach',
                  'managed_model_unload': 'unload'}


def managed_loading_complete(state):
    operations = state.get('managed_load_operations', {})
    return (bool(operations) and not any(reason.startswith('managed_load_') for reason in state['reasons'])
            and all(all(len(operation[kind]) == operation['models'] for kind in ('selection', 'clones', 'plan'))
                    for operation in operations.values()))


def observe_model_loading_metrics(state, event, facts):
    if event not in PLAN_EVENTS and event not in MEASURE_EVENTS and event not in ('managed_model_unload_result', 'managed_load_required'):
        return False
    def missing():
        if 'managed_load_metadata_unresolved' not in state['reasons']:
            state['reasons'].append('managed_load_metadata_unresolved')
    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    def retain(key, value, target='resolved'):
        state[target][key] = value
        state.setdefault('sources', {})[key] = 'native_model_manager'
        state.setdefault('measurement_availability' if target == 'measurements' else 'availability', {})[key] = 'live'
    def count(key):
        retain(key, state['measurements'].get(key, 0) + 1, 'measurements')
    if event == 'managed_load_required':
        retain('managedLoading.required', True)
        state['availability']['managedLoading.required'] = 'execution_start'
        state['sources']['managedLoading.required'] = 'native_sampler'
        return True
    if event in MEASURE_EVENTS:
        if not number(facts.get('seconds')) or type(facts.get('succeeded')) is not bool:
            missing()
            return True
        prefix = 'managedLoading.' + MEASURE_EVENTS[event]
        count(prefix + ('.completed' if facts['succeeded'] else '.failed'))
        retain(prefix + '.seconds', state['measurements'].get(prefix + '.seconds', 0) + facts['seconds'], 'measurements')
        binding = facts.get('operation')
        operation = state.get('managed_load_operations', {}).get(binding) if type(binding) is object else None
        index = facts.get('index')
        if operation is not None and type(index) is int and 0 <= index < operation['models']:
            key = 'managedLoading.' + str(operation['index']) + '.entry.' + str(index) + '.' + MEASURE_EVENTS[event] + '.seconds'
            retain(key, state['measurements'].get(key, 0) + facts['seconds'], 'measurements')
        if event == 'managed_model_unload' and number(facts.get('requested_bytes')):
            retain(prefix + '.lastRequestedBytes', facts['requested_bytes'], 'measurements')
        return True
    if event == 'managed_model_unload_result':
        if type(facts.get('detached')) is bool:
            count('managedLoading.unload.' + ('detached' if facts['detached'] else 'retained'))
        else:
            missing()
        return True
    operations = state.setdefault('managed_load_operations', {})
    binding = facts.get('operation')
    if type(binding) is not object:
        missing()
        return True
    if event == 'managed_load_plan':
        if (binding in operations or len(operations) >= 16 or type(facts.get('models')) is not int
                or not 0 <= facts['models'] <= 64
                or any(type(facts.get(key)) is not bool for key in ('force_patch_weights', 'force_full_load'))):
            missing()
            return True
        operation = {'index': len(operations), 'models': facts['models'], 'selection': set(), 'clones': set(), 'plan': set(),
                     'requirement': set(), 'usage': set()}
        operations[binding] = operation
        prefix = 'managedLoading.' + str(operation['index']) + '.'
        retain('managedLoading.batches', len(operations))
        retain(prefix + 'models', facts['models'])
        retain(prefix + 'forcePatchWeights', facts['force_patch_weights'])
        retain(prefix + 'forceFullLoad', facts['force_full_load'])
        for key, name in (('requested_bytes', 'requestedInferenceBytes'), ('minimum_bytes', 'minimumInferenceBytes')):
            if number(facts.get(key)):
                retain(prefix + name, facts[key], 'measurements')
            else:
                missing()
        return True
    operation = operations.get(binding)
    index = facts.get('index')
    if operation is None or type(index) is not int or not 0 <= index < operation['models']:
        missing()
        return True
    kind = {'managed_load_selection': 'selection', 'managed_load_clones': 'clones', 'managed_load_requirement': 'requirement',
            'managed_load_entry_plan': 'plan', 'managed_load_usage': 'usage'}[event]
    if index in operation[kind]:
        missing()
        return True
    prefix = 'managedLoading.' + str(operation['index']) + '.entry.' + str(index) + '.'
    values, target = {}, 'resolved'
    if kind == 'selection':
        try:
            device = str(facts['device'])
        except Exception:
            device = ''
        if (not 0 < len(device) <= 128 or type(facts.get('reused')) is not bool or type(facts.get('dynamic')) is not bool):
            missing()
            return True
        values = {'managerEntryReused': facts['reused'], 'memoryStrategy': 'dynamic' if facts['dynamic'] else 'standard', 'device': device}
    elif kind == 'clones':
        if type(facts.get('count')) is not int or facts['count'] < 0:
            missing()
            return True
        values = {'cloneDetachesPlanned': facts['count']}
    elif kind == 'plan':
        budget = facts.get('budget_bytes')
        if ('loaded_bytes' not in facts or facts.get('vram_state') not in ('DISABLED', 'NO_VRAM', 'LOW_VRAM', 'NORMAL_VRAM', 'HIGH_VRAM', 'SHARED')
                or facts.get('loaded_bytes') is not None and not number(facts['loaded_bytes'])
                or type(budget) not in (int, float) or not math.isfinite(budget)):
            missing()
            return True
        values = {'vramState': facts['vram_state'], 'managerLoadedBytes': facts.get('loaded_bytes'),
                  'managerLoadedBytesKnown': facts.get('loaded_bytes') is not None, 'weightLoadBudgetBytes': budget}
    else:
        keys = ('bytes',) if kind == 'requirement' else ('ram_bytes', 'vram_bytes')
        if any(not number(facts.get(key)) for key in keys):
            missing()
            return True
        # Pressure can change residency after the requirement pass. Post-load
        # usage and initial requirements are diagnostic, never matching inputs.
        names = ('requiredBytesBeforePressure',) if kind == 'requirement' else ('reportedRamBytes', 'reportedVramBytes')
        values, target = {name: facts[key] for key, name in zip(keys, names)}, 'measurements'
    operation[kind].add(index)
    for key, value in values.items():
        retain(prefix + key, value, target)
    return True
