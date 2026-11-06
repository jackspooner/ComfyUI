"""Native H3 full-sequence text work; vision preprocessing remains separate."""
import math


def h3_conditioning_complete(state):
    return (state.get('h3_conditioning_plans') == 1 and state.get('h3_conditioning_ready') is True
            and not any(reason.startswith('h3_conditioning_') for reason in state['reasons']))


def observe_h3_conditioning_metrics(state, event, facts):
    if event == 'tensor_shape_unavailable' and facts.get('operation') == 'h3_conditioning_plan':
        event = 'h3_conditioning_metadata_unavailable'
    if event not in ('h3_conditioning_plan', 'h3_conditioning_metadata_unavailable', 'h3_conditioning_forward'):
        return False

    def missing(reason='h3_conditioning_metadata_unresolved'):
        if reason not in state['reasons']:
            state['reasons'].append(reason)

    def retain(key, value, measured=False):
        state['measurements' if measured else 'resolved'][key] = value
        state.setdefault('sources', {})[key] = 'native_h3_conditioning'
        state.setdefault('measurement_availability' if measured else 'availability', {})[key] = 'live'

    if event == 'h3_conditioning_metadata_unavailable':
        missing()
        return True
    if event == 'h3_conditioning_forward':
        seconds, succeeded = facts.get('seconds'), facts.get('succeeded')
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0 or type(succeeded) is not bool:
            missing()
            return True
        if state.get('h3_conditioning_plans') != 1:
            missing('h3_conditioning_correlation_unresolved')
        for suffix, value in (('cpuSeconds', seconds), ('completed' if succeeded else 'failed', 1)):
            key = 'h3Conditioning.forward.' + suffix
            retain(key, state['measurements'].get(key, 0) + value, measured=True)
        return True

    # Save the loading requirement in the same snapshot as the forward plan.
    # A resolved sequence length cannot qualify an unobserved managed load.
    retain('managedLoading.required', True)
    retain('h3Conditioning.required', True)
    state['h3_conditioning_plans'] = state.get('h3_conditioning_plans', 0) + 1
    rows = state.pop('embedding_rows', None)
    if state['h3_conditioning_plans'] != 1:
        missing('h3_conditioning_multiple_invocations')
        return True
    shapes, precisions, devices = (facts.get(key) for key in ('shapes', 'precisions', 'devices'))
    if (facts.get('native') is not True or facts.get('input_kind') != 'embeddings'
            or any(not isinstance(value, dict) or set(value) != {'input'} for value in (shapes, precisions, devices))):
        missing()
        return True
    shape = shapes['input']
    if (not isinstance(shape, (tuple, list)) or len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape)
            or any(not isinstance(value, str) or not 0 < len(value) <= 128
                   for value in (precisions['input'], devices['input'], facts.get('compute_precision'), facts.get('output_precision')))
            or facts['compute_precision'] not in ('torch.float32', 'torch.float16', 'torch.bfloat16')
            or type(facts.get('layers')) is not int or facts['layers'] <= 0
            or type(facts.get('layer_norm')) is not bool
            or facts.get('attention_mask') is not False or facts.get('extra_options') is not False):
        missing()
        return True
    indices, selection = facts.get('indices'), facts.get('selection')
    if (not isinstance(indices, (tuple, list)) or len(indices) > 64 or any(type(value) is not int for value in indices)
            or selection not in ('last', 'indices', 'all') or selection != 'indices' and len(indices)
            or selection == 'indices' and not indices):
        missing('h3_conditioning_selection_unresolved')
        return True
    keys = ('positions', 'text', 'image', 'audio', 'unknown', 'padding')
    if (not isinstance(rows, list) or len(rows) != shape[0] or not 0 < len(rows) <= 128
            or any(not isinstance(row, dict) or any(type(row.get(key)) is not int or row[key] < 0 for key in keys)
                   or row['positions'] != shape[1] or sum(row[key] for key in keys[1:]) != row['positions']
                   or row['unknown'] for row in rows)):
        missing('h3_conditioning_modalities_unresolved')
        return True
    values = {'batch': shape[0], 'positions': shape[1], 'width': shape[2],
              'inputPrecision': precisions['input'], 'inputDevice': devices['input'],
              'computePrecision': facts['compute_precision'], 'outputPrecision': facts['output_precision'],
              'layers': facts['layers'], 'selection': selection, 'layerNorm': facts['layer_norm'],
              'regime': 'full_sequence', 'selectedLayers': len(indices)}
    for index, value in enumerate(indices):
        values['selectedLayer.' + str(index)] = value
    for key in keys[1:]:
        values[key + 'Positions'] = sum(row[key] for row in rows)
    for key, value in values.items():
        retain('h3Conditioning.' + key, value)
    state['h3_conditioning_ready'] = True
    return True
