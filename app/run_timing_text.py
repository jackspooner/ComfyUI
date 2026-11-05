"""Scalar facts from the shared native autoregressive loop."""
import math


def observe_text_metrics(state, event, facts):
    if event not in {'text_generation_plan', 'text_prefill_forward', 'text_decode_forward',
                     'text_generation_progress', 'text_generation_completed'}:
        return False
    sources = state.setdefault('sources', {})
    measurements = state['measurements']
    def reason(value):
        if value not in state['reasons']:
            state['reasons'].append(value)
    if event == 'text_generation_plan':
        state['text_plans'] = state.get('text_plans', 0) + 1
        if state['text_plans'] > 1:
            reason('multiple_text_generation_invocations')
        for key, source in (('inputPositions', 'input_positions'), ('textBatch', 'batch'),
                            ('outputTokenLimit', 'output_limit'), ('kvCapacity', 'kv_capacity')):
            value = facts.get(source)
            if type(value) is int and value >= 0:
                state['resolved'][key] = value
                sources[key] = 'native_text_loop'
        if type(facts.get('prefix_reused')) is bool:
            state['resolved']['prefixReused'] = facts['prefix_reused']
            sources['prefixReused'] = 'native_text_loop'
        if type(facts.get('do_sample')) is bool:
            state['resolved']['textSampling'] = facts['do_sample']
            sources['textSampling'] = 'native_text_loop'
        if isinstance(facts.get('precision'), str):
            state['precision'] = facts['precision'][:128]
    elif event in ('text_prefill_forward', 'text_decode_forward'):
        seconds = facts.get('seconds')
        if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
            prefix = 'prefill' if event == 'text_prefill_forward' else 'decode'
            key = prefix + 'ForwardBoundarySeconds'
            measurements[key] = measurements.get(key, 0) + seconds
            sources[key] = 'native_text_loop'
            key = prefix + ('ForwardsCompleted' if facts.get('succeeded') is True else 'ForwardsFailed')
            measurements[key] = measurements.get(key, 0) + 1
            sources[key] = 'native_text_loop'
    else:
        generated = facts.get('generated')
        if type(generated) is int and generated >= 0:
            measurements['generatedTokens'] = generated
            sources['generatedTokens'] = 'native_text_loop'
            state['progress'] = {'unit': 'generated_tokens', 'completed': generated, 'total': None,
                                 'reliable': state.get('text_plans') == 1}
    return True
