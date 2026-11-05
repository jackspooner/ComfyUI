"""Explicit MoGe boundary metadata; no tensor contents or device synchronization."""
import math

MOGE_NODES = {'LoadMoGeModel': 'loading', 'MoGeInference': 'inference',
              'MoGePanoramaInference': 'inference', 'SaveMoGeDepthPNG': 'publication'}
MOGE_CONTROLS = {'resolution_level', 'fov_x_degrees', 'batch_size', 'force_projection',
                 'apply_mask', 'refine_steps', 'split_resolution', 'merge_resolution'}


def observe_moge_metrics(state, event, facts):
    if event not in ('moge_node_plan', 'moge_model_plan', 'moge_processed_input',
                     'moge_refinement_plan', 'moge_forward', 'moge_load', 'moge_export_input'):
        return False

    def reason(value):
        if value not in state['reasons']:
            state['reasons'].append(value)

    def shape(value, rank):
        return isinstance(value, (tuple, list)) and len(value) == rank and all(type(v) is int and v > 0 for v in value)

    def resolved(prefix, values, availability='live'):
        for key, value in values.items():
            name = prefix + key
            state['resolved'][name] = value
            state.setdefault('sources', {})[name] = 'native_moge'
            state.setdefault('availability', {})[name] = availability

    def axes(prefix, value):
        resolved(prefix, {'axis.' + str(axis): size for axis, size in enumerate(value)})

    if event == 'moge_node_plan':
        state['moge_node_plans'] = state.get('moge_node_plans', 0) + 1
        source, views, batch = facts.get('shape'), facts.get('views'), facts.get('batch_size')
        if (not shape(source, 4) or source[1] != 3 or facts.get('operation') not in ('perspective', 'panorama')
                or type(views) is not int or views <= 0 or type(batch) is not int or batch <= 0):
            reason('moge_node_plan_unavailable')
        else:
            if facts['operation'] == 'perspective' and views != source[0] or facts['operation'] == 'panorama' and source[0] != 1:
                reason('moge_chunk_layout_unresolved')
            state['moge_expected_chunks'] = (views + batch - 1) // batch
            state['moge_expected_views'] = views
            state['moge_chunk_size'] = min(batch, views)
            resolved('moge.', {'operation': facts['operation'], 'inputBatch': source[0], 'inputHeight': source[2],
                              'inputWidth': source[3], 'views': views, 'chunkSize': min(batch, views),
                              'chunks': state['moge_expected_chunks']}, 'execution_start')
            state.setdefault('reference_features', {})['image'] = {
                key: {'value': value, 'availableAt': 'execution_start', 'source': 'native_moge'}
                for key, value in dict(batch=source[0], height=source[2], width=source[3], channels=3).items()}
            if facts['operation'] == 'panorama':
                merge = facts.get('merge_shape')
                if shape(merge, 2):
                    resolved('moge.', {'mergeHeight': merge[0], 'mergeWidth': merge[1]}, 'execution_start')
                else:
                    reason('moge_panorama_merge_shape_unavailable')
            if state['moge_expected_chunks'] > 16:
                reason('moge_invocation_limit')
        if state['moge_node_plans'] != 1:
            reason('multiple_moge_node_invocations')
    elif event == 'moge_export_input':
        state['moge_exports'] = state.get('moge_exports', 0) + 1
        source = facts.get('shape')
        mask = facts.get('mask_shape')
        if shape(source, 3) and shape(mask, 3) and tuple(source) == tuple(mask) and source[0] == 1:
            resolved('mogeExport.', {'batch': source[0], 'height': source[1], 'width': source[2]}, 'execution_start')
            state['moge_export_input_known'] = True
        else:
            reason('moge_export_input_unavailable')
        if state['moge_exports'] != 1:
            reason('multiple_moge_exports')
    else:
        if event == 'moge_load':
            index = state.get('moge_models', 0)
            if state.get('moge_expected_chunks') is not None and index >= state['moge_expected_chunks']:
                reason('moge_chunk_layout_unresolved')
        elif event == 'moge_model_plan':
            index = state.get('moge_models', 0)
            state['moge_models'] = index + 1
            state['moge_current_model'] = index
        else:
            index = state.get('moge_current_model', -1)
        if index < 0:
            reason('moge_model_context_unavailable')
            return True
        if index >= 16:
            reason('moge_invocation_limit')
            return True
        models = state.setdefault('moge_model_work', {})
        model = models.setdefault(index, {'processed': 0, 'refinements': 0})
        prefix = 'mogeModel.' + str(index) + '.'
        if event == 'moge_model_plan':
            source, tokens, passes = facts.get('shape'), facts.get('num_tokens'), facts.get('refine_steps')
            precision = facts.get('precision')
            if (not shape(source, 4) or source[1] != 3 or facts.get('version') not in ('v1', 'v2', 'v3')
                    or type(tokens) is not int or tokens <= 0 or type(passes) is not int or passes < 0
                    or precision not in ('torch.float16', 'torch.float32')):
                reason('moge_model_plan_unavailable')
            else:
                model.update(batch=source[0], passes=passes, version=facts['version'])
                if facts['version'] != 'v3' and passes != 0:
                    reason('moge_refinement_layout_unresolved')
                axes(prefix + 'input.', source)
                resolved(prefix, {'version': facts['version'], 'tokens': tokens, 'refineSteps': passes,
                                  'computePrecision': precision[:128]})
                state['precision'] = precision[:128]
                state.setdefault('sources', {})['executionPrecision'] = 'native_moge'
                for key in ('activation_precision', 'load_device', 'offload_device', 'memory_strategy'):
                    value = facts.get(key)
                    if isinstance(value, str) and value:
                        resolved(prefix, {key: value[:128]})
                    else:
                        reason('moge_execution_regime_unavailable')
                expected_activation = precision if facts['version'] == 'v3' else 'torch.float32'
                if facts.get('activation_precision') != expected_activation or facts.get('memory_strategy') not in ('standard', 'dynamic'):
                    reason('moge_execution_regime_unavailable')
                for key in ('force_projection', 'apply_mask', 'apply_metric_scale'):
                    if type(facts.get(key)) is bool:
                        resolved(prefix, {key: facts[key]})
                    else:
                        reason('moge_execution_options_unavailable')
                fov = facts.get('fov_x')
                if fov is None or type(fov) in (int, float) and math.isfinite(fov):
                    resolved(prefix, {'fovX': fov})
                else:
                    reason('moge_fov_unavailable')
                if passes > 16:
                    reason('moge_refinement_limit')
        elif event == 'moge_processed_input':
            model['processed'] += 1
            source = facts.get('shape')
            if (not shape(source, 4) or source[1] != 3 or source[0] != model.get('batch')
                    or source[2] % 14 or source[3] % 14 or model['processed'] != 1):
                reason('moge_processed_work_unavailable')
            else:
                axes(prefix + 'processed.', source)
                model['processed_known'] = True
                head = facts.get('head_shape')
                if head is not None or model.get('version') == 'v1':
                    if shape(head, 4) and head[0] == model.get('batch') and head[1] == 3:
                        axes(prefix + 'head.', head)
                    else:
                        reason('moge_head_work_unavailable')
        elif event == 'moge_refinement_plan':
            step = model['refinements']
            model['refinements'] += 1
            spatial, points = facts.get('spatial'), facts.get('points')
            conditioning = facts.get('conditioning_shape')
            if step >= 16:
                reason('moge_refinement_limit')
            elif (not shape(spatial, 3) or type(points) is not int or points <= 0 or not shape(conditioning, 4)
                    or conditioning[0] != model.get('batch') or points != model.get('batch', 0) * spatial[0] * spatial[1]):
                reason('moge_refinement_work_unavailable')
            else:
                axes(prefix + 'refine.' + str(step) + '.spatial.', spatial)
                axes(prefix + 'refine.' + str(step) + '.conditioning.', conditioning)
                resolved(prefix + 'refine.' + str(step) + '.', {'points': points})
        else:
            seconds = facts.get('seconds')
            if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
                boundary = 'load' if event == 'moge_load' else 'forward'
                state['measurements'][prefix + boundary + 'Seconds'] = seconds
                state['measurements'][prefix + boundary + 'Succeeded'] = facts.get('succeeded') is True
                for key in (boundary + 'Seconds', boundary + 'Succeeded'):
                    name = prefix + key
                    state.setdefault('sources', {})[name] = 'native_moge'
                    state.setdefault('measurement_availability', {})[name] = 'completion'

    models = state.get('moge_model_work', {})
    count = state.get('moge_models', 0)
    expected = state.get('moge_expected_chunks')
    if expected is not None and count > expected:
        reason('moge_chunk_layout_unresolved')
    if count and count == expected:
        batch = state['moge_chunk_size']
        views = state['moge_expected_views']
        if any(model.get('batch') != min(batch, views - index * batch) for index, model in models.items()):
            reason('moge_chunk_layout_unresolved')
    if any(model.get('passes') is not None and model['refinements'] > model['passes'] for model in models.values()):
        reason('moge_refinement_layout_unresolved')
    state['moge_workload_complete'] = bool(count and count == expected and len(models) == count
        and all(model.get('processed_known') and model['refinements'] == model.get('passes') for model in models.values())
        and not state['reasons'])
    return True
