"""Bind native scalar observations to the existing queue/context owner."""
from comfy import execution_metrics
from comfy_execution.utils import get_executing_context
from app.run_timing_workload import workload_reasons
from app.run_timing_image import image_pipeline_complete
from app.run_timing_clip_vision import clip_vision_plan_complete
from app.run_timing_ltx_audio import ltx_audio_encode_complete, ltx_audio_decode_complete
from app.run_timing_av import av_preparation_complete, av_preparation_execution_complete
from app.run_timing_h3_allocation import h3_allocation_complete
from app.run_timing_h3_layout import h3_layout_complete
from app.run_timing_h3_references import h3_reference_preparation_complete
from app.run_timing_h3_conditioning import h3_conditioning_complete
from app.run_timing_h3_vision import h3_vision_complete
from app.run_timing_h3_vae import h3_video_vae_complete, h3_reference_video_vae_complete
from app.run_timing_h3_audio import h3_audio_complete, h3_reference_audio_complete, scoped_h3_audio_replaces_legacy_limit
from app.run_timing_model_loading import managed_loading_complete
from app.run_timing_refmod import refmod_loader_complete


def apply_metrics(record, stage, state, observed_at, completed=False):
    def fact(value, available, key):
        return {'value': value, 'availableAt': available, 'observedAt': observed_at,
                'source': state.get('sources', {}).get(key, 'native_sampler')}
    workload = stage.get('workload')
    evidence = record['workload'].get('evidence')
    if workload is None or evidence is None:
        return
    if state.get('sampling_profile') == 'diffusion_video':
        workload['profile'] = 'diffusion_video'
        profiles = {item['workload']['profile'] for item in record.get('stages', [])
                    if item.get('workload') and (item['kind'] == 'sampling'
                                               or item['kind'] == 'inference' and item['workload']['profile'] == 'text')}
        evidence['profile'] = next(iter(profiles)) if len(profiles) == 1 else 'unknown'
    for key, value in state.get('resolved', {}).items():
        workload['resolved'][key] = fact(value, state.get('availability', {}).get(key, 'execution_start'), key)
        evidence['resolved'][stage['id'] + '.' + key] = workload['resolved'][key]
    for key, value in state.get('measurements', {}).items():
        available = state.get('measurement_availability', {}).get(key, 'completion' if completed else 'live')
        workload['measurements'][key] = fact(value, available, key)
    live = {key: {'value': value, 'availableAt': 'live', 'observedAt': observed_at, 'source': 'native_lora'}
            for key, value in state.get('live', {}).items()}
    if live:
        workload['resolved'].update(live)
        evidence['resolved'].update({stage['id'] + '.' + key: value for key, value in live.items()})
        order = next(index for index, item in enumerate(record['stages']) if item['id'] == stage['id'])
        for component in workload['components'] + [component for component in evidence['components'] if component['order'] == order]:
            if component['kind'] == 'lora':
                component['features'].update(live)
    if state.get('reference_features'):
        order = next(index for index, item in enumerate(record['stages']) if item['id'] == stage['id'])
        components = workload['components'] + [item for item in evidence['components'] if item['order'] == order]
        for component in components:
            if component['kind'] != 'reference':
                continue
            role = component['role'] if component['role'] == 'image_input' else component['role'].removeprefix(stage['id'] + ':')
            for key, value in state['reference_features'].get(role, {}).items():
                component['features'][key] = {**value, 'observedAt': observed_at}
    if state.get('control_features'):
        order = next(index for index, item in enumerate(record['stages']) if item['id'] == stage['id'])
        for prefix, values in state['control_features'].items():
            role = stage['id'] + ':' + prefix
            origin = next((item for item in record['stages'] if item['id'] == values['sourceStage']), None)
            original = next((item for item in origin['workload']['components'] if item['kind'] == 'controlnet'), None) if origin and origin.get('workload') else None
            features = {key: {'value': value, 'availableAt': 'live', 'observedAt': observed_at, 'source': 'native_controlnet'}
                        for key, value in values.items()}
            component = {'kind': 'controlnet', 'identity': original['identity'] if original else None,
                         'role': role, 'order': order, 'features': features}
            for target in (workload, evidence):
                target['components'] = [item for item in target['components'] if item['role'] != role] + [component]
    if state.get('precision'):
        record['workload']['precision'] = state['precision']
        precision = fact(state['precision'], state.get('availability', {}).get('executionPrecision', 'execution_start'), 'executionPrecision')
        workload['resolved']['executionPrecision'] = precision
        evidence['resolved'][stage['id'] + '.executionPrecision'] = precision
    reasons = set(stage['reasons'])
    if 'refmodLoader.required' in workload['resolved']:
        if refmod_loader_complete(state):
            reasons.discard('refmod_loader_work_unmeasured')
        else:
            reasons.add('refmod_loader_work_unmeasured')
    if 'managedLoading.required' in workload['resolved'] or state.get('managed_load_operations'):
        if managed_loading_complete(state):
            reasons.discard('managed_load_plan_unresolved')
        else:
            reasons.add('managed_load_plan_unresolved')
    if 'h3Layout.required' in workload['resolved']:
        if h3_layout_complete(state):
            reasons.discard('h3_packed_layout_unmeasured')
        else:
            reasons.add('h3_packed_layout_unmeasured')
    if 'h3References.required' in workload['resolved']:
        if h3_reference_preparation_complete(state):
            reasons.discard('h3_reference_preparation_unmeasured')
            if h3_reference_video_vae_complete(state) and h3_reference_audio_complete(state):
                reasons.discard('h3_reference_encoding_work_unmeasured')
            else:
                reasons.add('h3_reference_encoding_work_unmeasured')
        else:
            reasons.add('h3_reference_preparation_unmeasured')
            reasons.add('h3_reference_encoding_work_unmeasured')
    if 'h3AudioVae.required' in workload['resolved']:
        if h3_audio_complete(state):
            reasons.discard('h3_audio_work_unmeasured')
        else:
            reasons.add('h3_audio_work_unmeasured')
    if 'h3VideoVae.required' in workload['resolved']:
        if h3_video_vae_complete(state):
            reasons.discard('h3_video_vae_work_unmeasured')
        else:
            reasons.add('h3_video_vae_work_unmeasured')
    if 'h3Conditioning.required' in workload['resolved']:
        if h3_conditioning_complete(state):
            reasons.discard('h3_conditioning_text_work_unmeasured')
        else:
            reasons.add('h3_conditioning_text_work_unmeasured')
        if h3_conditioning_complete(state) and h3_vision_complete(state):
            reasons.discard('h3_conditioning_vision_work_unmeasured')
        elif state.get('h3_vision_operations') or state['resolved'].get('h3Conditioning.imagePositions') or state['resolved'].get('h3Conditioning.audioPositions'):
            reasons.add('h3_conditioning_vision_work_unmeasured')
    if 'jointAllocation.nativePlanRequired' in workload['resolved']:
        if h3_allocation_complete(state):
            reasons.discard('h3_allocation_workload_unmeasured')
            if completed and (state['measurements'].get('jointAllocation.completed') != 1
                              or state['measurements'].get('jointAllocation.failed', 0)):
                reasons.add('h3_allocation_execution_unresolved')
        else:
            reasons.add('h3_allocation_workload_unmeasured')
    av_operation = workload['resolved'].get('jointPreparation.operation')
    if 'jointPreparation.nativePlanRequired' in workload['resolved']:
        if av_operation and av_preparation_complete(state, av_operation['value']):
            reasons.discard('joint_preparation_workload_unmeasured')
            if completed and not av_preparation_execution_complete(state):
                reasons.add('joint_preparation_execution_unresolved')
        else:
            reasons.add('joint_preparation_workload_unmeasured')
    image_loader = workload['resolved'].get('imageLoader.maskSelectionRequired')
    if image_loader is not None:
        reused = stage['reused'] is True and stage['startedOffsetSeconds'] is None
        if reused or image_pipeline_complete(state, image_loader['value']):
            reasons.discard('reference_processing_unmeasured')
            reuse = {'value': reused, 'availableAt': 'live', 'observedAt': observed_at, 'source': 'native_cache' if reused else 'native_image_loader'}
            workload['resolved']['imageLoader.sourceReused'] = reuse
            evidence['resolved'][stage['id'] + '.imageLoader.sourceReused'] = reuse
        else:
            reasons.add('reference_processing_unmeasured')
    if 'modelEvaluations' in state.get('resolved', {}) and 'cfgBranches' in state['resolved']:
        reasons.discard('model_evaluations_unmeasured')
    if all(key in state.get('resolved', {}) for key in ('latentWidth', 'latentHeight', 'latentFrames')):
        reasons.discard('latent_shape_unresolved')
    if workload['profile'] == 'audio' and 'audioLatentLength' in state.get('resolved', {}):
        reasons.discard('latent_shape_unresolved')
    audio_requirements = {'audio_source_workload_unmeasured': 'audioSource_count',
                          'audio_latent_workload_unmeasured': 'audioPrepared_count',
                          'audio_processed_workload_unmeasured': 'audioProcessed_count',
                          'audio_decode_workload_unmeasured': 'audioDecode_count',
                          'audio_output_unmeasured': 'audioOutput_count'}
    for reason, count in audio_requirements.items():
        if state.get(count) == 1:
            reasons.discard(reason)
    if 'audioInput.durationSeconds' in workload['resolved']:
        reasons.discard('audio_input_unmeasured')
    if state.get('moge_workload_complete'):
        reasons.discard('moge_workload_unmeasured')
    for marker, reason, known, forwards, output in (
        ('ltxAudio.encodePlanRequired', 'ltx_audio_encode_workload_unmeasured', ltx_audio_encode_complete(state),
         ('ltxAudio.encode',), 'ltx_audio_latent_output_count'),
        ('ltxAudio.decodePlanRequired', 'ltx_audio_decode_workload_unmeasured', ltx_audio_decode_complete(state),
         ('ltxAudio.decode', 'ltxAudio.vocoder'), 'ltx_audio_vocoder_input_count'),
    ):
        if marker not in workload['resolved']:
            continue
        if known:
            reasons.discard(reason)
            measured = state['measurements']
            if completed and (state.get(output) != 1 or any(measured.get(name + '.completed') != 1
                                                          or measured.get(name + '.failed', 0) for name in forwards)):
                reasons.add('ltx_audio_execution_unresolved')
        else:
            reasons.add(reason)
    if 'visionEncode.nativePlanRequired' in workload['resolved']:
        if clip_vision_plan_complete(state):
            reasons.discard('clip_vision_workload_unmeasured')
            measured = state['measurements']
            if completed and (measured.get('visionEncodeForwardCompleted') != 1
                              or measured.get('visionEncodeForwardFailed', 0)
                              or measured.get('visionPreprocessCompleted') != 1
                              or state.get('clip_vision_outputs') != 1):
                reasons.add('clip_vision_execution_unresolved')
        else:
            reasons.add('clip_vision_workload_unmeasured')
    if 'imageEncode.nativePlanRequired' in workload['resolved']:
        values, measured = state['resolved'], state['measurements']
        reused = stage['reused'] is True and stage['startedOffsetSeconds'] is None
        known = state.get('vae_inputs') == 1 and state.get('image_encode_plan_complete') and not measured.get('vaeTiledFallbacks')
        known = known and measured.get('vaeTileCountingComplete') is True and values.get('vaeEncode.0.tiled') == values.get('vaeEncode.plan.tiled')
        known = known and all(values.get('vaeEncode.0.axis.' + str(axis)) == values.get('vaeEncode.plan.axis.' + str(axis)) for axis in range(4))
        if reused or known:
            reasons.discard('image_encode_workload_unmeasured')
            reuse = {'value': reused, 'availableAt': 'live', 'observedAt': observed_at, 'source': 'native_cache' if reused else 'native_vae'}
            workload['resolved']['imageEncode.sourceReused'] = reuse
            evidence['resolved'][stage['id'] + '.imageEncode.sourceReused'] = reuse
            if completed and not reused and (measured.get('vaeForwardAttempts') != values['vaeEncode.plan.forwards']
                                            or measured.get('vaeForwardCompleted') != values['vaeEncode.plan.forwards']):
                reasons.add('image_encode_forwards_unresolved')
        else:
            reasons.add('image_encode_workload_unmeasured')
    if state.get('moge_exports') == 1 and state.get('moge_export_input_known'):
        reasons.discard('moge_publication_input_unmeasured')
    if workload['profile'] == 'audio' and state.get('measurements', {}).get('vaeTileCountingComplete') is False:
        reasons.add('audio_tile_work_unmeasured')
    if state.get('text_plans') == 1 and all(key in state.get('resolved', {}) for key in ('inputPositions', 'textBatch', 'outputTokenLimit', 'kvCapacity')):
        reasons.discard('text_loop_coverage_unresolved')
        if completed:
            measured = state.get('measurements', {})
            generated = measured.get('generatedTokens')
            if (type(generated) is not int or measured.get('prefillForwardsCompleted') != 1
                    or measured.get('decodeForwardsCompleted', 0) != max(0, generated - 1)
                    or measured.get('prefillForwardsFailed', 0) or measured.get('decodeForwardsFailed', 0)):
                reasons.add('text_loop_coverage_unresolved')
    if state.get('resolved', {}).get('modalityCoverageComplete') is True:
        reasons.discard('text_input_modalities_unmeasured')
    reasons.update(state.get('reasons', []))
    if scoped_h3_audio_replaces_legacy_limit(state):
        reasons.discard('multiple_audio_invocations')
        reasons.discard('audio_invocation_limit')
    if completed and state.get('measurements', {}).get('noiseEvaluations') != state.get('resolved', {}).get('modelEvaluations'):
        reasons.add('sampling_evaluation_count_unresolved')
    if completed and state.get('resolved', {}).get('cfgBranches') is not None:
        expected = state['resolved'].get('modelEvaluations')
        if expected is not None and state.get('measurements', {}).get('cfgBranchEvaluations') != expected * state['resolved']['cfgBranches']:
            reasons.add('cfg_behavior_unmeasured')
    stage['reasons'] = sorted(reasons)
    input_reasons = workload_reasons(reasons)
    workload.update(complete=not input_reasons, reasons=input_reasons)
    remaining = set(evidence['reasons']) - {'model_evaluations_unmeasured', 'latent_shape_unresolved',
                'text_loop_coverage_unresolved', 'text_input_modalities_unmeasured', 'audio_input_unmeasured',
                'moge_workload_unmeasured', 'moge_publication_input_unmeasured', 'reference_processing_unmeasured',
                'image_encode_workload_unmeasured', 'clip_vision_workload_unmeasured', *audio_requirements}
    remaining -= {'ltx_audio_encode_workload_unmeasured', 'ltx_audio_decode_workload_unmeasured'}
    remaining.discard('joint_preparation_workload_unmeasured')
    remaining.discard('h3_allocation_workload_unmeasured')
    remaining.discard('h3_packed_layout_unmeasured')
    remaining.discard('h3_reference_preparation_unmeasured')
    remaining.discard('h3_reference_encoding_work_unmeasured')
    remaining.discard('h3_conditioning_text_work_unmeasured')
    remaining.discard('h3_conditioning_vision_work_unmeasured')
    remaining.discard('h3_video_vae_work_unmeasured')
    remaining.discard('h3_audio_work_unmeasured')
    remaining.discard('multiple_audio_invocations')
    remaining.discard('audio_invocation_limit')
    remaining.discard('managed_load_plan_unresolved')
    remaining.update(reason for item in record.get('stages', []) for reason in item.get('workload', {}).get('reasons', []))
    input_reasons = workload_reasons(remaining)
    evidence.update(complete=not input_reasons, reasons=input_reasons)
    if state.get('progress') and stage['kind'] in ('sampling', 'inference'):
        stage['progress'] = dict(state['progress'])


def attach_runtime_metrics(queue):
    def observe(event, facts):
        context = get_executing_context()
        if context is not None:
            node_id = execution_metrics.current_stage() or context.node_id
            with queue.mutex:
                if event == 'stage_started':
                    item = next((item for item in queue.currently_running.values() if item[1] == context.prompt_id), None)
                    if item is not None:
                        queue.timing.safe_resolve_started(context.prompt_id, item[2])
                if event == 'stage_inputs':
                    queue.timing.safe_inputs(context.prompt_id, node_id, facts['class_type'], facts['inputs'])
                elif event.startswith('stage_'):
                    action = {'stage_started': 'started', 'stage_finished': 'finished',
                              'stage_failed': 'failed', 'stage_reused': 'reused'}.get(event)
                    if action:
                        queue.timing.safe_stage(context.prompt_id, node_id, action)
                else:
                    queue.timing.safe_metric(context.prompt_id, node_id, event, facts)
    execution_metrics.set_observer(observe)
