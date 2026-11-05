"""Explicit, prompt-free workload adapters; no node execution or GPU queries."""
import math


NODE_STAGES = {
    'CheckpointLoaderSimple': 'loading', 'CheckpointLoader': 'loading', 'UNETLoader': 'loading',
    'CLIPLoader': 'loading', 'DualCLIPLoader': 'loading', 'TripleCLIPLoader': 'loading',
    'VAELoader': 'loading', 'ControlNetLoader': 'loading', 'DiffControlNetLoader': 'loading',
    'LoraLoader': 'preparation', 'LoraLoaderModelOnly': 'preparation',
    'EmptyLatentImage': 'preparation', 'EmptySD3LatentImage': 'preparation',
    'EmptyHunyuanLatentVideo': 'preparation', 'EmptyHunyuanVideo15Latent': 'preparation',
    'EmptyLTXVLatentVideo': 'preparation', 'LoadImage': 'preparation',
    'PrimitiveInt': 'preparation', 'PrimitiveFloat': 'preparation', 'PrimitiveBoolean': 'preparation',
    'ControlNetApply': 'preparation', 'ControlNetApplyAdvanced': 'preparation',
    'CLIPTextEncode': 'encoding', 'VAEEncode': 'encoding', 'VAEEncodeTiled': 'encoding',
    'VAEEncodeForInpaint': 'encoding', 'KSampler': 'sampling', 'KSamplerAdvanced': 'sampling',
    'SamplerCustom': 'sampling', 'SamplerCustomAdvanced': 'sampling', 'BasicGuider': 'preparation',
    'CFGGuider': 'preparation', 'DualCFGGuider': 'preparation', 'KSamplerSelect': 'preparation',
    'BasicScheduler': 'preparation', 'RandomNoise': 'preparation', 'DisableNoise': 'preparation',
    'VAEDecode': 'decoding', 'VAEDecodeTiled': 'decoding', 'SaveImage': 'publication', 'PreviewImage': 'publication',
}
CONTROLS = {'steps', 'cfg', 'denoise', 'sampler_name', 'scheduler', 'start_at_step', 'end_at_step',
            'add_noise', 'return_with_leftover_noise', 'width', 'height', 'length', 'batch_size',
            'tile_size', 'overlap', 'temporal_size', 'temporal_overlap', 'grow_mask_by',
            'strength_model', 'strength_clip', 'strength', 'start_percent', 'end_percent',
            'weight_dtype', 'type', 'device'}
LATENTS = {'EmptyLatentImage': (8, 1, 4), 'EmptySD3LatentImage': (8, 1, 16),
           'EmptyHunyuanLatentVideo': (8, 4, 16), 'EmptyHunyuanVideo15Latent': (16, 4, 32),
           'EmptyLTXVLatentVideo': (32, 8, 128)}


def links(graph, node):
    return [value[0] for value in node.get('inputs', {}).values()
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph]


def describe(graph, base, resource, observed_at, file_identity, digest, available_at='submission'):
    reasons, ordered, visiting = set(), [], set()

    def visit(node_id):
        if node_id in visiting or node_id in ordered:
            return
        visiting.add(node_id)
        for parent in links(graph, graph[node_id]):
            visit(parent)
        ordered.append(node_id)

    for node_id in graph:
        visit(node_id)

    def scalar(value, seen=()):
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph and value[0] not in seen:
            node = graph[value[0]]
            if node.get('class_type') in ('PrimitiveInt', 'PrimitiveFloat', 'PrimitiveBoolean') and value[1] == 0:
                return scalar(node.get('inputs', {}).get('value'), (*seen, value[0]))
            return None
        if type(value) in (str, bool, int, float) and (not isinstance(value, float) or math.isfinite(value)):
            return value
        return None

    def fact(value, source):
        return {'value': value, 'availableAt': available_at, 'observedAt': observed_at, 'source': source}

    def latent_shape(node_id, seen=()):
        if node_id in seen or node_id not in graph:
            return {}
        node = graph[node_id]
        kind, inputs = node.get('class_type'), node.get('inputs', {})
        if kind in LATENTS:
            spatial, temporal, channels = LATENTS[kind]
            values = {key: scalar(inputs.get(key, default)) for key, default in
                      (('width', None), ('height', None), ('length', 1), ('batch_size', 1))}
            if any(type(value) not in (int, float) or value <= 0 for value in values.values()):
                return {}
            return {'latentWidth': values['width'] // spatial, 'latentHeight': values['height'] // spatial,
                    'latentFrames': (values['length'] - 1) // temporal + 1, 'latentChannels': channels,
                    'batch': values['batch_size']}
        if kind in ('KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'SamplerCustomAdvanced'):
            value = inputs.get('latent_image')
            if isinstance(value, list) and len(value) == 2:
                return latent_shape(value[0], (*seen, node_id))
        return {}

    components, resolved, requested, stages, base_models = [], {}, {}, [], []
    sampling_profiles = set()
    for order, node_id in enumerate(ordered):
        node = graph[node_id]
        kind, inputs = node.get('class_type', ''), node.get('inputs', {})
        stage_kind = NODE_STAGES.get(kind, 'unknown')
        stage_reasons = [] if stage_kind != 'unknown' else ['unsupported_workload_adapter']
        if stage_reasons:
            reasons.add('unsupported_workload_adapter')
        node_facts = {}
        for key in (CONTROLS & inputs.keys()) if stage_kind != 'unknown' else ():
            value = scalar(inputs[key])
            if value is None:
                stage_reasons.append('linked_control_unresolved')
                reasons.add('linked_control_unresolved')
            else:
                node_facts[key] = fact(value, kind + '.' + key)
                requested[node_id + '.' + key] = value
        model_keys = ('ckpt_name', 'unet_name', 'clip_name', 'clip_name1', 'clip_name2', 'clip_name3', 'vae_name', 'lora_name', 'control_net_name')
        for key in model_keys if stage_kind != 'unknown' else ():
            value = inputs.get(key)
            if not isinstance(value, str):
                continue
            identity = file_identity(key, value)
            component_kind = 'lora' if key == 'lora_name' else 'controlnet' if key == 'control_net_name' else 'model'
            if identity is None:
                reasons.add('component_identity_unresolved')
            component_facts = dict(node_facts)
            if component_kind == 'lora':
                names = ('strength_model', 'strength_clip') if kind == 'LoraLoader' else ('strength_model',)
                strengths = [node_facts.get(name, {}).get('value') for name in names]
                component_facts['bypassed'] = fact(None if any(value is None for value in strengths) else all(value == 0 for value in strengths), kind)
                component_facts['applicationMode'] = fact('native_patcher', kind)
            components.append({'kind': component_kind, 'identity': identity, 'role': key, 'order': order, 'features': component_facts})
            if component_kind == 'model':
                base_models.append([key, identity])
        if kind in ('KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'SamplerCustomAdvanced'):
            shape = latent_shape(node_id)
            if not shape:
                reasons.add('latent_shape_unresolved')
                stage_reasons.append('latent_shape_unresolved')
            node_facts.update({key: fact(value, kind + '.latent_image') for key, value in shape.items()})
            sampling_profiles.add('diffusion_video' if shape.get('latentFrames', 1) > 1 else 'diffusion_image')
            # Step callbacks are measured later; configured steps do not assert NFE or CFG branch execution.
            stage_reasons.append('model_evaluations_unmeasured')
            reasons.add('model_evaluations_unmeasured')
        if kind == 'LoadImage':
            # Source paths are deliberately excluded. Processed reference shapes require runtime evidence.
            components.append({'kind': 'reference', 'identity': None, 'role': 'image_input', 'order': order, 'features': {'sourceStage': fact(node_id, kind)}})
            reasons.add('reference_processing_unmeasured')
        if kind in ('ControlNetApply', 'ControlNetApplyAdvanced'):
            stage_reasons.append('control_active_steps_unmeasured')
            reasons.add('control_active_steps_unmeasured')
        for key, value in node_facts.items():
            resolved[node_id + '.' + key] = value
        stage_workload = {'version': 2, 'extractorVersion': 'comfy-explicit-2', 'observedAt': observed_at,
                          'profile': next(iter(sampling_profiles)) if len(sampling_profiles) == 1 else 'unknown',
                          'implementation': {'revision': base['profileRevision'], 'compatibilityId': None},
                          'modelIdentity': None, 'requested': {}, 'resolved': node_facts, 'measurements': {},
                          'components': [], 'complete': not stage_reasons, 'reasons': stage_reasons}
        stages.append({'id': node_id, 'kind': stage_kind, 'dependsOn': sorted(set(links(graph, node))),
                       'resource': resource, 'workload': stage_workload, 'phase': 'queued',
                       'startedOffsetSeconds': None, 'completedOffsetSeconds': None, 'durationSeconds': None,
                       'reused': None, 'reasons': list(stage_reasons)})
    model_identity = digest(base_models) if base_models and all(identity for _, identity in base_models) else None
    evidence = {'version': 2, 'extractorVersion': 'comfy-explicit-2', 'observedAt': observed_at,
                'profile': next(iter(sampling_profiles)) if len(sampling_profiles) == 1 else 'unknown',
                'implementation': {'revision': base['profileRevision'], 'compatibilityId': None},
                'modelIdentity': model_identity, 'requested': requested, 'resolved': resolved, 'measurements': {},
                'components': components, 'complete': not reasons, 'reasons': sorted(reasons)}
    for stage in stages:
        stage['workload']['modelIdentity'] = model_identity
    return evidence, stages


def resolved_inputs(kind, inputs, observed_at):
    if kind not in NODE_STAGES:
        return {}, []
    def fact(value):
        return {'value': value, 'availableAt': 'execution_start', 'observedAt': observed_at, 'source': kind + '.resolved_inputs'}
    facts = {key: fact(value) for key, value in inputs.items() if key in CONTROLS
             and type(value) in (str, bool, int, float) and (not isinstance(value, float) or math.isfinite(value))}
    references = []
    image_inputs = {'VAEEncode': ('pixels',), 'VAEEncodeTiled': ('pixels',), 'VAEEncodeForInpaint': ('pixels', 'mask'),
                    'ControlNetApply': ('image',), 'ControlNetApplyAdvanced': ('image',)}
    for key in image_inputs.get(kind, ()):
        shape = getattr(inputs.get(key), 'shape', ())
        if len(shape) not in (3, 4) or not all(type(value) is int and value > 0 for value in shape):
            continue
        width, height = (shape[-1], shape[-2]) if key == 'mask' else (shape[-2], shape[-3])
        batch = shape[0] if key == 'mask' or len(shape) == 4 else 1
        references.append((key, {'width': fact(width), 'height': fact(height), 'batch': fact(batch)}))
    if kind in ('KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'SamplerCustomAdvanced'):
        for role in ('positive', 'negative'):
            conditioning = inputs.get(role)
            if not isinstance(conditioning, list):
                continue
            facts[role + '.conditioningCount'] = fact(len(conditioning))
            for index, entry in enumerate(conditioning):
                if not isinstance(entry, (list, tuple)) or not entry:
                    continue
                shape = getattr(entry[0], 'shape', ())
                if len(shape) == 3 and all(type(value) is int and value > 0 for value in shape):
                    facts[role + '.' + str(index) + '.encodedTokens'] = fact(shape[-2])
    if kind == 'CLIPTextEncode' and isinstance(inputs.get('text'), str):
        facts['inputCharacters'] = fact(len(inputs['text']))
    return facts, references
