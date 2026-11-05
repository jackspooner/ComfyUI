"""Explicit, prompt-free workload adapters; no node execution or GPU queries."""
import math
from app.run_timing_audio import waveform_features
from app.run_timing_vision import MOGE_NODES, MOGE_CONTROLS

EXTRACTOR_VERSION = 'comfy-explicit-7'
IMAGE_LOADERS = {'LoadImage', 'LoadImageOutput', 'LoadImageMask'}
IMAGE_INPUTS = {'VAEEncode': ('pixels',), 'VAEEncodeTiled': ('pixels',), 'VAEEncodeForInpaint': ('pixels', 'mask'),
                'ControlNetApply': ('image',), 'ControlNetApplyAdvanced': ('image',),
                'CLIPVisionEncode': ('image',),
                'MoGeInference': ('image',), 'MoGePanoramaInference': ('image',)}
MEASUREMENT_ONLY_REASONS = frozenset((
    'audio_output_unmeasured', 'audio_output_metadata_unavailable', 'stage_completion_unobserved',
    'image_preparation_measurement_unavailable',
))
AUDIO_NODES = {'LoadAudio': 'preparation', 'EmptyLatentAudio': 'preparation',
               'VAEEncodeAudio': 'encoding', 'VAEDecodeAudio': 'decoding', 'VAEDecodeAudioTiled': 'decoding',
               'SaveAudio': 'publication', 'SaveAudioMP3': 'publication', 'SaveAudioOpus': 'publication',
               'PreviewAudio': 'publication'}


NODE_STAGES = {
    'CheckpointLoaderSimple': 'loading', 'CheckpointLoader': 'loading', 'UNETLoader': 'loading',
    'CLIPLoader': 'loading', 'DualCLIPLoader': 'loading', 'TripleCLIPLoader': 'loading',
    'VAELoader': 'loading', 'ControlNetLoader': 'loading', 'DiffControlNetLoader': 'loading', 'CLIPVisionLoader': 'loading',
    'LoraLoader': 'preparation', 'LoraLoaderModelOnly': 'preparation',
    'EmptyLatentImage': 'preparation', 'EmptySD3LatentImage': 'preparation',
    'EmptyHunyuanLatentVideo': 'preparation', 'EmptyHunyuanVideo15Latent': 'preparation',
    'EmptyLTXVLatentVideo': 'preparation', 'LoadImage': 'preparation',
    'LoadImageOutput': 'preparation', 'LoadImageMask': 'preparation',
    'PrimitiveInt': 'preparation', 'PrimitiveFloat': 'preparation', 'PrimitiveBoolean': 'preparation',
    'ControlNetApply': 'preparation', 'ControlNetApplyAdvanced': 'preparation',
    'CLIPTextEncode': 'encoding', 'CLIPVisionEncode': 'encoding', 'VAEEncode': 'encoding', 'VAEEncodeTiled': 'encoding',
    'VAEEncodeForInpaint': 'encoding', 'KSampler': 'sampling', 'KSamplerAdvanced': 'sampling',
    'SamplerCustom': 'sampling', 'SamplerCustomAdvanced': 'sampling', 'BasicGuider': 'preparation',
    'CFGGuider': 'preparation', 'DualCFGGuider': 'preparation', 'KSamplerSelect': 'preparation',
    'BasicScheduler': 'preparation', 'RandomNoise': 'preparation', 'DisableNoise': 'preparation',
    'VAEDecode': 'decoding', 'VAEDecodeTiled': 'decoding', 'SaveImage': 'publication', 'PreviewImage': 'publication',
    'TextGenerate': 'inference', 'TextGenerateLTX2Prompt': 'inference',
    **AUDIO_NODES,
    **MOGE_NODES,
}
CONTROLS = {'steps', 'cfg', 'denoise', 'sampler_name', 'scheduler', 'start_at_step', 'end_at_step',
            'add_noise', 'return_with_leftover_noise', 'width', 'height', 'length', 'batch_size',
            'tile_size', 'overlap', 'temporal_size', 'temporal_overlap', 'grow_mask_by',
            'strength_model', 'strength_clip', 'strength', 'start_percent', 'end_percent',
            'weight_dtype', 'type', 'device', 'max_length', 'mtp', 'thinking', 'response_format',
            'context_budget', 'fps', 'source_fps', 'auto_resize', 'min_short_edge', 'seconds', 'normalize', 'quality', 'crop'}
LATENTS = {'EmptyLatentImage': (8, 1, 4), 'EmptySD3LatentImage': (8, 1, 16),
           'EmptyHunyuanLatentVideo': (8, 4, 16), 'EmptyHunyuanVideo15Latent': (16, 4, 32),
           'EmptyLTXVLatentVideo': (32, 8, 128)}


def workload_reasons(reasons):
    """Missing outcomes/spans do not change otherwise resolved input work."""
    return sorted(set(reasons) - MEASUREMENT_ONLY_REASONS)


def links(graph, node):
    return [value[0] for value in node.get('inputs', {}).values()
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph]


def describe(graph, base, resource, observed_at, file_identity, digest, available_at='submission', loader_aliases=None,
             implementation_identity=None):
    reasons, ordered, visiting = set(), [], set()

    def node_kind(node):
        declared = node.get('class_type', '')
        native = (loader_aliases or {}).get(declared)
        return native if NODE_STAGES.get(native) == 'loading' else declared

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
        kind, inputs = node_kind(node), node.get('inputs', {})
        if kind == 'EmptyLatentAudio':
            seconds, batch = scalar(inputs.get('seconds')), scalar(inputs.get('batch_size', 1))
            if (type(seconds) not in (int, float) or seconds <= 0 or not math.isfinite(seconds * 44100)
                    or type(batch) is not int or batch <= 0):
                return {}
            length = round((seconds * 44100 / 2048) / 2) * 2
            return {'audioLatentLength': length, 'latentChannels': 64, 'batch': batch} if length > 0 else {}
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
            if isinstance(value, list) and len(value) == 2 and value[1] == 0:
                return latent_shape(value[0], (*seen, node_id))
        return {}

    def audio_latent(node_id, seen=()):
        if node_id in seen or node_id not in graph:
            return False
        node = graph[node_id]
        kind = node_kind(node)
        if kind in ('EmptyLatentAudio', 'VAEEncodeAudio'):
            return True
        value = node.get('inputs', {}).get('latent_image')
        return (kind in ('KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'SamplerCustomAdvanced')
                and isinstance(value, list) and len(value) == 2 and value[1] == 0
                and audio_latent(value[0], (*seen, node_id)))

    components, resolved, requested, stages, base_models = [], {}, {}, [], []
    model_sources, stage_revisions, implementations = {}, {}, {}
    sampling_profiles = set()
    for order, node_id in enumerate(ordered):
        component_start = len(components)
        node = graph[node_id]
        kind, inputs = node_kind(node), node.get('inputs', {})
        stage_kind = NODE_STAGES.get(kind, 'unknown')
        stage_reasons = [] if stage_kind != 'unknown' else ['unsupported_workload_adapter']
        if stage_reasons:
            reasons.add('unsupported_workload_adapter')
        node_facts = {}
        controls = MOGE_CONTROLS if kind in MOGE_NODES else CONTROLS
        for key in (controls & inputs.keys()) if stage_kind != 'unknown' else ():
            value = scalar(inputs[key])
            if kind in MOGE_NODES:
                # Native runtime plans supply effective work; v1/v2, for example,
                # ignore requested refinement passes. Keep those controls audit-only.
                requested[node_id + '.' + key] = value if type(value) in (bool, int, float) else None
                continue
            if value is None:
                stage_reasons.append('linked_control_unresolved')
                reasons.add('linked_control_unresolved')
            else:
                node_facts[key] = fact(value, kind + '.' + key)
                requested[node_id + '.' + key] = value
        model_keys = ('ckpt_name', 'unet_name', 'clip_name', 'clip_name1', 'clip_name2', 'clip_name3', 'vae_name', 'lora_name', 'control_net_name')
        if kind == 'LoadMoGeModel':
            model_keys = ('model_name',)
        for key in model_keys if stage_kind != 'unknown' else ():
            value = inputs.get(key)
            if not isinstance(value, str):
                continue
            selection = 'moge_model_name' if kind == 'LoadMoGeModel' else 'clip_vision_name' if kind == 'CLIPVisionLoader' and key == 'clip_name' else key
            identity = file_identity(selection, value)
            component_kind = 'lora' if key == 'lora_name' else 'controlnet' if key == 'control_net_name' else 'model'
            if identity is None:
                stage_reasons.append('component_identity_unresolved')
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
            sampling_profiles.add('audio' if audio_latent(node_id) else
                                  'diffusion_video' if shape.get('latentFrames', 1) > 1 else 'diffusion_image')
            # Step callbacks are measured later; configured steps do not assert NFE or CFG branch execution.
            stage_reasons.append('model_evaluations_unmeasured')
            reasons.add('model_evaluations_unmeasured')
        if kind in IMAGE_LOADERS:
            # Source paths are deliberately excluded. Processed reference shapes require runtime evidence.
            components.append({'kind': 'reference', 'identity': None, 'role': 'image_input', 'order': order, 'features': {'sourceStage': fact(node_id, kind)}})
            node_facts['imageLoader.maskSelectionRequired'] = fact(kind == 'LoadImageMask', kind)
            stage_reasons.append('reference_processing_unmeasured')
            reasons.add('reference_processing_unmeasured')
        if kind == 'LoadAudio':
            components.append({'kind': 'reference', 'identity': None, 'role': 'audio_input', 'order': order,
                               'features': {'sourceStage': fact(node_id, kind)}})
            stage_reasons.append('audio_source_workload_unmeasured')
        for role in IMAGE_INPUTS.get(kind, ()):
            source = inputs.get(role)
            features = {'sourceStage': fact(source[0], kind)} if isinstance(source, list) and len(source) == 2 and isinstance(source[0], str) and source[0] in graph else {}
            components.append({'kind': 'reference', 'identity': None, 'role': node_id + ':' + role, 'order': order, 'features': features})
            if not features:
                stage_reasons.append('reference_source_unresolved')
        if kind in ('VAEEncode', 'VAEEncodeTiled', 'VAEEncodeForInpaint'):
            node_facts['imageEncode.nativePlanRequired'] = fact(True, kind)
            stage_reasons.append('image_encode_workload_unmeasured')
        if kind == 'CLIPVisionEncode':
            node_facts['visionEncode.nativePlanRequired'] = fact(True, kind)
            stage_reasons.append('clip_vision_workload_unmeasured')
        if kind in ('MoGeInference', 'MoGePanoramaInference'):
            stage_reasons.append('moge_workload_unmeasured')
            if kind == 'MoGePanoramaInference':
                stage_reasons.append('moge_panorama_merge_work_unmeasured')
        if kind == 'SaveMoGeDepthPNG':
            stage_reasons.append('moge_publication_input_unmeasured')
        if kind == 'EmptyLatentAudio':
            shape = latent_shape(node_id)
            node_facts.update({key: fact(value, kind) for key, value in shape.items()})
            if not shape:
                stage_reasons.append('latent_shape_unresolved')
        if kind == 'VAEEncodeAudio':
            stage_reasons.append('audio_processed_workload_unmeasured')
        if kind in ('VAEDecodeAudio', 'VAEDecodeAudioTiled'):
            stage_reasons.extend(('audio_decode_workload_unmeasured', 'audio_output_unmeasured'))
        if kind in AUDIO_NODES and stage_kind == 'publication':
            stage_reasons.append('audio_input_unmeasured')
        if kind in ('TextGenerate', 'TextGenerateLTX2Prompt'):
            sampling_profiles.add('text')
            stage_reasons.extend(('text_loop_coverage_unresolved', 'text_input_modalities_unmeasured'))
            reasons.update(stage_reasons)
        if kind in ('ControlNetApply', 'ControlNetApplyAdvanced'):
            selection = inputs.get('control_net')
            loader = graph.get(selection[0], {}) if isinstance(selection, list) and len(selection) == 2 and isinstance(selection[0], str) else {}
            selected = loader.get('inputs', {}).get('control_net_name') if node_kind(loader) in ('ControlNetLoader', 'DiffControlNetLoader') else None
            identity = file_identity('control_net_name', selected) if isinstance(selected, str) else None
            if identity is None:
                stage_reasons.append('component_identity_unresolved')
            strength = node_facts.get('strength', {}).get('value')
            features = dict(node_facts, bypassed=fact(None if strength is None else strength == 0, kind))
            components.append({'kind': 'controlnet', 'identity': identity, 'role': node_id + ':control',
                               'order': order, 'features': features})
            if strength != 0:
                stage_reasons.append('control_active_steps_unmeasured')
                reasons.add('control_active_steps_unmeasured')
        for key, value in node_facts.items():
            resolved[node_id + '.' + key] = value
        parents = sorted(set(links(graph, node)))
        # Scope identities to reachable work once in dependency order. Keep the
        # whole-run identities below for audit and whole-run comparisons.
        models = {source: values for parent in parents for source, values in model_sources.get(parent, {}).items()}
        own_models = [[component['role'], component['identity']] for component in components[component_start:]
                      if component['kind'] == 'model']
        if own_models:
            models[node_id] = own_models
        model_sources[node_id] = models
        selected_models = sorted((value for values in models.values() for value in values),
                                 key=lambda value: (value[0], value[1] or ''))
        stage_model = digest(selected_models) if selected_models and all(identity for _, identity in selected_models) else None
        stage_revision = base['profileRevision']
        if implementation_identity is not None:
            declared = node.get('class_type', '')
            if declared not in implementations:
                implementations[declared] = implementation_identity(declared)
            if any(parent not in stage_revisions for parent in parents):
                stage_reasons.append('stage_dependency_identity_unresolved')
            dependency_revisions = sorted([key, stage_revisions.get(value[0]), value[1]]
                                          for key, value in inputs.items()
                                          if isinstance(value, list) and len(value) == 2
                                          and isinstance(value[0], str) and value[0] in graph)
            stage_revision = digest(['dependency-stage-1', implementations[declared], sorted(inputs), dependency_revisions])
        stage_revisions[node_id] = stage_revision
        input_reasons = workload_reasons(stage_reasons)
        reasons.update(input_reasons)
        stage_workload = {'version': 2, 'extractorVersion': EXTRACTOR_VERSION, 'observedAt': observed_at,
                          'profile': 'vision' if kind in MOGE_NODES or kind in IMAGE_LOADERS or kind in ('VAEEncode', 'VAEEncodeTiled', 'VAEEncodeForInpaint', 'CLIPVisionLoader', 'CLIPVisionEncode') else 'audio' if kind in AUDIO_NODES or audio_latent(node_id) else
                                     next(iter(sampling_profiles)) if len(sampling_profiles) == 1 else 'unknown',
                          'implementation': {'revision': stage_revision, 'compatibilityId': None},
                          'modelIdentity': stage_model, 'requested': {}, 'resolved': node_facts, 'measurements': {},
                          'components': components[component_start:], 'complete': not input_reasons, 'reasons': input_reasons}
        stages.append({'id': node_id, 'kind': stage_kind, 'dependsOn': parents,
                       'resource': resource, 'workload': stage_workload, 'phase': 'queued',
                       'startedOffsetSeconds': None, 'completedOffsetSeconds': None, 'durationSeconds': None,
                       'reused': None, 'reasons': list(stage_reasons)})
    model_identity = digest(base_models) if base_models and all(identity for _, identity in base_models) else None
    evidence = {'version': 2, 'extractorVersion': EXTRACTOR_VERSION, 'observedAt': observed_at,
                'profile': next(iter(sampling_profiles)) if len(sampling_profiles) == 1 else
                           'audio' if not sampling_profiles and any(node_kind(node) in AUDIO_NODES for node in graph.values()) else
                           'vision' if not sampling_profiles and any(node_kind(node) in MOGE_NODES or node_kind(node) in ('CLIPVisionLoader', 'CLIPVisionEncode') for node in graph.values()) else 'unknown',
                'implementation': {'revision': base['profileRevision'], 'compatibilityId': None},
                'modelIdentity': model_identity, 'requested': requested, 'resolved': resolved, 'measurements': {},
                'components': components, 'complete': not reasons, 'reasons': sorted(reasons)}
    return evidence, stages


def resolved_inputs(kind, inputs, observed_at):
    if kind not in NODE_STAGES:
        return {}, []
    def fact(value):
        return {'value': value, 'availableAt': 'execution_start', 'observedAt': observed_at, 'source': kind + '.resolved_inputs'}
    facts = {key: fact(value) for key, value in inputs.items() if key in CONTROLS
             and type(value) in (str, bool, int, float) and (not isinstance(value, float) or math.isfinite(value))}
    if kind in MOGE_NODES:
        facts = {}
    references = []
    for key in IMAGE_INPUTS.get(kind, ()):
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
    if kind in AUDIO_NODES and isinstance(inputs.get('audio'), dict):
        audio = inputs['audio']
        values = waveform_features(getattr(audio.get('waveform'), 'shape', ()), audio.get('sample_rate'))
        if values is not None:
            facts.update({'audioInput.' + key: fact(value) for key, value in values.items()})
            references.append(('audio', {key: fact(value) for key, value in values.items()}))
    return facts, references
