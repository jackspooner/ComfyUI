"""Explicit native image-input preparation; scalar metadata only."""
import math


IMAGE_EVENTS = {'video_decode_work', 'image_pillow_work', 'image_loaded', 'image_mask_selected',
                'image_source_decode', 'image_output_prepare', 'image_mask_prepare'}
VIDEO_COUNTS = {'videoFrames', 'videoPixels', 'processedFrames', 'processedPixels', 'convertedPixels',
                'alignmentPixels', 'rotatedFrames', 'croppedFrames', 'resizedFrames', 'audioFrames', 'audioSamples',
                'resampledAudioFrames', 'resampledAudioSamples', 'decodeErrors', 'alphaFrames', 'retainedAudioSamples'}
VIDEO_DETAILS = {'videoWidth', 'videoHeight', 'videoBitDepth', 'audioRate', 'audioChannels'}
VIDEO_TEXT = {'videoFormat', 'videoCodec', 'conversionFormat', 'audioFormat', 'audioCodec', 'decoderVersion'}


def observe_image_metrics(state, event, facts):
    if event not in IMAGE_EVENTS:
        return False
    sources, availability = state.setdefault('sources', {}), state.setdefault('availability', {})

    def resolved(prefix, values):
        for key, value in values.items():
            name = prefix + key
            state['resolved'][name] = value
            sources[name], availability[name] = 'native_image_loader', 'live'

    def reference(values):
        state.setdefault('reference_features', {}).setdefault('image_input', {}).update({
            key: {'value': value, 'source': 'native_image_loader', 'availableAt': 'live'} for key, value in values.items()})

    def shape(value, rank):
        return isinstance(value, (tuple, list)) and len(value) == rank and all(type(size) is int and size > 0 for size in value)

    if event in ('image_source_decode', 'image_output_prepare', 'image_mask_prepare'):
        name = event + ('.' + facts.get('decoder', 'unknown') if event == 'image_source_decode' else '')
        seconds = facts.get('seconds')
        counts = state.setdefault('image_boundaries', {})
        counts[name] = counts.get(name, 0) + 1
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0 or facts.get('succeeded') is not True:
            state['reasons'].append('image_preparation_measurement_unavailable')
        else:
            state['measurements'][name + '.seconds'] = state['measurements'].get(name + '.seconds', 0) + seconds
            sources[name + '.seconds'] = 'native_image_loader'
    elif event == 'video_decode_work':
        state['image_video_count'] = state.get('image_video_count', 0) + 1
        valid = type(facts.get('version')) is int and facts['version'] == 1 and all(type(facts.get(key)) is int and facts[key] >= 0 for key in VIDEO_COUNTS)
        valid = valid and all(type(value) is int and value > 0 for key, value in facts.items() if key in VIDEO_DETAILS)
        valid = valid and all(isinstance(value, str) and 0 < len(value) <= 128 for key, value in facts.items() if key in VIDEO_TEXT)
        valid = valid and facts.get('fullDecode') is True and type(facts.get('audioStream')) is bool
        if facts.get('videoFrames', 0):
            valid = valid and all(key in facts for key in ('videoWidth', 'videoHeight', 'videoBitDepth', 'videoFormat', 'conversionFormat'))
        if facts.get('audioFrames', 0):
            valid = valid and all(key in facts for key in ('audioRate', 'audioChannels', 'audioFormat'))
        valid = valid and all(key in facts for key in ('videoCodec', 'decoderVersion')) and (not facts.get('audioStream') or 'audioCodec' in facts)
        if valid:
            values = {key: value for key, value in facts.items() if key in VIDEO_COUNTS | VIDEO_DETAILS | VIDEO_TEXT | {'fullDecode', 'audioStream'}}
            resolved('imageDecode.', values)
            reference({'decode.' + key: value for key, value in values.items()})
        state['image_video_complete'] = valid and facts.get('complete') is True and facts['decodeErrors'] == 0
    elif event == 'image_pillow_work':
        state['image_pillow_count'] = state.get('image_pillow_count', 0) + 1
        counts = ('frames', 'decodedPixels', 'selectedFrames', 'alphaFrames', 'sourceWidth', 'sourceHeight')
        valid = type(facts.get('version')) is int and facts['version'] == 1 and all(type(facts.get(key)) is int and facts[key] >= 0 for key in counts)
        valid = valid and all(isinstance(facts.get(key), str) and 0 < len(facts[key]) <= 128 for key in ('format', 'mode'))
        if valid:
            values = {key: facts[key] for key in (*counts, 'format', 'mode')}
            resolved('imagePillow.', values)
            reference({'pillow.' + key: value for key, value in values.items()})
        state['image_pillow_complete'] = valid and facts.get('consistent') is True and facts['frames'] == facts['selectedFrames']
        state['image_pillow_complete'] = state['image_pillow_complete'] and facts['format'] != 'unknown'
    elif event == 'image_loaded':
        dimensions = facts.get('shape')
        if not shape(dimensions, 4):
            state['reasons'].append('reference_shape_unavailable')
            return True
        reference(dict(batch=dimensions[0], height=dimensions[1], width=dimensions[2], channels=dimensions[3]))
        state['image_outputs'] = state.get('image_outputs', 0) + 1
        mask = facts.get('mask_shape')
        valid = type(facts.get('pipeline_version')) is int and facts['pipeline_version'] == 1 and shape(mask, 3) and mask[0] == dimensions[0] and type(facts.get('alpha')) is bool
        valid = valid and facts.get('loader') in ('pyav', 'pillow') and all(
            isinstance(facts.get(key), str) and 0 < len(facts[key]) <= 128 for key in ('precision', 'device'))
        state['image_output_complete'] = valid
        if valid:
            values = dict(loader=facts['loader'], outputBatch=dimensions[0], outputHeight=dimensions[1], outputWidth=dimensions[2],
                          outputChannels=dimensions[3], maskBatch=mask[0], maskHeight=mask[1], maskWidth=mask[2],
                          alpha=facts['alpha'], outputPrecision=facts['precision'], outputDevice=facts['device'])
            resolved('imageLoader.', values)
            reference({'output.' + key: value for key, value in values.items()})
    elif event == 'image_mask_selected':
        state['image_mask_outputs'] = state.get('image_mask_outputs', 0) + 1
        valid = shape(facts.get('shape'), 3) and facts.get('channel') in ('alpha', 'red', 'green', 'blue')
        valid = valid and facts.get('mode') in ('alpha_mask', 'channel_clone', 'empty')
        state['image_mask_complete'] = valid
        if valid:
            batch, height, width = facts['shape']
            values = dict(channel=facts['channel'], mode=facts['mode'], batch=batch, height=height, width=width)
            resolved('imageMask.', values)
            reference({'mask.' + key: value for key, value in values.items()})
    return True


def image_pipeline_complete(state, mask_required):
    values = state['resolved']
    if state.get('image_outputs') != 1 or not state.get('image_output_complete') or state.get('image_video_count') != 1 or not state.get('image_video_complete'):
        return False
    pixels = values['imageLoader.outputBatch'] * values['imageLoader.outputHeight'] * values['imageLoader.outputWidth']
    if values['imageLoader.loader'] == 'pyav':
        if values['imageDecode.processedFrames'] != values['imageLoader.outputBatch'] or values['imageDecode.processedPixels'] != pixels:
            return False
        if values['imageDecode.videoFrames'] != values['imageDecode.processedFrames'] or values['imageDecode.videoPixels'] != pixels or values['imageDecode.convertedPixels'] != pixels:
            return False
        if values['imageDecode.croppedFrames'] or values['imageDecode.resizedFrames']:
            return False
        alpha_frames = values['imageDecode.alphaFrames']
    elif state.get('image_pillow_count') != 1 or not state.get('image_pillow_complete'):
        return False
    elif values['imagePillow.selectedFrames'] != values['imageLoader.outputBatch'] or values['imageDecode.processedFrames'] != 0:
        return False
    else:
        if values['imagePillow.decodedPixels'] != pixels or pixels != values['imagePillow.frames'] * values['imagePillow.sourceWidth'] * values['imagePillow.sourceHeight']:
            return False
        alpha_frames = values['imagePillow.alphaFrames']
    if alpha_frames not in (0, values['imageLoader.outputBatch']) or values['imageLoader.alpha'] != bool(alpha_frames):
        return False
    mask_shape = (values['imageLoader.maskHeight'], values['imageLoader.maskWidth'])
    if mask_shape != ((values['imageLoader.outputHeight'], values['imageLoader.outputWidth']) if alpha_frames else (64, 64)):
        return False
    if not mask_required:
        return True
    if state.get('image_mask_outputs') != 1 or not state.get('image_mask_complete'):
        return False
    alpha_mask = values['imageMask.channel'] == 'alpha'
    if (values['imageMask.mode'] == 'alpha_mask') != alpha_mask:
        return False
    expected = mask_shape if alpha_mask else (values['imageLoader.outputHeight'], values['imageLoader.outputWidth'])
    return (values['imageMask.batch'], values['imageMask.height'], values['imageMask.width']) == (values['imageLoader.outputBatch'], *expected)
