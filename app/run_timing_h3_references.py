"""Bounded native H3 reference preparation, without tensors or prompt contents."""
import math
import re

MAX_REFERENCES = 32
PREFIXES = {'image': 'ref_images.ref_image_', 'video': 'ref_videos.ref_video_', 'audio': 'ref_audios.ref_audio_'}
TIMERS = {'h3_reference_resize', 'h3_reference_visual_encode', 'h3_reference_audio_encode',
          'h3_reference_tokenize', 'h3_reference_text_encode'}
EVENTS = {'h3_reference_begin', 'h3_reference_source', 'h3_reference_processed', 'h3_reference_output',
          'h3_reference_prepared', 'h3_reference_metadata_unavailable'} | TIMERS


def h3_reference_preparation_complete(state):
    return (state.get('h3_reference_begins') == 1 and state.get('h3_reference_prepared') == 1
            and state.get('h3_reference_ready') is True
            and not any(reason.startswith('h3_reference_') for reason in state['reasons']))


def observe_h3_reference_metrics(state, event, facts):
    if event == 'tensor_shape_unavailable' and facts.get('operation') in EVENTS:
        event = 'h3_reference_metadata_unavailable'
    if event not in EVENTS:
        return False

    def missing(reason='h3_reference_metadata_unresolved'):
        if reason not in state['reasons']:
            state['reasons'].append(reason)

    def retain(key, value, measured=False, available='live'):
        state['measurements' if measured else 'resolved'][key] = value
        state.setdefault('sources', {})[key] = 'native_h3_reference'
        state.setdefault('measurement_availability' if measured else 'availability', {})[key] = available

    def count(key, value=1):
        retain(key, state['measurements'].get(key, 0) + value, measured=True)

    def metadata(names, ranks):
        shapes, precisions, devices = (facts.get(key) for key in ('shapes', 'precisions', 'devices'))
        if any(not isinstance(value, dict) or set(value) != names for value in (shapes, precisions, devices)):
            return None
        if any(not isinstance(shape, (list, tuple)) or len(shape) != ranks[name]
               or any(type(size) is not int or size <= 0 for size in shape) for name, shape in shapes.items()):
            return None
        if any(not isinstance(value, str) or not 0 < len(value) <= 128 for values in (precisions, devices) for value in values.values()):
            return None
        return {name: tuple(shape) for name, shape in shapes.items()}

    if event == 'h3_reference_begin':
        state['h3_reference_begins'] = state.get('h3_reference_begins', 0) + 1
        if state['h3_reference_begins'] != 1:
            missing('h3_reference_multiple_invocations')
        retain('h3References.required', True)
        return True
    if event == 'h3_reference_metadata_unavailable':
        missing()
        return True
    if state.get('h3_reference_begins') != 1:
        missing('h3_reference_correlation_unresolved')
        return True
    entries = state.setdefault('h3_reference_entries', [])
    if event in TIMERS:
        seconds, succeeded = facts.get('seconds'), facts.get('succeeded')
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
                or type(succeeded) is not bool):
            missing()
            return True
        if event not in ('h3_reference_tokenize', 'h3_reference_text_encode'):
            index = facts.get('index')
            if type(index) is not int or not 0 <= index < len(entries):
                missing('h3_reference_correlation_unresolved')
                return True
        prefix = 'h3References.' + event.removeprefix('h3_reference_')
        count(prefix + '.cpuSeconds', seconds)
        count(prefix + ('.completed' if succeeded else '.failed'))
        return True
    if event == 'h3_reference_prepared':
        state['h3_reference_prepared'] = state.get('h3_reference_prepared', 0) + 1
        expected_items = len(entries) + sum(entry['soundtrack'] for entry in entries)
        expected_blocks = sum(entry['visual_encode'] or entry['role'] == 'audio' and entry['audio_encode'] for entry in entries)
        if (state['h3_reference_prepared'] != 1 or type(facts.get('references')) is not int or facts['references'] != len(entries)
                or type(facts.get('tokenizer_items')) is not int or facts['tokenizer_items'] != expected_items
                or type(facts.get('blocks')) is not int or facts['blocks'] != expected_blocks
                or any(not entry['processed'] for entry in entries)):
            missing('h3_reference_preparation_unresolved')
            return True
        state['h3_reference_ready'] = True
        retain('h3References.count', len(entries))
        retain('h3References.tokenizerItems', expected_items)
        retain('h3References.blocks', facts['blocks'], measured=True, available='completion')
        for role in PREFIXES:
            retain('h3References.' + role + '.count', sum(entry['role'] == role for entry in entries))
        return True

    index, role, slot = (facts.get(key) for key in ('index', 'role', 'slot'))
    if state.get('h3_reference_prepared'):
        missing('h3_reference_correlation_unresolved')
        return True
    if (type(index) is not int or not 0 <= index < MAX_REFERENCES or not isinstance(role, str) or role not in PREFIXES
            or not isinstance(slot, str) or not re.fullmatch(re.escape(PREFIXES[role]) + r'\d{1,4}', slot)):
        missing('h3_reference_limit_or_socket_unresolved')
        return True
    prefix = 'h3References.entry.' + str(index) + '.'
    if event == 'h3_reference_source':
        if (index != len(entries) or any(entry['slot'] == slot for entry in entries)
                or entries and list(PREFIXES).index(role) < list(PREFIXES).index(entries[-1]['role'])):
            missing('h3_reference_correlation_unresolved')
            return True
        flags = {key: facts.get(key) for key in ('visual_encode', 'audio_encode', 'soundtrack')}
        if (any(type(value) is not bool for value in flags.values())
                or role == 'image' and (flags['audio_encode'] or flags['soundtrack'])
                or role == 'audio' and (flags['visual_encode'] or flags['soundtrack'])
                or role == 'video' and flags['audio_encode'] and not (flags['visual_encode'] and flags['soundtrack'])):
            missing()
            return True
        names = ({'source'} if role != 'audio' else set()) | ({'waveform'} if role == 'audio' or flags['soundtrack'] else set())
        shapes = metadata(names, {'source': 4, 'waveform': 3})
        if shapes is None:
            missing()
            return True
        if 'source' in shapes:
            shape = shapes['source']
            if shape[3] < 3 or any(type(facts.get(key)) is not int or facts[key] <= 0 for key in ('resize_width', 'resize_height')):
                missing()
                return True
            if role == 'video' and (type(facts.get('frame_limit')) is not int or facts['frame_limit'] < 5):
                missing()
                return True
        if 'waveform' in shapes and (type(facts.get('sample_rate')) is not int or facts['sample_rate'] <= 0):
            missing()
            return True
        entry = dict(role=role, slot=slot, shapes=shapes, processed=role == 'audio', output=False, **flags)
        entries.append(entry)
        retain(prefix + 'role', role)
        state.setdefault('reference_features', {}).setdefault(slot, {}).update({
            key: {'value': value, 'source': 'native_h3_reference', 'availableAt': 'live'}
            for key, value in {'referenceRole': role, 'referenceOrder': index,
                               'visualEncode': flags['visual_encode'], 'audioEncode': flags['audio_encode']}.items()})
        for key, value in flags.items():
            retain(prefix + key, value)
        # Original batch/channel dimensions are audit evidence. Stills resize
        # only their first RGB item; videos resize all incoming frames before trim.
        for name, shape in shapes.items():
            for axis, size in enumerate(shape):
                retain(prefix + 'source.' + name + '.axis.' + str(axis), size, measured=True)
        if 'source' in shapes:
            source = shapes['source']
            entry['width'], entry['height'] = facts['resize_width'], facts['resize_height']
            for key, value in {'source.height': source[1], 'source.width': source[2],
                               'resize.frames': 1 if role == 'image' else source[0],
                               'resize.width': entry['width'], 'resize.height': entry['height'],
                               'resize.channels': 3, 'resize.crop': 'disabled', 'resize.method': 'lanczos',
                               'resize.precision': facts['precisions']['source'], 'resize.device': facts['devices']['source']}.items():
                retain(prefix + key, value)
            if role == 'video':
                entry['frame_limit'] = facts['frame_limit']
        if 'waveform' in shapes:
            waveform = shapes['waveform']
            audio_values = {'audio.batch': 1, 'audio.channels': waveform[1], 'audio.samples': waveform[2],
                            'audio.sampleRate': facts['sample_rate'], 'audio.precision': facts['precisions']['waveform'],
                            'audio.device': facts['devices']['waveform']}
            for key, value in audio_values.items():
                retain(prefix + key, value, measured=not flags['audio_encode'])
            if flags['audio_encode']:
                audio_slot = slot if role == 'audio' else 'ref_video_audios.ref_video_audio_' + slot.rsplit('_', 1)[-1]
                state['reference_features'].setdefault(audio_slot, {}).update({
                    key: {'value': value, 'source': 'native_h3_reference', 'availableAt': 'live'}
                    for key, value in dict(audio_values, referenceOrder=index).items()})
        return True

    if index >= len(entries) or (entries[index]['role'], entries[index]['slot']) != (role, slot):
        missing('h3_reference_correlation_unresolved')
        return True
    entry = entries[index]
    if event == 'h3_reference_processed':
        shapes = metadata({'visual', 'tokenizer'}, {'visual': 4, 'tokenizer': 4})
        if entry['processed'] or shapes is None or role == 'audio':
            missing('h3_reference_preparation_unresolved')
            return True
        frames = 1 if role == 'image' else min(entry['shapes']['source'][0], entry['frame_limit'])
        if role == 'video':
            frames -= (frames - 5) % 17
        expected_visual = (frames, entry['height'], entry['width'], 3)
        expected_tokenizer = (1 if role == 'image' else (frames - 1) // 12 + 1, *expected_visual[1:])
        if frames < 1 or shapes['visual'] != expected_visual or shapes['tokenizer'] != expected_tokenizer:
            missing('h3_reference_preparation_unresolved')
            return True
        entry['processed'] = True
        entry['visual_shape'] = shapes['visual']
        values = {'processedFrames': frames, 'processedWidth': entry['width'], 'processedHeight': entry['height'],
                  'tokenizerFrames': expected_tokenizer[0], 'tokenizerFrameStride': 0 if role == 'image' else 12}
        for name in ('visual', 'tokenizer'):
            values[name + '.precision'] = facts['precisions'][name]
            values[name + '.device'] = facts['devices'][name]
        for key, value in values.items():
            retain(prefix + key, value)
        state.setdefault('reference_features', {}).setdefault(slot, {}).update({
            key: {'value': value, 'source': 'native_h3_reference', 'availableAt': 'live'} for key, value in values.items()})
    elif event == 'h3_reference_output':
        names = ({'visual'} if entry['visual_encode'] else set()) | ({'audio'} if entry['audio_encode'] else set())
        shapes = metadata(names, {'visual': 5, 'audio': 4})
        if not names or shapes is None or entry['output'] or not entry['processed']:
            missing('h3_reference_output_unresolved')
            return True
        entry['output'] = True
        for name, shape in shapes.items():
            for axis, size in enumerate(shape):
                retain(prefix + 'output.' + name + '.axis.' + str(axis), size, measured=True, available='completion')
            for key in ('precisions', 'devices'):
                retain(prefix + 'output.' + name + '.' + key, facts[key][name], measured=True, available='completion')
    return True
