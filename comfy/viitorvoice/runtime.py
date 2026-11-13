"""Sentence-edit planning, native generation and original-PCM splicing."""
import json
import math

import torch
from torch.nn import functional as F
import torchaudio

import comfy.model_management as mm
import comfy.model_prefetch as prefetch
import comfy.ops
from comfy.audio_codec import AudioCodes, AudioCodeSegment, DUALCODEC_V1
from comfy.console_progress import ConsoleProgress, advance
from .alignment import expand_items_to_chars, normalize_alignment_result
from .edit_plan import build_multi_local_edit_token_plan
from .sampling import generate as refine
from .text import normalize_nvv_tags, resolve_language
from . import text_diff


def prepare_audio(audio):
    waveform = audio['waveform']
    rate = audio['sample_rate']
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] == 0 or not waveform.is_floating_point() or not torch.isfinite(waveform).all():
        raise ValueError('ViiTor source_audio requires one finite floating-point [1, channel, sample] recording')
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise ValueError('ViiTor source_audio sample rate must be a positive integer')
    value = torchaudio.functional.resample(waveform.detach().cpu().float().mean(1, keepdim=True), rate, 24000)
    value = F.pad(value, (7200, 7200))
    if value.shape[-1] > 720000:
        raise ValueError('ViiTor source audio plus its 600 millisecond context padding exceeds the 30 second encoder limit')
    return dict(waveform=value, sample_rate=24000)


def plan(codes, alignment_json, original, edited, language, granularity, padding_ms, expand_mask_ratio, length_scale):
    codes.validate(DUALCODEC_V1)
    if len(codes.segments) != 1 or codes.segments[0].codes.shape[:2] != (1, 12):
        raise ValueError('ViiTor requires one source recording with all twelve DualCodec codebooks')
    original, edited = normalize_nvv_tags(original.strip()), normalize_nvv_tags(edited.strip())
    if not original or not edited:
        raise ValueError('ViiTor requires complete original and edited transcripts')
    if granularity not in ('auto', 'word', 'character') or not 0 <= padding_ms <= 2000 or not .25 <= length_scale <= 4:
        raise ValueError('Unsupported ViiTor edit granularity, padding or length scale')
    alignment = json.loads(alignment_json)
    if alignment.get('warnings'):
        raise ValueError('VIITOR_ALIGNMENT_INCOMPLETE: '+json.dumps(alignment['warnings'], ensure_ascii=False))
    items = normalize_alignment_result(alignment['segments'], original)
    if any(item.start_char is None or item.end_char is None for item in items):
        raise ValueError('VIITOR_ALIGNMENT_INCOMPLETE: timestamps could not be mapped to the source transcript')
    language = resolve_language(language)
    if granularity == 'character' or (granularity == 'auto' and not text_diff._should_use_word_diff(language=language, text=original)):
        items = expand_items_to_chars(items, original)
    diff = text_diff._build_text_diff(original_text=original, edited_text=edited, language=language)
    groups = text_diff._merge_overlapping_index_groups([
        text_diff._alignment_indices_for_char_span(items, start, end) for start, end in diff.changed_spans])
    replacements = []
    for group in groups:
        start, end = text_diff._char_span_for_alignment_indices(items, group)
        left, right = text_diff._edited_span_for_original_span(diff.opcodes, start, end, original_text=original)
        replacements.append(edited[left:right])
    segment = codes.segments[0]
    source = segment.codes[0, :, :segment.lengths[0]].clone()
    result = build_multi_local_edit_token_plan(source_tokens=source, alignment_items=items, original_text=original,
        selection=';'.join(','.join(map(str, group)) for group in groups), replacement_text=replacements,
        audio_duration=source.shape[-1]/25, padding_ms=padding_ms, expand_mask_ratio=expand_mask_ratio,
        length_scale=length_scale, min_edit_frames=6, reference_context_frames=300)
    return dict(plan=result, alignments=[item.to_dict() for item in items], language=language,
                original_text=original, edited_text=edited, align_granularity=granularity)


def generate(component, prepared, steps, cfg, seed):
    edit = prepared['plan']
    config = component.config
    masks = torch.tensor(config['audio_mask_ids'])[:, None]
    target = torch.where(edit.editable_audio_mask[None], masks, edit.target_tokens)
    from .sampling import conditioning
    ids, _ = conditioning(component.tokenizer, config, edit.span.edited_full_text,
        target, edit.reference_tokens, prepared['language'])
    length, branches = ids.shape[-1], 2 if cfg else 1
    if length > config['llm_config']['max_position_embeddings']:
        raise ValueError('ViiTor edit and reference exceed the pinned text/audio context')
    memory = branches*(4*16*length*length*4+length*(sum(config['audio_codebook_sizes'])+16*1024)*4)+256*1024**2
    device = component.patcher.load_device
    try:
        with ConsoleProgress('ViiTorVoice', 'masked refinement', total=steps, unit='steps'):
            with prefetch.pause_malloc_graph():
                mm.load_models_gpu([component.patcher], memory_required=memory)
            with mm.cuda_device_context(device), comfy.ops.use_quantized_matmul(component.patcher.model, device):
                raw = refine(component.patcher.model, component.tokenizer, config, target.to(device),
                    edit.editable_audio_mask, edit.reference_tokens, edit.span.edited_full_text, prepared['language'],
                    steps, cfg, seed, component.options(), lambda current, total: advance())
    finally:
        prefetch.cleanup_prefetch_queues()
    keep = torch.ones(raw.shape[-1], dtype=torch.bool)
    for special in ('audio_pause_ids', 'audio_separator_ids'):
        keep &= ~(raw == torch.tensor(config[special])[:, None]).all(0)
    valid = raw[:, keep][None].contiguous()
    codes = AudioCodes(DUALCODEC_V1, (AudioCodeSegment(valid, (valid.shape[-1],)),))
    codes.validate(DUALCODEC_V1)
    return codes, dict(prepared, keep=keep, seed=seed)


def mask_ranges(mask):
    edges = torch.diff(F.pad(mask.to(torch.int8), (1, 1)))
    return list(zip(torch.where(edges == 1)[0].tolist(), torch.where(edges == -1)[0].tolist()))


def source_frame(edit, position):
    old_cursor = new_cursor = 0
    for segment, frames in zip(edit.span.segments, edit.replacement_frames):
        start = new_cursor+segment.start_frame-old_cursor
        if position <= start:
            return old_cursor+position-new_cursor
        if position < start+frames:
            return segment.start_frame+(position-start)*segment.old_frames/frames
        old_cursor, new_cursor = segment.end_frame, start+frames
    return old_cursor+position-new_cursor


def splice(source_audio, decoded, generated):
    """Replace only declared mask regions; copy original channels/PCM elsewhere."""
    original = source_audio['waveform'].detach().cpu()
    rate = source_audio['sample_rate']
    if original.ndim != 3 or original.shape[0] != 1:
        raise ValueError('ViiTor splicing requires the original single-recording AUDIO input')
    edited = generated['plan']
    keep = generated['keep']
    if keep.shape != edited.editable_audio_mask.shape:
        raise ValueError('ViiTor structural-frame mapping differs from the edit draft')
    wave = decoded['waveform'].detach().cpu()
    if decoded['sample_rate'] != 24000 or wave.shape[:2] != (1, 1):
        raise ValueError('ViiTor splice requires decoded mono 24 kHz DualCodec audio')
    pieces, regions, cursor, output_cursor = [], [], 0, 0
    retained = F.pad(keep.to(torch.int64).cumsum(0), (1, 0))
    for start, end in mask_ranges(edited.editable_audio_mask):
        # Alignment and codec time include the pinned synthetic 300 ms edges.
        left = max(0, min(original.shape[-1], math.floor((source_frame(edited, start)/25-.3)*rate)))
        right = max(left, min(original.shape[-1], math.ceil((source_frame(edited, end)/25-.3)*rate)))
        if left < cursor:
            raise ValueError('ViiTor expanded edit regions overlap after restoring source coordinates')
        unchanged = original[..., cursor:left]
        pieces.append(unchanged)
        output_cursor += unchanged.shape[-1]
        first, last = int(retained[start])*960, int(retained[end])*960
        replacement = wave[..., first:min(last, wave.shape[-1])]
        if replacement.shape[-1]:
            replacement = torchaudio.functional.resample(replacement.float(), 24000, rate)
            replacement = replacement.to(original.dtype).repeat(1, original.shape[1], 1)
            # Blend inside the declared replacement region only. Prefix/suffix
            # tensors remain exact views of the original PCM at every rate.
            blend = min(round(.01*rate), replacement.shape[-1]//2, (right-left)//2)
            if blend:
                ramp = torch.linspace(0, 1, blend, dtype=original.dtype)[None, None]
                replacement[..., :blend] = original[..., left:left+blend]*(1-ramp)+replacement[..., :blend]*ramp
                replacement[..., -blend:] = replacement[..., -blend:]*(1-ramp)+original[..., right-blend:right]*ramp
        else:
            replacement = original[..., :0]
        pieces.append(replacement)
        regions.append(dict(source_start_sample=left, source_end_sample=right,
            output_start_sample=output_cursor, output_end_sample=output_cursor+replacement.shape[-1], blend_samples=blend if replacement.shape[-1] else 0))
        output_cursor += replacement.shape[-1]
        cursor = right
    pieces.append(original[..., cursor:])
    result = dict(waveform=torch.cat(pieces, -1), sample_rate=rate)
    metadata = dict(backend='comfyui-native-viitorvoice', seed=generated['seed'], language=generated['language'],
        align_granularity=generated['align_granularity'],
        changed=True, remaining_mask_tokens=0, alignments=generated['alignments'], edits=regions,
        original_text=generated['original_text'], edited_text=generated['edited_text'], sample_rate=str(rate),
        duration_seconds=str(result['waveform'].shape[-1]/rate), trace_id='',
        runtime=dict(backend='comfyui-native-viitorvoice', running=False))
    return result, json.dumps(metadata, ensure_ascii=False, indent=2)
