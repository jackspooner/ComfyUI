import copy

import pytest

from app.run_timing_h3_vae import (MAX_OPERATIONS, MAX_CLIPS, MAX_FORWARDS, h3_video_vae_complete,
    h3_reference_video_vae_complete, observe_h3_vae_metrics)
from app.run_timing_h3_references import observe_h3_reference_metrics
from app.run_timing_h3_audio import h3_reference_audio_complete
from app_test.run_timing_h3_references_test import metadata, source, processed, start, latest


def state():
    return {'resolved': {}, 'measurements': {}, 'reasons': []}


def events(binding, frames=1, reference=0, tiling=True):
    shape = (1, 3, frames, 32, 32)
    result = [('h3_vae_begin', metadata({'pixels': shape}, binding=binding, reference_index=reference,
        native=True, tiling=tiling, tile_size=48, tile_overlap_min=16, clip_length=17, token_drop=3,
        spatial_ratio=16, temporal_ratio=4, tokens_chunk_size=5, device='cpu',
        training=False, grad_enabled=False, inference_mode=True))]
    for _ in range(1 if frames == 1 else (frames + 16) // 17):
        clip = (1, 3, 1 if frames == 1 else 17, 32, 32)
        result.append(('h3_vae_clip', metadata({'pixels': clip}, binding=binding)))
        if tiling:
            result.append(('h3_vae_tiles', dict(binding=binding, y_starts=(0,), y_lengths=(32,), y_overlaps=(),
                          x_starts=(0,), x_lengths=(32,), x_overlaps=())))
        result.extend([('h3_vae_forward_input', metadata({'pixels': clip}, binding=binding)),
                       ('h3_vae_forward', dict(binding=binding, seconds=.01, succeeded=True))])
    result.extend([('h3_vae_encode', dict(binding=binding, seconds=.1, succeeded=True)),
        ('h3_vae_output', metadata({'latents': (1, 24, 1 if frames == 1 else (frames + 16) // 17 * 5 - 3, 2, 2)}, binding=binding))])
    return result


@pytest.mark.parametrize('frames,tiling', [(1, True), (22, True), (22, False)])
def test_complete_plans_keep_actual_chunks_and_forwards_live_and_outcomes_measured(frames, tiling):
    value = state()
    for event, facts in events(object(), frames, tiling=tiling):
        observe_h3_vae_metrics(value, event, facts)
    assert h3_video_vae_complete(value)
    assert value['resolved']['h3VideoVae.calls'] == 1
    assert value['resolved']['managedLoading.required'] is True
    assert value['resolved']['h3VideoVae.0.clip.0.axis.2'] == (1 if frames == 1 else 17)
    assert value['resolved']['h3VideoVae.0.forward.0.shape'] == f'1,3,{1 if frames == 1 else 17},32,32'
    assert not any('.output.' in key or 'cpuSeconds' in key or 'tile_size' in key for key in value['resolved'])
    assert value['measurement_availability']['h3VideoVae.0.output.axis.2'] == 'completion'
    assert ('h3VideoVae.0.clip_length' in value['resolved']) is (frames != 1)


@pytest.mark.parametrize('mutate', [lambda f: f.update(native=False), lambda f: f.update(tiling=None),
    lambda f: f.update(tokens_chunk_size=6), lambda f: f.update(reference_index=33),
    lambda f: f['precisions'].update(pixels='unknown'), lambda f: f.update(training=None)])
def test_unknown_configurations_do_not_qualify(mutate):
    value = state()
    sequence = events(object())
    mutate(sequence[0][1])
    for event, facts in sequence:
        observe_h3_vae_metrics(value, event, facts)
    assert not h3_video_vae_complete(value) and value['reasons']


@pytest.mark.parametrize('omit', ['h3_vae_clip', 'h3_vae_tiles', 'h3_vae_forward_input', 'h3_vae_forward', 'h3_vae_encode', 'h3_vae_output'])
def test_missing_phase_evidence_cannot_become_complete(omit):
    value = state()
    for event, facts in events(object()):
        if event != omit:
            observe_h3_vae_metrics(value, event, facts)
    assert not h3_video_vae_complete(value)


def test_tile_geometry_and_forward_shape_must_agree():
    for event_name, key, changed in [('h3_vae_tiles', 'x_starts', (1,)),
                                     ('h3_vae_tiles', 'y_lengths', (31,)),
                                     ('h3_vae_forward', 'succeeded', False)]:
        value = state()
        for event, facts in events(object()):
            if event == event_name:
                facts[key] = changed
            observe_h3_vae_metrics(value, event, facts)
        assert not h3_video_vae_complete(value) and value['reasons']


def test_reference_correlation_requires_one_matching_operation_and_output_per_selected_visual():
    value = state()
    for event, facts in [('h3_reference_begin', {}), ('h3_reference_source', source()),
                         ('h3_reference_processed', processed()), ('h3_reference_output',
                          metadata({'visual': (1, 24, 1, 2, 2)}, index=0, role='image', slot='ref_images.ref_image_0')),
                         ('h3_reference_prepared', dict(references=1, tokenizer_items=1, blocks=1))]:
        observe_h3_reference_metrics(value, event, facts)
    for event, facts in events(object()):
        observe_h3_vae_metrics(value, event, facts)
    assert h3_reference_video_vae_complete(value)
    value['h3_reference_entries'][0]['audio_encode'] = True
    assert h3_reference_video_vae_complete(value) and not h3_reference_audio_complete(value)
    value['h3_reference_entries'][0]['audio_encode'] = False
    for mutation in ('index', 'shape', 'duplicate'):
        other = copy.deepcopy(value)
        if mutation == 'index':
            next(iter(other['h3_vae_operations'].values()))['reference_index'] = None
        elif mutation == 'shape':
            other['h3_reference_entries'][0]['visual_shape'] = (1, 64, 32, 3)
        else:
            other['h3_vae_operations'][object()] = copy.deepcopy(next(iter(other['h3_vae_operations'].values())))
        assert not h3_reference_video_vae_complete(other)


def test_coalesced_store_snapshots_leave_loading_and_text_gaps(tmp_path):
    store = start(tmp_path)
    sequence = events(object())
    store.safe_metric('prompt', 'refs', *sequence[0])
    changes = store.db.total_changes
    for event, facts in sequence[1:-1]:
        store.safe_metric('prompt', 'refs', event, facts)
    assert store.db.total_changes == changes
    store.safe_metric('prompt', 'refs', *sequence[-1])
    work = latest(store)['stages'][1]['workload']
    assert 'h3_video_vae_work_unmeasured' not in work['reasons']
    assert {'managed_load_plan_unresolved', 'h3_conditioning_text_work_unmeasured',
            'h3_reference_preparation_unmeasured'} <= set(work['reasons'])
    assert not work['complete']
    assert 'binding' not in repr(work)


def test_duplicates_limits_and_probe_errors_are_bounded():
    value = state()
    for _ in range(MAX_OPERATIONS + 2):
        observe_h3_vae_metrics(value, *events(object())[0])
    assert len(value['h3_vae_operations']) == MAX_OPERATIONS and value['reasons']
    other = state()
    sequence = events(object())
    for event, facts in sequence + [sequence[-1]]:
        observe_h3_vae_metrics(other, event, facts)
    assert not h3_video_vae_complete(other)
    observe_h3_vae_metrics(other, 'tensor_shape_unavailable', {'operation': 'h3_vae_output'})
    assert other['reasons'] == ['h3_video_vae_metadata_unresolved']


def test_clip_and_planned_forward_limits_apply_across_the_stage():
    value = state()
    for event, facts in events(object(), frames=17 * (MAX_CLIPS + 1)):
        observe_h3_vae_metrics(value, event, facts)
    assert sum(len(op['clips']) for op in value['h3_vae_operations'].values()) == MAX_CLIPS
    assert not h3_video_vae_complete(value)
    other = state()
    for _ in range(3):
        sequence = events(object())
        for event, facts in sequence[:3]:
            if event in ('h3_vae_begin', 'h3_vae_clip'):
                facts['shapes']['pixels'] = (1, 3, 1, 272, 272)
            else:
                for axis in ('x', 'y'):
                    facts[axis + '_starts'] = tuple(range(0, 256, 16))
                    facts[axis + '_lengths'] = (32,) * 16
                    facts[axis + '_overlaps'] = (16,) * 15
            observe_h3_vae_metrics(other, event, facts)
    planned = sum(len(clip.get('tiles', [])) for op in other['h3_vae_operations'].values() for clip in op['clips'])
    assert planned == MAX_FORWARDS and other['reasons']
