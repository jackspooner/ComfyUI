"""Execution-local lifecycle checks used by the native Omni acceptance command."""
import json
from contextlib import nullcontext
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from unittest import mock

import torch


def memory_snapshot():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    driver = None
    for row in rows.splitlines():
        pid, memory = row.split(',')
        if int(pid) == os.getpid() and memory.strip().isdigit():
            driver = int(memory) * 1024**2
    rss = next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
    return {'gpu_driver_bytes': driver, 'torch_allocated_bytes': torch.cuda.memory_allocated(),
            'torch_reserved_bytes': torch.cuda.memory_reserved(), 'host_rss_bytes': int(rss.split()[1]) * 1024,
            'peak_host_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def check_lifecycle(clip, tokens, expected_ids, transition_model, budget, contention, output):
    # Core imports must follow the caller's native allocator initialization.
    import comfy.model_management as management
    import comfy.sd
    import comfy.text_encoders.qwen3_omni as omni

    if output.exists():
        raise FileExistsError(output)
    types = [token['type'] for token, *_ in tokens['qwen3_omni'][0] if isinstance(token, dict)]
    if 'audio' not in types or 'image' not in types:
        raise ValueError('Lifecycle validation requires a combined image/audio fixture')
    model = clip.cond_stage_model.qwen3_omni.transformer
    report = {'status': 'running', 'transition_model': str(transition_model), 'steps': []}

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')

    def ensure_window():
        if contention.is_set():
            raise RuntimeError('Normal queue work interrupted lifecycle validation')

    def generate():
        ensure_window()
        return clip.generate(tokens, do_sample=False, max_length=budget, seed=17, mtp=False)

    def success(name, **values):
        torch.cuda.synchronize()
        ensure_window()
        report['steps'].append({'name': name, **values, 'memory': memory_snapshot()})
        save()
        print('lifecycle', name, flush=True)

    def cancel(stage):
        stamp = {}
        def request():
            stamp['requested_at'] = time.perf_counter()
            management.interrupt_current_processing(True)
        original_features = omni.audio_features
        def features(*args, **kwargs):
            timer = threading.Timer(.001, request)
            timer.start()
            try:
                return original_features(*args, **kwargs)
            finally:
                timer.join()
        original_sample, count = model.sample_token, [0]
        def sample(*args, **kwargs):
            count[0] += 1
            if count[0] == 2:
                request()
            return original_sample(*args, **kwargs)
        hook = None
        if stage == 'prefill':
            hook = model.model.layers[2].register_forward_pre_hook(lambda *args: request())
        context = (mock.patch.object(omni, 'audio_features', features) if stage == 'preprocessing' else
                   mock.patch.object(model, 'sample_token', sample) if stage == 'decode' else nullcontext())
        try:
            with context:
                try:
                    generate()
                except management.InterruptProcessingException:
                    ensure_window()
                    torch.cuda.synchronize()
                    if 'requested_at' not in stamp:
                        raise AssertionError('Cancellation came from outside the selected phase')
                    success('cancel_' + stage, propagation_seconds=time.perf_counter() - stamp['requested_at'])
                else:
                    raise AssertionError('Generation did not stop during ' + stage)
        finally:
            if hook is not None:
                hook.remove()
            management.interrupt_current_processing(False)

    save()
    try:
        for index in range(3):
            assert generate() == expected_ids, 'Repeated deterministic output differs'
            success('repeat_' + str(index + 1), ids_equal=True)
        repeat_memory = report['steps'][-1]['memory']
        for stage in ('preprocessing', 'prefill', 'decode'):
            cancel(stage)
        assert generate() == expected_ids, 'Output changed after cancellation'
        success('cancellation_recovery', ids_equal=True)
        delta = report['steps'][-1]['memory']['torch_allocated_bytes'] - repeat_memory['torch_allocated_bytes']
        assert delta <= 64 * 1024**2, 'Retained Torch allocations grew after cancellation'
        report['retained_torch_delta_bytes'] = delta
        management.unload_all_models()
        management.soft_empty_cache()
        success('offloaded')
        assert generate() == expected_ids, 'Output changed after offload/reload'
        success('reloaded', ids_equal=True)
        other = comfy.sd.load_clip([str(transition_model)], model_options={'initial_device': torch.device('cpu'),
            'load_device': clip.patcher.load_device, 'offload_device': clip.patcher.offload_device})
        other_ids = other.generate(other.tokenize('Reply exactly with READY.', thinking=False), do_sample=False,
                                   max_length=64, seed=17, mtp=False)
        assert other_ids, 'Transition model returned no tokens'
        success('other_model', generated_ids=other_ids, raw_text=other.decode(other_ids))
        assert generate() == expected_ids, 'Output changed after another model ran'
        success('model_transition_recovery', ids_equal=True)
        management.unload_all_models()
        management.soft_empty_cache()
        success('final_offload')
    finally:
        error = sys.exc_info()[1]
        report['status'] = 'complete' if error is None else 'failed_or_interrupted'
        if error is not None:
            report['error'] = {'type': type(error).__name__, 'message': str(error)}
        save()
    return report
