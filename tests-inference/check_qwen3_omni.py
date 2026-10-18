"""Local native Omni acceptance. Acquire an idle GPU window and the existing GPU hold first."""
import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import sys
import threading
import time
import urllib.request
from unittest import mock

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen3_omni_checkpoint import FP8_REVISION, SOURCE_REVISION, sha256_file
from qwen3_omni_lifecycle import check_lifecycle, memory_snapshot


def capture(model, tensors=True):
    values, handles = {}, []
    image_features, pixels, deepstack = [], [], []
    stack = ExitStack()
    original_preprocess, original_forward = model.preprocess_embed, model.model.forward
    original_sample, original_inputs = model.sample_token, model.image_inputs
    times, logits, policies = [], [], []

    def preprocess(embed, device):
        result, extra = original_preprocess(embed, device)
        if embed['type'] == 'audio':
            values['audio_features'] = result.detach().cpu().clone()
        elif embed['type'] == 'image':
            image_features.append(result.detach().cpu().clone())
            deepstack.append([part.detach().cpu().clone() for part in extra['deepstack']])
        return result, extra

    def vision_input(module, arguments):
        pixels.append(arguments[0].detach().cpu().clone())

    def image_inputs(embeds, info):
        result = original_inputs(embeds, info)
        values['position_ids'] = result[0].detach().cpu().unsqueeze(1).clone()
        return result

    def forward(*args, **kwargs):
        if 'prefill_embeds' not in values:
            values['prefill_embeds'] = kwargs['embeds'].detach().cpu().clone()
        return original_forward(*args, **kwargs)

    def sample(*args, **kwargs):
        torch.cuda.synchronize()
        times.append(time.perf_counter())
        if tensors and len(logits) < 8:
            logits.append(args[0][0].detach().float().cpu().clone())
        if not policies:
            quantized = [module for module in model.modules() if getattr(module, 'quant_format', None) == 'nvfp4']
            policies.append({'quantized_modules': len(quantized),
                             'fp8_block_modules': sum(getattr(module, 'quant_format', None) == 'float8_e4m3fn_block128' for module in model.modules()),
                             'fast_modules': sum(not module._full_precision_mm for module in quantized),
                             'nvfp4_input_passes': sorted({module._nvfp4_input_passes for module in quantized})})
        return original_sample(*args, **kwargs)

    stack.enter_context(mock.patch.object(model, 'sample_token', sample))
    if tensors:
        stack.enter_context(mock.patch.object(model, 'preprocess_embed', preprocess))
        stack.enter_context(mock.patch.object(model, 'image_inputs', image_inputs))
        stack.enter_context(mock.patch.object(model.model, 'forward', forward))
        handles.append(model.visual.register_forward_pre_hook(vision_input))
    for index, layer in enumerate(model.model.layers if tensors else []):
        def router_hook(module, args, output, index=index):
            if f'router_{index}_indices' not in values:
                probabilities = output.float().softmax(-1)
                weights, indices = probabilities.topk(model.model.config.num_experts_per_tok, dim=-1)
                weights = (weights / weights.sum(-1, keepdim=True)).to(output.dtype)
                for name, value in (('logits', output), ('weights', weights), ('indices', indices)):
                    values[f'router_{index}_{name}'] = value.detach().cpu().clone()
        handles.append(layer.mlp.gate.register_forward_hook(router_hook))

    def finish():
        for handle in handles:
            handle.remove()
        stack.close()
        if image_features:
            values['vision_features'] = torch.cat(image_features)
            values['pixel_values_compute'] = torch.cat(pixels)
            for index in range(len(deepstack[0])):
                values[f'vision_deepstack_{index}'] = torch.cat([parts[index] for parts in deepstack])
        if logits:
            values['decode_logits'] = torch.stack(logits)
        return values, times, policies
    return finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=('nvfp4', 'decoded-reference', 'fp8-block128'), default='nvfp4')
    parser.add_argument('--nvfp4-input-passes', type=int, help='Research override for the shared residual activation operation')
    parser.add_argument('--case', action='append', help='Select named frozen cases; may repeat')
    parser.add_argument('--tokens', type=int, help='Explicit smoke/performance budget override, recorded in results')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--vram-headroom', type=float, help='Explicit extra DynamicVRAM headroom in GiB, matching the serving CLI policy; recorded in results')
    parser.add_argument('--profile', action='store_true', help='Record a CUDA/CPU trace for the first case')
    parser.add_argument('--comfy-url', default='http://127.0.0.1:8188', help='Serving queue to check before and during isolated GPU validation')
    parser.add_argument('--lifecycle-transition-model', type=Path, help='After the final combined fixture, verify repeat/cancel/offload and transition to this existing text-generation CLIP')
    options = parser.parse_args()
    if options.precision == 'decoded-reference' and options.nvfp4_input_passes is not None:
        raise ValueError('Activation passes apply to NVFP4 execution, not the decoded reference')
    options.output.mkdir(parents=True, exist_ok=False)
    manifest_path = options.model.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    source_revision = FP8_REVISION if options.precision == 'fp8-block128' else SOURCE_REVISION
    if manifest['source_revision'] != source_revision or sha256_file(options.model) != manifest['output_sha256']:
        raise ValueError('Candidate artifact hash/revision does not match its provenance manifest')
    cases = json.loads(options.cases.read_text())
    if not cases or len({case['id'] for case in cases}) != len(cases) or any(re.fullmatch(r'[A-Za-z0-9_-]+', case['id']) is None for case in cases):
        raise ValueError('Provide nonempty fixtures with unique safe capture IDs')
    if options.case:
        if set(options.case) - {case['id'] for case in cases}:
            raise ValueError('Unknown selected fixture case')
        cases = [case for case in cases if case['id'] in options.case]
    queue_url = options.comfy_url.rstrip('/') + '/queue'
    def active_queue():
        with urllib.request.urlopen(queue_url, timeout=10) as response:
            queue = json.load(response)
        return [entry[1] for name in ('queue_running', 'queue_pending') for entry in queue[name]]
    if active_queue():
        raise RuntimeError('Serving ComfyUI queue became active during integrity verification; no GPU validation started')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    from comfy.cli_args import args
    args.use_ck_attention = True
    import comfy_aimdo.control
    comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True)
    import comfy.sd
    import comfy.memory_management
    import comfy.model_patcher
    vram_headroom = args.vram_headroom if options.vram_headroom is None else options.vram_headroom
    if not comfy_aimdo.control.init_devices([(0, int(vram_headroom * 1024**3))]):
        raise RuntimeError('Native acceptance requires the serving dynamic-VRAM allocator')
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    device = torch.device('cuda:0')
    model_options = {'initial_device': torch.device('cpu'), 'load_device': device, 'offload_device': torch.device('cpu')}
    if options.precision == 'decoded-reference':
        # Keep the packed banks; the existing operation decodes one indexed matrix.
        model_options['custom_operations'] = comfy.ops.mixed_precision_ops({}, torch.bfloat16, full_precision_mm=True, disabled=['nvfp4'])
    elif options.precision == 'nvfp4' and 'nvfp4' in comfy.ops.get_disabled_quant_formats(device):
        raise RuntimeError('Accelerated NVFP4 is unavailable; decoded execution is not a production substitute')
    elif options.nvfp4_input_passes is not None:
        model_options['custom_operations'] = comfy.ops.mixed_precision_ops({}, torch.bfloat16, full_precision_mm=True,
                                                                         nvfp4_input_passes=options.nvfp4_input_passes)
    repository = Path(__file__).resolve().parents[1]
    source_files = ['comfy/text_encoders/qwen3_omni.py', 'comfy/text_encoders/llama.py', 'comfy/ops.py',
                    'comfy/quant_ops.py', 'comfy/sd.py', 'comfy/sd1_clip.py', 'comfy/model_management.py',
                    'comfy/model_prefetch.py', 'comfy/model_patcher.py']
    source_hashes = {path: sha256_file(repository / path) for path in source_files}
    report = {'source_revision': source_revision, 'artifact_sha256': manifest['output_sha256'],
              'manifest_sha256': sha256_file(manifest_path), 'cases_sha256': sha256_file(options.cases),
              'harness_sha256': sha256_file(__file__), 'precision': options.precision,
              'gpu': torch.cuda.get_device_name(), 'tokens_override': options.tokens,
              'capture_enabled': options.capture, 'profile_enabled': options.profile,
              'source_code_sha256': source_hashes,
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
              'source_changes': subprocess.check_output(['git', 'status', '--porcelain', '--', *source_files], cwd=repository, text=True),
              'nvfp4_input_passes_override': options.nvfp4_input_passes,
              'lifecycle_harness_sha256': sha256_file(Path(__file__).with_name('qwen3_omni_lifecycle.py')),
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'comfy-kitchen', 'comfy-aimdo')},
              'cache_condition': 'Fresh native process; OS file cache uncontrolled and warmed by integrity verification', 'cases': []}
    report['vram_headroom_gib'] = vram_headroom
    start = time.perf_counter()
    clip = comfy.sd.load_clip([str(options.model)], clip_type=comfy.sd.CLIPType.QWEN3_OMNI, model_options=model_options)
    report['checkpoint_load_seconds'] = time.perf_counter() - start
    report['status'] = 'running'
    (options.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    model = clip.cond_stage_model.qwen3_omni.transformer
    stop_monitor = threading.Event()
    contention = threading.Event()
    driver_peak = [0]

    def monitor():
        while not stop_monitor.is_set():
            try:
                prompts = active_queue()
            except (OSError, ValueError) as error:
                prompts = ['queue state unavailable: ' + str(error)]
            if prompts:
                (options.output / 'contention.json').write_text(json.dumps({'prompt_ids_or_error': prompts}, indent=2) + '\n')
                contention.set()
                comfy.model_management.interrupt_current_processing(True)
                return
            output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
            for line in output.splitlines():
                pid, memory = line.split(',')
                if int(pid) == os.getpid():
                    driver_peak[0] = max(driver_peak[0], int(memory) * 1024**2)
            stop_monitor.wait(.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        for index, case in enumerate(cases):
            if contention.is_set():
                raise RuntimeError('Serving queue became active; stop isolated validation and release only its own hold after exit')
            if source_hashes != {path: sha256_file(repository / path) for path in source_files}:
                raise RuntimeError('Inference source changed during acceptance; preserve this run and retry a stable revision')
            image = torch.from_numpy(np.load(case['image_tensor'], allow_pickle=False)) if case.get('image_tensor') else None
            if case.get('images') and image is None:
                raise ValueError('Use frozen fixture version 2 IMAGE tensors for native acceptance')
            audio = None
            if case.get('audio'):
                waveform, rate = sf.read(case['audio'], dtype='float32', always_2d=True)
                audio = {'waveform': torch.from_numpy(waveform.T.copy()).unsqueeze(0), 'sample_rate': rate}
            tokens = clip.tokenize(case['prompt'], image=image, audio=audio, thinking=False,
                                   skip_template=case.get('skip_template', False), system_prompt=case.get('system_prompt', ''))
            finish_capture = capture(model, tensors=options.capture)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) if options.profile and index == 0 else None
            try:
                with profile if profile is not None else ExitStack():
                    budget = case['max_new_tokens'] if options.tokens is None else options.tokens
                    ids = clip.generate(tokens, do_sample=False, max_length=budget, seed=17, mtp=False)
                torch.cuda.synchronize()
                if contention.is_set():
                    raise RuntimeError('Serving queue became active during this case; it is not an accepted measurement')
            finally:
                captured, times, policies = finish_capture()
            elapsed = time.perf_counter() - start
            raw = clip.decode(ids)
            before, separator, final = raw.partition('</think>')
            if not separator:
                final = '' if before.lstrip().startswith('<think>') else raw
            result = {'id': case['id'], 'text': final.strip(), 'raw_text': raw, 'generated_ids': ids,
                      'eos_reached': bool(ids and ids[-1] == model.model.config.stop_tokens[0]),
                      'elapsed_seconds': elapsed, 'peak_torch_gpu_bytes': torch.cuda.max_memory_allocated(),
                      'peak_host_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      'memory_after_generation': memory_snapshot()}
            result['assertions'] = [{'alternatives': group, 'passed': any(value.casefold() in final.casefold() for value in group)}
                                    for group in case.get('assert_all', [])]
            result['fixture_passed'] = all(assertion['passed'] for assertion in result['assertions'])
            if options.capture:
                torch.save(captured, options.output / (case['id'] + '.pt'))
            result['execution_policy'] = policies
            result['first_token_seconds'] = times[0] - start
            if len(times) > 1:
                result['decode_tokens_per_second'] = (len(times) - 1) / (times[-1] - times[0])
            if options.precision == 'nvfp4' and (not policies or not policies[0]['fast_modules']):
                raise RuntimeError('Native run selected decoded execution; accelerated acceptance failed')
            if options.precision == 'fp8-block128' and (not policies or not policies[0]['fp8_block_modules'] or policies[0]['quantized_modules']):
                raise RuntimeError('Native run did not use the published FP8 block layout')
            if profile is not None:
                profile.export_chrome_trace(str(options.output / 'native-trace.json'))
                result['operator_counts'] = {entry.key: entry.count for entry in profile.key_averages() if 'nvfp4' in entry.key}
            report['cases'].append(result)
            report['peak_driver_gpu_bytes'] = driver_peak[0]
            (options.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            print(case['id'], result['text'], flush=True)
        if options.lifecycle_transition_model is not None:
            report['lifecycle'] = check_lifecycle(clip, tokens, ids, options.lifecycle_transition_model, budget,
                                                   contention, options.output / 'lifecycle.json')
        if source_hashes != {path: sha256_file(repository / path) for path in source_files}:
            raise RuntimeError('Inference source changed during validation')
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=3)
        error = sys.exc_info()[1]
        report['status'] = 'complete' if error is None else 'failed_or_interrupted'
        if error is not None:
            report['error'] = {'type': type(error).__name__, 'message': str(error)}
        report['peak_driver_gpu_bytes'] = driver_peak[0]
        (options.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
