"""Local native Omni acceptance. Acquire an idle GPU window and the existing GPU hold first."""
import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from unittest import mock

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen3_omni_checkpoint import SOURCE_REVISION, sha256_file


def capture(model):
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
        if len(logits) < 8:
            logits.append(args[0][0].detach().float().cpu().clone())
        if not policies:
            quantized = [module for module in model.modules() if getattr(module, 'quant_format', None) == 'nvfp4']
            policies.append({'quantized_modules': len(quantized),
                             'fast_modules': sum(not module._full_precision_mm for module in quantized)})
        return original_sample(*args, **kwargs)

    stack.enter_context(mock.patch.object(model, 'preprocess_embed', preprocess))
    stack.enter_context(mock.patch.object(model, 'image_inputs', image_inputs))
    stack.enter_context(mock.patch.object(model.model, 'forward', forward))
    stack.enter_context(mock.patch.object(model, 'sample_token', sample))
    handles.append(model.visual.register_forward_pre_hook(vision_input))
    for index, layer in enumerate(model.model.layers):
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
    parser.add_argument('--precision', choices=('nvfp4', 'decoded-reference'), default='nvfp4')
    parser.add_argument('--case', action='append', help='Select named frozen cases; may repeat')
    parser.add_argument('--tokens', type=int, help='Explicit smoke/performance budget override, recorded in results')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Record a CUDA/CPU trace for the first case')
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    manifest_path = options.model.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest['source_revision'] != SOURCE_REVISION or sha256_file(options.model) != manifest['output_sha256']:
        raise ValueError('Candidate artifact hash/revision does not match its provenance manifest')
    cases = json.loads(options.cases.read_text())
    if options.case:
        if set(options.case) - {case['id'] for case in cases}:
            raise ValueError('Unknown selected fixture case')
        cases = [case for case in cases if case['id'] in options.case]
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    from comfy.cli_args import args
    args.use_ck_attention = True
    import comfy_aimdo.control
    comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True)
    import comfy.sd
    import comfy.memory_management
    import comfy.model_patcher
    if not comfy_aimdo.control.init_devices([(0, int(args.vram_headroom * 1024**3))]):
        raise RuntimeError('Native acceptance requires the serving dynamic-VRAM allocator')
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    device = torch.device('cuda:0')
    model_options = {'initial_device': torch.device('cpu'), 'load_device': device, 'offload_device': torch.device('cpu')}
    if options.precision == 'decoded-reference':
        # Keep the packed banks; the existing operation decodes one indexed matrix.
        model_options['custom_operations'] = comfy.ops.mixed_precision_ops({}, torch.bfloat16, full_precision_mm=True, disabled=['nvfp4'])
    elif 'nvfp4' in comfy.ops.get_disabled_quant_formats(device):
        raise RuntimeError('Accelerated NVFP4 is unavailable; decoded execution is not a production substitute')
    report = {'source_revision': SOURCE_REVISION, 'artifact_sha256': manifest['output_sha256'],
              'manifest_sha256': sha256_file(manifest_path), 'cases_sha256': sha256_file(options.cases),
              'harness_sha256': sha256_file(__file__), 'precision': options.precision,
              'gpu': torch.cuda.get_device_name(), 'tokens_override': options.tokens,
              'capture_enabled': options.capture, 'profile_enabled': options.profile,
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'comfy-kitchen', 'comfy-aimdo')},
              'cache_condition': 'Fresh native process; OS file cache uncontrolled and warmed by integrity verification', 'cases': []}
    start = time.perf_counter()
    clip = comfy.sd.load_clip([str(options.model)], clip_type=comfy.sd.CLIPType.QWEN3_OMNI, model_options=model_options)
    report['checkpoint_load_seconds'] = time.perf_counter() - start
    model = clip.cond_stage_model.qwen3_omni.transformer
    stop_monitor = threading.Event()
    driver_peak = [0]

    def monitor():
        while not stop_monitor.is_set():
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
            image = torch.from_numpy(np.load(case['image_tensor'], allow_pickle=False)) if case.get('image_tensor') else None
            if case.get('images') and image is None:
                raise ValueError('Use frozen fixture version 2 IMAGE tensors for native acceptance')
            audio = None
            if case.get('audio'):
                waveform, rate = sf.read(case['audio'], dtype='float32', always_2d=True)
                audio = {'waveform': torch.from_numpy(waveform.T.copy()).unsqueeze(0), 'sample_rate': rate}
            tokens = clip.tokenize(case['prompt'], image=image, audio=audio, thinking=False, system_prompt=case.get('system_prompt', ''))
            finish_capture = capture(model) if options.capture else None
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) if options.profile and index == 0 else None
            try:
                with profile if profile is not None else ExitStack():
                    budget = case['max_new_tokens'] if options.tokens is None else options.tokens
                    ids = clip.generate(tokens, do_sample=False, max_length=budget, seed=17, mtp=False)
                torch.cuda.synchronize()
            finally:
                if finish_capture is not None:
                    captured, times, policies = finish_capture()
            elapsed = time.perf_counter() - start
            raw = clip.decode(ids)
            before, separator, final = raw.partition('</think>')
            if not separator:
                final = '' if before.lstrip().startswith('<think>') else raw
            result = {'id': case['id'], 'text': final.strip(), 'raw_text': raw, 'generated_ids': ids,
                      'eos_reached': bool(ids and ids[-1] == model.model.config.stop_tokens[0]),
                      'elapsed_seconds': elapsed, 'peak_torch_gpu_bytes': torch.cuda.max_memory_allocated(),
                      'peak_host_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
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
            if profile is not None:
                profile.export_chrome_trace(str(options.output / 'native-trace.json'))
                result['operator_counts'] = {entry.key: entry.count for entry in profile.key_averages() if 'nvfp4' in entry.key}
            report['cases'].append(result)
            report['peak_driver_gpu_bytes'] = driver_peak[0]
            (options.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            print(case['id'], result['text'], flush=True)
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=3)


if __name__ == '__main__':
    main()
