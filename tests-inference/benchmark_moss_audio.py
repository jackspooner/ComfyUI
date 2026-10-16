"""Explicit local-only MOSS benchmark; acquire exclusive GPU ownership before running.

Run each backend in a fresh process three times, then use --iterations 3 for warm
executions. Native uses serving ComfyUI Python; reference uses Transformers 4.57.1
and --reference-source pointing at the pinned OpenMOSS tree. No files are fetched.
"""
import argparse
import functools
import json
import os
import subprocess
import threading
from pathlib import Path
import resource
import sys
import time
from unittest import mock

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['native','reference'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--audio', required=True, nargs='+')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reference-source')
    parser.add_argument('--iterations',type=int,default=1)
    parser.add_argument('--tokens',type=int,default=128)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--cache-condition',required=True)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--lifecycle',action='store_true')
    parser.add_argument('--transition-model')
    parser.add_argument('--prompt',default='Describe the foreground and background sounds and the order of events. Distinguish speech, music and environmental sounds. State uncertainty where needed.')
    options=parser.parse_args()
    device=torch.device(options.device)
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    report=dict(backend=options.backend,model=options.model,audio=options.audio,
                cache_condition=options.cache_condition,capture_enabled=bool(options.capture),torch=torch.__version__,runs=[])
    captured={}
    driver_peak = [0]
    stop_monitor = threading.Event()
    def monitor_vram():
        while not stop_monitor.is_set():
            output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
            for line in output.splitlines():
                pid, memory = line.split(',')
                if int(pid) == os.getpid():
                    driver_peak[0] = max(driver_peak[0], int(memory) * 1024**2)
            stop_monitor.wait(0.5)
    if device.type == 'cuda':
        threading.Thread(target=monitor_vram, daemon=True).start()
    # This validation process is inference-only, matching ComfyUI execution.
    with torch.set_grad_enabled(False):
        if options.backend=='native':
            from comfy.cli_args import args
            args.cpu=device.type=='cpu'
            args.use_ck_attention=True
            if device.type=='cuda':
                import comfy_aimdo.control
                comfy_aimdo.control.init(simple_vram_headroom=None,nvml_pressure=True)
            import comfy.sd
            if device.type=='cuda':
                import comfy.memory_management
                import comfy.model_patcher
                if not comfy_aimdo.control.init_devices([(0,int(args.vram_headroom*1024**3))]):
                    raise RuntimeError('Native benchmark requires the serving dynamic-VRAM allocator')
                comfy.model_patcher.CoreModelPatcher=comfy.model_patcher.ModelPatcherDynamic
                comfy.memory_management.aimdo_enabled=True
            import comfy.text_encoders.moss_audio as moss
            start=time.perf_counter()
            clip=comfy.sd.load_clip([options.model],clip_type=comfy.sd.CLIPType.MOSS_AUDIO,
                                   model_options={'initial_device':torch.device('cpu'),'load_device':device,'offload_device':torch.device('cpu')})
            report['checkpoint_load_seconds']=time.perf_counter()-start
            start=time.perf_counter()
            clip.load_model()
            sync()
            report['gpu_placement_seconds']=time.perf_counter()-start
            model=clip.cond_stage_model.moss_audio.transformer
            encoder=model.audio_encoder
            decode=lambda ids:clip.decode(ids)
            def prepare():
                return clip.tokenize(options.prompt,audio=audio,thinking=True)
            def generate(inputs):
                return clip.generate(inputs,do_sample=False,max_length=options.tokens,seed=42)
            feature_owner=moss
            feature_name='audio_features'
            logits_owner=model
            logits_name='sample_token'
            def save_logits(call_args,output):
                return call_args[0]
            def encoder_tensors(output):
                return output
        else:
            if options.reference_source is None:
                parser.error('--reference-source is required for the reference backend')
            sys.path.insert(0,options.reference_source)
            from src.modeling_moss_audio import MossAudioModel
            from src.processing_moss_audio import MossAudioProcessor
            import comfy.audio
            start=time.perf_counter()
            model=MossAudioModel.from_pretrained(options.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
            processor=MossAudioProcessor.from_pretrained(options.model,local_files_only=True,enable_time_marker=True)
            report['checkpoint_load_seconds']=time.perf_counter()-start
            start=time.perf_counter()
            model.to(device)
            sync()
            report['gpu_placement_seconds']=time.perf_counter()-start
            encoder=model.audio_encoder
            decode=lambda ids:processor.decode(ids,skip_special_tokens=True)
            def prepare():
                mono=comfy.audio.resample(audio['waveform'].mean(dim=1),rate,16000)[0]
                inputs=processor(text=options.prompt,audios=[mono],return_tensors='pt').to(device)
                inputs['audio_data']=inputs['audio_data'].to(torch.bfloat16)
                inputs['audio_input_mask']=inputs['input_ids']==processor.audio_token_id
                return inputs
            def generate(inputs):
                return model.generate(**inputs,do_sample=False,max_new_tokens=options.tokens,use_cache=True)[0,inputs['input_ids'].shape[1]:].tolist()
            feature_owner=processor
            feature_name='_extract_mel'
            logits_owner=model
            logits_name='forward'
            def save_logits(call_args,output):
                return output.logits[:,-1]
            def encoder_tensors(output):
                return [output.last_hidden_state[0],*[x[0] for x in output.hidden_states]]
        for iteration,audio_path in enumerate(options.audio * options.iterations):
            waveform,rate=sf.read(audio_path,dtype='float32',always_2d=True)
            audio={'waveform':torch.from_numpy(waveform.T.copy())[None], 'sample_rate':rate}
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            metrics={'audio':audio_path,'audio_encoder_calls':0,'preprocessing_seconds':0.0,'encoder_seconds':0.0}
            token_times=[]
            original_features=getattr(feature_owner,feature_name)
            original_encoder=encoder.forward
            original_logits=getattr(logits_owner,logits_name)
            def features(*a,**kw):
                sync();start=time.perf_counter();result=original_features(*a,**kw);sync()
                metrics['preprocessing_seconds']+=time.perf_counter()-start
                if options.capture and iteration==0:captured['mel']=result.detach().float().cpu()
                return result
            def encode(*a,**kw):
                sync();start=time.perf_counter();result=original_encoder(*a,**kw);sync()
                metrics['encoder_seconds']+=time.perf_counter()-start
                metrics['audio_encoder_calls']+=1
                if options.capture and iteration==0:
                    for i,value in enumerate(encoder_tensors(result)):captured[f'encoder_{i}']=value.detach().cpu()
                return result
            @functools.wraps(original_logits)
            def logits(*a,**kw):
                result=original_logits(*a,**kw);sync();token_times.append(time.perf_counter())
                if options.capture and iteration==0 and len(token_times)<=4:
                    captured[f'logits_{len(token_times)-1}']=save_logits(a,result).detach().float().cpu()
                return result
            sync();start=time.perf_counter()
            with mock.patch.object(feature_owner,feature_name,features),mock.patch.object(encoder,'forward',encode),mock.patch.object(logits_owner,logits_name,logits):
                inputs=prepare()
                ids=generate(inputs)
            sync();end=time.perf_counter()
            metrics.update(total_seconds=end-start,time_to_first_token_seconds=token_times[0]-start,
                           decode_tokens_per_second=(len(ids)-1)/(token_times[-1]-token_times[0]) if len(ids)>1 else None,
                           output_token_count=len(ids),token_ids=ids,text=decode(ids),
                           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                           driver_peak_process_vram_bytes=driver_peak[0],
                           torch_peak_vram_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                           torch_peak_vram_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=='cuda' else 0)
            if metrics['audio_encoder_calls']!=1:raise RuntimeError(f"Audio encoder ran {metrics['audio_encoder_calls']} times")
            report['runs'].append(metrics)
            options.report.parent.mkdir(parents=True,exist_ok=True)
            options.report.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in metrics.items() if k not in ('token_ids','text')}),flush=True)
            print(metrics['text'],flush=True)
        if options.lifecycle:
            if options.backend != 'native':
                parser.error('--lifecycle requires native')
            from comfy import model_management
            model_management.interrupt_current_processing(True)
            try:
                generate(prepare())
            except model_management.InterruptProcessingException:
                report['cancellation_observed'] = True
            else:
                raise AssertionError('Cancellation was not observed')
            finally:
                model_management.interrupt_current_processing(False)
            try:
                clip.generate(prepare(), do_sample=False, max_length=40960, seed=42)
            except ValueError as error:
                if 'context tokens' not in str(error):
                    raise
                report['context_capacity_error'] = str(error)
            else:
                raise AssertionError('Context overflow was not rejected')
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            recovered = generate(prepare())
            if recovered != ids:
                raise AssertionError('Offload/reload changed greedy output')
            report['offload_reload_output_equal'] = True
            text_ids = clip.generate(clip.tokenize('Reply with the word ready.', thinking=False), do_sample=False, max_length=32, seed=42)
            report['text_only_output'] = decode(text_ids)
            if options.transition_model:
                other = comfy.sd.load_clip([options.transition_model], model_options={'load_device': device, 'offload_device': torch.device('cpu')})
                other_ids = other.generate(other.tokenize('Reply with the word ready.', thinking=False), do_sample=False, max_length=32, seed=42)
                report['transition_model'] = options.transition_model
                report['transition_output'] = other.decode(other_ids)
                returned = generate(prepare())
                if returned != ids:
                    raise AssertionError('Model transition changed MOSS greedy output')
                report['transition_return_output_equal'] = True
            options.report.write_text(json.dumps(report, indent=2) + '\n')
        stop_monitor.set()
        if options.capture:
            options.capture.parent.mkdir(parents=True,exist_ok=True)
            torch.save(captured,options.capture)


if __name__=='__main__':main()
