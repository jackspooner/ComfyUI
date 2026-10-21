"""Pinned offline Fish reference with layer-wise weight residency for 32 GB GPUs.

Only checkpoint loading/residency is adapted. Upstream layers, prompt builder,
codebook decoder, sampling and codec perform the actual reference computation.
Run under the installation's coordinated exclusive GPU admission.
"""
import argparse
import json
import os
import resource
from pathlib import Path
import sys
import time
from types import MethodType
from contextlib import ExitStack
import torch
import psutil
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_drama3 import Source, UPSTREAM_REVISION, SOURCE_REVISION


def install_streaming_backbone(backbone, source, config, device, dtype, resident_layers, LayerContext, cached_layers=0):
    """Retain a bounded prefix on CUDA; stream remaining original FP8 layers."""
    keys = [[] for _ in backbone.model.layers]
    for key in source.mapping:
        if key.startswith('model.language_model.layers.') and '.mlp.experts.' not in key:
            keys[int(key.split('.')[3])].append(key)

    def load_layer(index, target=device):
        prefix = f'model.language_model.layers.{index}.'
        layer_state = {k[len(prefix):]: source.get(k).to(target) for k in keys[index]}
        c = config.text
        for bank, projections, rows, cols in [('w13', ('gate_proj', 'up_proj'), 2 * c.moe_intermediate_size, c.hidden_size),
                                               ('w2', ('down_proj',), c.hidden_size, c.moe_intermediate_size)]:
            weights = torch.empty(c.num_experts, rows, cols, dtype=torch.float8_e4m3fn, device=target)
            scales = torch.empty(c.num_experts, rows // 128, cols // 128, dtype=torch.float32, device=target)
            for expert in range(c.num_experts):
                base = f'{prefix}mlp.experts.{expert}.'
                weights[expert].copy_(torch.cat([source.get(base + proj + '.weight').view(torch.uint8) for proj in projections]).view(torch.float8_e4m3fn))
                scales[expert].copy_(torch.cat([source.get(base + proj + '.weight_scale_inv') for proj in projections]))
            layer_state[f'mlp.experts.{bank}_weight'] = weights
            layer_state[f'mlp.experts.{bank}_scale_inv'] = scales
        return layer_state

    for index in range(resident_layers):
        backbone.model.layers[index].load_state_dict(load_layer(index), assign=True, strict=True)
    host_weights = {i: load_layer(i, torch.device('cpu')) for i in range(resident_layers, resident_layers + cached_layers)}

    def setup(self, max_batch_size, max_seq_len):
        for layer in self.model.layers:
            mixer = layer.self_attn if layer.layer_type == 'full_attention' else layer.linear_attn
            if layer.layer_type == 'full_attention':
                mixer.allocate_cache(max_batch_size, max_seq_len, dtype, device)
            else:
                mixer.allocate_cache(max_batch_size, dtype, device)
    def forward(self, input_ids=None, *, inputs_embeds=None, input_pos=None, static_kv_len=None):
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        width = config.text.rotary_dim
        inv = config.text.rope_theta ** (-torch.arange(0, width, 2, device=device, dtype=torch.float32) / width)
        phases = input_pos.float()[:, None] * inv
        phases = torch.cat((phases, phases), -1)
        context = LayerContext(input_pos=input_pos, cos=phases.cos(), sin=phases.sin(),
            kv_len=int(input_pos[-1]) + 1, continue_state=int(input_pos[0]) > 0, use_cache=True)
        for index, layer in enumerate(self.model.layers):
            if index < resident_layers:
                hidden = layer(hidden, context)
                continue
            layer_state = {k: v.to(device) for k, v in host_weights[index].items()} if index in host_weights else load_layer(index)
            layer.load_state_dict(layer_state, assign=True, strict=True)
            hidden = layer(hidden, context)
            layer.to_empty(device='meta')
            del layer_state
        return self.model.norm(hidden)
    backbone.setup_caches = MethodType(setup, backbone)
    backbone.forward = MethodType(forward, backbone)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resident-layers', type=int, default=24)
    p.add_argument('--host-cache-layers', type=int, default=0, help='Number of nonresident original FP8 layers cached in CPU RAM for faster PCIe streaming.')
    options = p.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(options.upstream))
    from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
    from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
    from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
    from fish_speech.models.text2semantic.fish_omni.layers import LayerContext
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, TOWER_CHECKPOINT_PREFIXES
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt, generate_drama3
    from tokenizers import Tokenizer
    raw_config = json.loads((options.source / 'config.json').read_text())
    config = FishOmniConfig.from_dict(raw_config)
    if not 0 <= options.resident_layers <= config.text.num_hidden_layers or not 0 <= options.host_cache_layers <= config.text.num_hidden_layers - options.resident_layers:
        raise ValueError('Invalid resident layer count')
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    with torch.device('meta'):
        backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=dtype)
        tower = FishOmniAudioDecoder(config, dtype=dtype)
    os.environ['FISH_DRAMA3_CUDA_GRAPH'] = '0'
    torch.manual_seed(options.seed)
    with ExitStack() as stack:
        source = Source(options.source, stack)
        for name, key in [('embed_tokens', 'model.language_model.embed_tokens.weight'), ('norm', 'model.language_model.norm.weight')]:
            getattr(backbone.model, name).load_state_dict({'weight': source.get(key).to(device)}, assign=True)
        backbone.lm_head.load_state_dict({'weight': source.get('lm_head.weight').to(device)}, assign=True)
        state = {}
        for key in source.mapping:
            for old, new in TOWER_CHECKPOINT_PREFIXES.items():
                if key.startswith(old):
                    state[new + key[len(old):]] = source.get(key).to(device)
        tower.load_state_dict(state, assign=True, strict=True)
        del state
        if options.host_cache_layers:
            prefixes = tuple(f'model.language_model.layers.{i}.' for i in range(options.resident_layers, options.resident_layers + options.host_cache_layers))
            required = sum(v['data_offsets'][1] - v['data_offsets'][0] for k, v in source.specs.items() if k.startswith(prefixes))
            available = psutil.virtual_memory().available
            if required + 2 * 1024**3 > available:
                raise RuntimeError(f'Original FP8 host cache needs {required} bytes plus 2 GiB workspace; only {available} bytes available. Reduce --host-cache-layers for a smaller CPU cache.')
        install_streaming_backbone(backbone, source, config, device, dtype, options.resident_layers, LayerContext, options.host_cache_layers)
        tokenizer = Tokenizer.from_file(str(options.source / 'tokenizer.json'))
        model = FishOmniModel(config, backbone, tower, tokenizer)
        prompt = build_drama3_prompt(options.text, tokenizer, config)
        started = time.monotonic()
        frames, reason = generate_drama3(model, prompt, max_new_tokens=options.max_frames, temperature=.7, top_p=.7, top_k=30)
        save_file({'codes': frames[:, 1:].T.contiguous()}, str(options.output / 'codes.safetensors'))
        report = dict(upstream_revision=UPSTREAM_REVISION, source_revision=SOURCE_REVISION, torch_version=str(torch.__version__),
                      process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(), text=options.text, seed=options.seed, frames=len(frames),
                      finish_reason=reason, seconds=time.monotonic()-started, residency='original FP8 with bounded resident prefix and layer streaming; upstream math unchanged', resident_layers=options.resident_layers, host_cache_layers=options.host_cache_layers)
        (options.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
