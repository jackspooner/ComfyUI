"""Separate NVFP4 weight and activation error on real first-layer MoE inputs.

Run only under coordinated exclusive GPU admission. This bounded component check
is not a speech-quality baseline or a replacement for upstream FP8 generation.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.convert_drama3 import Source, decode_fp8, SOURCE_REVISION
from comfy.fish_audio.config import FishOmniConfig
from comfy.fish_audio.model import DeltaNet
from comfy.fish_audio.prompt import build_drama3_prompt
from comfy.text_encoders.llama import RMSNorm
import comfy.ops


def relative_error(actual, expected):
    difference = actual.float() - expected.float()
    return dict(relative_rmse=(difference.square().sum() / expected.float().square().sum().clamp_min(1e-30)).sqrt().item(),
                max_abs=difference.abs().max().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    options = p.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    sys.path.insert(0, str(options.upstream))
    from fish_speech.models.text2semantic.fish_omni.layers import GatedDeltaNet, ZeroCenteredRMSNorm, LayerContext
    torch.set_grad_enabled(False)
    config = FishOmniConfig.from_dict(json.loads((options.source / 'config.json').read_text()))
    c, device, dtype = config.text, torch.device('cuda'), torch.bfloat16
    tokenizer = Tokenizer.from_file(str(options.source / 'tokenizer.json'))
    prompt = build_drama3_prompt('The quarterly report is due on Wednesday. Please finish it by noon.', tokenizer, config)
    with ExitStack() as stack:
        source = Source(options.source, stack)
        packed = stack.enter_context(safe_open(str(options.model), framework='pt'))
        prefix = 'model.language_model.layers.0.'
        mixer = DeltaNet(c, dtype, device, comfy.ops.manual_cast)
        mixer.load_state_dict({k[len(prefix+'linear_attn.'):]: source.get(k).to(device)
            for k in source.mapping if k.startswith(prefix+'linear_attn.')}, assign=True, strict=True)
        norms = [RMSNorm(c.hidden_size, eps=c.rms_norm_eps, add=True, dtype=dtype, device=device) for _ in range(2)]
        for norm, name in zip(norms, ('input_layernorm', 'post_attention_layernorm')):
            norm.load_state_dict({'weight': source.get(prefix+name+'.weight').to(device)}, assign=True)
        x = source.get('model.language_model.embed_tokens.weight')[prompt[:, 0]].to(device)[None]
        source_norm = ZeroCenteredRMSNorm(c.hidden_size, c.rms_norm_eps, dtype=dtype, device=device)
        source_norm.load_state_dict(norms[0].state_dict())
        norm_error = relative_error(norms[0](x), source_norm(x))
        reference = GatedDeltaNet(c, dtype=dtype, device=device)
        reference.load_state_dict(mixer.state_dict())
        reference.allocate_cache(1, dtype, device)
        normalized = source_norm(x)
        cache, port_errors, outputs = [], {}, []
        for start, end in ((0, len(prompt)-1), (len(prompt)-1, len(prompt))):
            context = LayerContext(input_pos=torch.arange(start, end, device=device), cos=None, sin=None,
                kv_len=end, continue_state=start > 0, use_cache=True)
            expected = reference(normalized[:, start:end], context)
            actual = mixer(normalized[:, start:end], start, cache, len(prompt))
            port_errors['prefill' if start == 0 else 'decode'] = relative_error(actual, expected)
            outputs.append(actual)
        x = norms[1](x + torch.cat(outputs, 1)).flatten(0, 1)
        router = F.linear(x, source.get(prefix+'mlp.gate.weight').to(device))
        experts = router[-1].topk(c.num_experts_per_tok).indices.tolist()
        results = {}
        for expert in experts:
            dense, weight_only, accelerated = [], [], []
            for bank, projections, inputs, outputs in [('gate_up_proj', ('gate_proj','up_proj'), c.hidden_size, 2*c.moe_intermediate_size),
                                                       ('down_proj', ('down_proj',), c.moe_intermediate_size, c.hidden_size)]:
                values = []
                for proj in projections:
                    key = prefix+f'mlp.experts.{expert}.{proj}'
                    values.append(decode_fp8(source.get(key+'.weight'), source.get(key+'.weight_scale_inv')).to(device))
                dense.append(torch.cat(values))
                key = 'backbone.layers.0.mlp.experts.'+bank+'.'
                state = {name: packed.get_slice(key+name)[expert].to(device) for name in ('weight','weight_scale','weight_scale_2')}
                state['comfy_quant'] = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
                modules = []
                for full_precision in (True, False):
                    op = comfy.ops.mixed_precision_ops(compute_dtype=dtype, full_precision_mm=full_precision)
                    module = op.Linear(inputs, outputs, bias=False, dtype=dtype, device=device)
                    module.load_state_dict(state.copy(), strict=True, assign=True)
                    modules.append(module)
                weight_only.append(modules[0]); accelerated.append(modules[1])
            def swiglu(value):
                gate, up = value.chunk(2, -1)
                return F.silu(gate) * up
            baseline = F.linear(swiglu(F.linear(x, dense[0])), dense[1])
            weights = weight_only[1](swiglu(weight_only[0](x)))
            fast = accelerated[1](swiglu(accelerated[0](x)))
            results[str(expert)] = dict(weight_error=relative_error(weights, baseline),
                activation_increment=relative_error(fast, weights), combined=relative_error(fast, baseline))
        report = dict(source_revision=SOURCE_REVISION, rows=len(prompt), experts=results,
            bf16_normalization_port_error=norm_error, bf16_delta_port_errors=port_errors,
            baseline='original FP8 weights dequantized to BF16; identical native first-layer post-attention inputs',
            limitation='one layer and prompt; excludes upstream dynamic FP8 activation error and full-speech quality')
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
