"""Offline Omni reference using pinned Transformers and source-format decoders."""
import argparse
from contextlib import ExitStack
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRotaryEmbedding, Qwen3OmniMoeVisionRotaryEmbedding, SinusoidsPositionEmbedding)
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
try:
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
except ImportError:
    NVFP4QTensor = None

from scripts.qwen3_omni_checkpoint import FP8_REVISION, PROJECTION_SAMPLES, SOURCE_REVISION, SourceTensors, sha256_file, validate_source


class PackedLinear(nn.Module):
    def __init__(self, get_tensor, prefix, device):
        super().__init__()
        for name in ('weight', 'weight_scale', 'weight_scale_2'):
            self.register_buffer(name, get_tensor(prefix + '.' + name).to(device))
        self.shape = (self.weight.shape[0], self.weight.shape[1] * 2)
        self.register_buffer('bias', get_tensor(prefix + '.bias').to(device) if get_tensor.contains(prefix + '.bias') else None)

    def decode(self, dtype=torch.bfloat16):
        packed = NVFP4QTensor(self.shape, dtype, self.weight)
        return packed.dequantize(dtype=dtype, scale=self.weight_scale, double_scale=self.weight_scale_2,
                                 block_sizes={-1: 16})

    def forward(self, inputs):
        return F.linear(inputs, self.decode(inputs.dtype), self.bias)


class ExpertBank(nn.Module):
    """Preserve upstream expert dispatch; decode only the indexed expert's matrices."""
    def __init__(self, get_tensor, prefix, count, projections, device, projection_class=PackedLinear):
        super().__init__()
        self.experts = nn.ModuleList([nn.ModuleList([projection_class(get_tensor, f'{prefix}.{i}.{proj}', device)
                                                   for proj in projections]) for i in range(count)])

    def __getitem__(self, index):
        parts = [projection.decode() for projection in self.experts[int(index)]]
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]


class BlockFP8Linear(nn.Module):
    """Keep source FP8 buffers file-backed on CPU and decode selected matrices."""
    def __init__(self, get_tensor, prefix, device):
        super().__init__()
        self.register_buffer('weight', get_tensor(prefix + '.weight'))
        self.register_buffer('weight_scale_inv', get_tensor(prefix + '.weight_scale_inv'))
        self.register_buffer('bias', get_tensor(prefix + '.bias').to(device) if get_tensor.contains(prefix + '.bias') else None)
        self.compute_device = device
        self.shape = tuple(self.weight.shape)

    def decode(self, dtype=torch.bfloat16):
        # This pinned upstream decoder is independent of the native layout.
        return Fp8Dequantize(None)._dequantize_one(self.weight.to(self.compute_device), self.weight_scale_inv.to(self.compute_device), dtype)

    def forward(self, inputs):
        return F.linear(inputs, self.decode(inputs.dtype), self.bias)


def load_reference(source, device):
    config = Qwen3OmniMoeConfig.from_pretrained(source, local_files_only=True).thinker_config
    config._attn_implementation = 'sdpa'
    config.text_config._experts_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    with ExitStack() as stack:
        get_tensor = SourceTensors(source, stack)
        projection_class = BlockFP8Linear if get_tensor.contains('thinker.model.layers.0.self_attn.q_proj.weight_scale_inv') else PackedLinear
        for name, module in list(model.named_modules()):
            prefix = 'thinker.' + name
            if isinstance(module, nn.Linear) and (get_tensor.contains(prefix + '.weight_scale') or get_tensor.contains(prefix + '.weight_scale_inv')):
                parent, attribute = name.rsplit('.', 1)
                setattr(model.get_submodule(parent), attribute, projection_class(get_tensor, prefix, device))
        for i, layer in enumerate(model.model.layers):
            experts = layer.mlp.experts
            for name, projections in [('gate_up_proj', ('gate_proj', 'up_proj')), ('down_proj', ('down_proj',))]:
                delattr(experts, name)
                setattr(experts, name, ExpertBank(get_tensor, f'thinker.model.layers.{i}.mlp.experts',
                                                  config.text_config.num_experts, projections, device, projection_class))
        for name, parameter in list(model.named_parameters()):
            value = get_tensor('thinker.' + name).to(device=device).clone()
            if list(value.shape) != list(parameter.shape):
                raise ValueError(f'Reference parameter shape differs: {name}')
            parent, attribute = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attribute, nn.Parameter(value))
    # Recreate only upstream non-persistent deterministic buffers omitted by meta init.
    for name, module in list(model.named_modules()):
        replacement = None
        if isinstance(module, SinusoidsPositionEmbedding):
            replacement = SinusoidsPositionEmbedding(module.length, module.channels, module.max_timescale)
        elif isinstance(module, (Qwen3OmniMoeThinkerTextRotaryEmbedding, Qwen3OmniMoeVisionRotaryEmbedding)):
            replacement = type(module)(module.config)
        if replacement is not None:
            parent, attribute = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attribute, replacement.to(device))
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError('Reference still contains unloaded meta tensors')
    return model.eval()


def prepare_case(processor, case):
    content = []
    images = [torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2, 0, 1).float() / 255
              for path in case.get('images', [])]
    if case.get('image_tensor'):
        batch = torch.from_numpy(np.load(case['image_tensor'], allow_pickle=False))
        images = list(batch.permute(0, 3, 1, 2).unbind())
    content.extend({'type': 'image', 'image': image} for image in images)
    audio = None
    if case.get('audio'):
        audio, rate = sf.read(case['audio'], dtype='float32', always_2d=True)
        if rate != 16000:
            raise ValueError('Reference fixtures must supply independently resampled 16 kHz audio')
        audio = audio.mean(axis=1)
        padded_length = (max(201, len(audio)) + 159) // 160 * 160
        audio = np.pad(audio, (0, padded_length - len(audio)))
        content.append({'type': 'audio', 'audio': audio})
    content.append({'type': 'text', 'text': case['prompt']})
    messages = []
    if case.get('system_prompt'):
        messages.append({'role': 'system', 'content': case['system_prompt']})
    messages.append({'role': 'user', 'content': content})
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return processor(text=text, images=images or None, audio=[audio] if audio is not None else None,
                     return_tensors='pt', padding=True, do_rescale=False), text


def component_evidence(source, output, integrity, versions):
    report = {'versions': versions, 'source': integrity, 'device': 'cpu',
              'decoder': 'Transformers Fp8Dequantize' if integrity['source_revision'] == FP8_REVISION else 'ModelOpt NVFP4QTensor.dequantize, fast=False', 'projections': []}
    with ExitStack() as stack:
        get_tensor = SourceTensors(source, stack)
        for index, prefix in enumerate(PROJECTION_SAMPLES):
            if get_tensor.contains(prefix + '.weight_scale_inv'):
                projection = BlockFP8Linear(get_tensor, prefix, 'cpu')
                values = {'fp32': projection.decode(torch.float32), 'bf16': projection.decode(torch.bfloat16)}
            elif get_tensor.contains(prefix + '.weight_scale'):
                projection = PackedLinear(get_tensor, prefix, 'cpu')
                values = {'fp32': projection.decode(torch.float32), 'bf16': projection.decode(torch.bfloat16)}
            else:
                values = {'fp32': get_tensor(prefix + '.weight').float(), 'bf16': get_tensor(prefix + '.weight')}
            path = output / f'projection_{index:02d}.pt'
            torch.save(values, path)
            report['projections'].append({'prefix': prefix, 'file': path.name, 'sha256': sha256_file(path),
                                          'shape': list(values['bf16'].shape)})
            print(prefix, 'decoded', flush=True)
    (output / 'components.json').write_text(json.dumps(report, indent=2) + '\n')


def capture_activations(model):
    values, handles = {}, []

    def save(prefix, inputs):
        if prefix not in values and inputs.numel():
            values[prefix] = inputs.reshape(-1, inputs.shape[-1])[:128].detach().cpu().clone()

    for prefix in PROJECTION_SAMPLES:
        name = prefix.removeprefix('thinker.')
        if '.experts.' not in name:
            def projection_hook(module, args, prefix=prefix):
                save(prefix, args[0])
            handles.append(model.get_submodule(name).register_forward_pre_hook(projection_hook))
    for layer_index in (0, 23, 47):
        def expert_hook(module, args, layer_index=layer_index):
            hidden, selected, _ = args
            for expert in (0, 63, 127):
                prefix = f'thinker.model.layers.{layer_index}.mlp.experts.{expert}'
                if prefix + '.down_proj' in values:
                    continue
                tokens, _ = torch.where(selected == expert)
                if not tokens.numel():
                    continue
                inputs = hidden[tokens[:128]]
                save(prefix + '.gate_proj', inputs)
                save(prefix + '.up_proj', inputs)
                gate, up = F.linear(inputs, module.gate_up_proj[expert]).chunk(2, dim=-1)
                save(prefix + '.down_proj', module.act_fn(gate) * up)
        handles.append(model.model.layers[layer_index].mlp.experts.register_forward_pre_hook(expert_hook))
    return values, handles


def capture_parity(model):
    values, handles = {}, []

    def audio_hook(module, args, output):
        values['audio_features'] = output.last_hidden_state.detach().cpu().clone()

    def vision_hook(module, args, output):
        values['vision_features'] = output.pooler_output.detach().cpu().clone()
        for index, features in enumerate(output.deepstack_features):
            values[f'vision_deepstack_{index}'] = features.detach().cpu().clone()

    def position_hook(module, args):
        if 'position_ids' not in values:
            values['position_ids'] = args[1].detach().cpu().clone()

    handles.append(model.audio_tower.register_forward_hook(audio_hook))
    handles.append(model.visual.register_forward_hook(vision_hook))
    handles.append(model.model.rotary_emb.register_forward_pre_hook(position_hook))
    for index, layer in enumerate(model.model.layers):
        def router_hook(module, args, output, index=index):
            if f'router_{index}_indices' not in values:
                for name, value in zip(('logits', 'weights', 'indices'), output):
                    values[f'router_{index}_{name}'] = value.detach().cpu().clone()
        handles.append(layer.mlp.gate.register_forward_hook(router_hook))
    return values, handles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--components', action='store_true', help='Decode fixed component probes on CPU instead of loading the full model')
    parser.add_argument('--capture-activations', action='store_true', help='Record up to 128 actual input rows per component probe for native benchmarks')
    parser.add_argument('--capture-parity', action='store_true', help='Record encoder features, positions and prefill routing for native parity')
    args = parser.parse_args()
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers')}
    if args.source_revision == SOURCE_REVISION:
        versions['nvidia-modelopt'] = importlib.metadata.version('nvidia-modelopt')
    if versions['transformers'] != '5.20.0' or versions.get('nvidia-modelopt', '0.42.0') != '0.42.0':
        raise ValueError(f'Reference dependency pins differ: {versions}')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    _, integrity = validate_source(args.source, args.source_revision)
    if args.source_revision not in (SOURCE_REVISION, FP8_REVISION):
        raise ValueError('Unverified reference checkpoint revision')
    if args.components:
        component_evidence(args.source, args.output, integrity, versions)
        return
    if args.cases is None:
        parser.error('--cases is required for full-model reference generation')
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.source, local_files_only=True)
    if processor.tokenizer.eos_token_id is None:
        raise ValueError('Reference tokenizer has no EOS token')
    model = load_reference(args.source, args.device)
    activations, handles = capture_activations(model) if args.capture_activations else ({}, [])
    report = {'versions': versions, 'source': integrity, 'dtype': 'bfloat16',
              'source_code_sha256': {'harness': sha256_file(__file__),
                  'checkpoint_reader': sha256_file(Path(__file__).with_name('qwen3_omni_checkpoint.py')),
                  'upstream_model': sha256_file(inspect.getfile(Qwen3OmniMoeThinkerForConditionalGeneration)),
                  'upstream_processor': sha256_file(inspect.getfile(Qwen3OmniMoeProcessor)),
                  'weight_decoder': sha256_file(inspect.getfile(Fp8Dequantize if args.source_revision == FP8_REVISION else NVFP4QTensor))},
              'image_input': 'ComfyUI-equivalent FP32 CHW RGB in [0,1]; upstream processor do_rescale=False',
              'audio_padding': 'Minimal whole 160-sample hop, with length above STFT reflection padding',
              'reference': 'Upstream Transformers with independently decoded block FP8 weights, CPU weight offload, BF16 compute/cache; FP8 activation parity unverified' if args.source_revision == FP8_REVISION else 'Upstream Transformers with ModelOpt decoded weights; publisher W4A8 parity unverified',
              'eos_token_id': processor.tokenizer.eos_token_id, 'pad_token_id': processor.tokenizer.pad_token_id, 'cases': []}
    for case in json.loads(args.cases.read_text()):
        inputs, template = prepare_case(processor, case)
        inputs = inputs.to(args.device)
        # The HF processor returns FP32 media; upstream encoder owners select compute dtype.
        start = time.perf_counter()
        parity, parity_handles = capture_parity(model) if args.capture_parity else ({}, [])
        try:
            output = model.generate(**inputs, max_new_tokens=case.get('max_new_tokens', 512), do_sample=False,
                                    use_cache=True, return_dict_in_generate=True, output_logits=True,
                                    eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
        finally:
            for handle in parity_handles:
                handle.remove()
        if args.device == 'cuda':
            torch.cuda.synchronize()
        ids = output.sequences[0, inputs.input_ids.shape[-1]:].tolist()
        raw = processor.tokenizer.decode(ids, skip_special_tokens=True)
        reasoning, separator, final = raw.partition('</think>')
        if not separator:
            final = '' if reasoning.lstrip().startswith('<think>') else raw
        result = {'id': case['id'], 'input_ids': inputs.input_ids[0].tolist(), 'generated_ids': ids,
                  'text': final.strip(), 'raw_text': raw, 'template': template,
                  'eos_reached': bool(ids and ids[-1] == processor.tokenizer.eos_token_id),
                  'elapsed_seconds': time.perf_counter() - start,
                  'images': [{'path': path, 'sha256': sha256_file(path)} for path in case.get('images', [])]}
        lowered = result['text'].casefold()
        result['assertions'] = [{'alternatives': alternatives, 'passed': any(value.casefold() in lowered for value in alternatives)}
                                for alternatives in case.get('assert_all', [])]
        result['fixture_passed'] = all(assertion['passed'] for assertion in result['assertions'])
        evidence = {key: value.cpu() for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        evidence.update(parity)
        evidence['decode_logits'] = torch.stack([value[0].float().cpu() for value in output.logits[:8]])
        torch.save(evidence, args.output / (case['id'] + '.pt'))
        if case.get('audio'):
            result['audio'] = {'path': case['audio'], 'sha256': sha256_file(case['audio'])}
        if case.get('image_tensor'):
            result['image_tensor'] = {'path': case['image_tensor'], 'sha256': sha256_file(case['image_tensor'])}
        for key in ('image_grid_thw', 'feature_attention_mask'):
            if key in inputs:
                result[key] = inputs[key].tolist()
        report['cases'].append(result)
        if args.capture_activations:
            path = args.output / 'activations.pt'
            torch.save(activations, path)
            report['activations'] = {'file': path.name, 'sha256': sha256_file(path),
                                     'shapes': {key: list(value.shape) for key, value in activations.items()},
                                     'missing_probes': sorted(set(PROJECTION_SAMPLES) - set(activations))}
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(case['id'], result['text'], flush=True)
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()
