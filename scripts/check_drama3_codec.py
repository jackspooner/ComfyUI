"""Compare the native codec with pinned offline Fish Speech on CPU."""
import argparse
import json
from pathlib import Path
import sys
import torch
import soundfile as sf
from safetensors.torch import load_file
from hydra.utils import instantiate
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy.cli_args import args
args.cpu = True
import comfy.ops
from comfy.fish_audio.codec import Codec
from scripts.convert_drama3 import codec_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--codes', type=Path, help='Optional original generator [10, frames] safetensors to compare and render.')
    options = p.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    supplied_codes = load_file(str(options.codes))['codes'][None] if options.codes else None
    capacity = max(128, supplied_codes.shape[-1] if supplied_codes is not None else 0)
    sys.path.insert(0, str(options.upstream))
    config = OmegaConf.load(options.upstream / 'fish_speech/configs/modded_dac_vq.yaml')
    # Build source structure on meta. Its multi-gigabyte nonpersistent tables
    # are replaced by bounded tables for this fixture, without changing math.
    with torch.device('meta'):
        reference = instantiate(config)
        native = Codec(torch.bfloat16, 'meta', comfy.ops.manual_cast)
    state = torch.load(str(options.source), map_location='cpu', mmap=True, weights_only=True)
    state = state.get('state_dict', state)
    state = {k: v for k, v in state.items() if not k.endswith(('.freqs_cis', '.causal_mask'))}
    reference.load_state_dict(state, assign=True, strict=True)
    from fish_speech.models.dac.modded_dac import Transformer, precompute_freqs_cis
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
    for module in reference.modules():
        if isinstance(module, Transformer):
            module.freqs_cis = precompute_freqs_cis(capacity, 64, 10000, torch.bfloat16)
            module.causal_mask = torch.ones(capacity, capacity, dtype=torch.bool).tril()
    reference = reference.float()
    for module in list(reference.modules()):
        if is_parametrized(module, 'weight'):
            remove_parametrizations(module, 'weight', leave_parametrized=True)
        elif hasattr(module, 'weight_g'):
            torch.nn.utils.remove_weight_norm(module)
    reference = reference.to(dtype=torch.bfloat16).eval()
    native.load_state_dict(codec_weights(options.source)[0], strict=True, assign=True)
    reference = reference.float()
    native = native.float()
    torch.set_grad_enabled(False)
    torch.manual_seed(81)
    codes = supplied_codes if supplied_codes is not None else torch.randint(0, 4096, (1, 10, 4))
    expected = reference.from_indices(codes.clone())
    actual = native.decode(codes.clone())
    report = {'decode_shape': list(actual.shape), 'decode_max_abs': (actual.float()-expected.float()).abs().max().item(),
              'decode_rmse': (actual.float()-expected.float()).square().mean().sqrt().item()}
    # Reference encoding runs the complete real checkpoint with a short waveform.
    waveform = .1 * torch.randn(1, 1, 8192, dtype=torch.float32)
    expected_codes = reference.encode(waveform)[0]
    actual_codes = native.encode(waveform)
    report['reference_codes_equal'] = torch.equal(actual_codes, expected_codes)
    report['code_agreement'] = (actual_codes == expected_codes).float().mean().item()
    if supplied_codes is not None:
        sf.write(options.output / 'reference.wav', expected[0, 0].cpu().numpy(), 44100, subtype='FLOAT')
        sf.write(options.output / 'native-fp32.wav', actual[0, 0].cpu().numpy(), 44100, subtype='FLOAT')
        report['code_source'] = str(options.codes)
    (options.output / 'report.json').write_text(json.dumps(report, indent=2))
    torch.save({'codes': codes, 'reference': expected, 'native': actual, 'reference_codes': expected_codes, 'native_codes': actual_codes}, options.output / 'fixtures.pt')
    print(json.dumps(report))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    if not report['reference_codes_equal']:
        raise AssertionError('Codec reference encoding disagrees')


if __name__ == '__main__':
    main()
