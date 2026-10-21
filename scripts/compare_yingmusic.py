"""Compare native YingMusic math with definitions extracted from the pinned upstream source.

Only mathematical class definitions are evaluated. The upstream package's training,
Hub, filesystem-relative tokenizer and model-loading entrypoints are not imported.
"""
import argparse
import importlib.util
import ast
import math
from pathlib import Path
import sys
from typing import Optional
import torch
from torch import nn, autocast
from torch.nn import functional as F
from einops import rearrange

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-code", required=True, type=Path)
options = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = [sys.argv[0], '--cpu']
import comfy.ops
from comfy.ldm.yingmusic.model import YingMusic

root = options.source_code / "src/YingMusicSinger/models"
namespace = dict(torch=torch, nn=nn, Module=nn.Module, math=math, F=F, Optional=Optional,
                 rearrange=rearrange, arange=torch.arange, stack=torch.stack, cat=torch.cat,
                 autocast=autocast, exists=lambda x: x is not None)
def definitions(path, names):
    tree = ast.parse(Path(path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    code = 'from __future__ import annotations\n' + '\n'.join(ast.unparse(node) for node in selected)
    exec(compile(code, str(path), 'exec'), namespace)
definitions(Path(importlib.util.find_spec("x_transformers").origin).with_name("x_transformers.py"), {'RotaryEmbedding','rotate_half','apply_rotary_pos_emb'})
definitions(root / 'modules.py', {'SinusPositionEmbedding','ConvPositionEmbedding','precompute_freqs_cis','get_pos_embed_indices','GRN','ConvNeXtV2Block','AdaLayerNorm','AdaLayerNorm_Final','FeedForward','Attention','AttnProcessor','DiTBlock','TimestepGuidanceEmbedding'})
definitions(root / 'dit.py', {'TextEmbedding','InputEmbedding','DiT'})
torch.manual_seed(19)
reference = namespace['DiT'](dim=64, depth=2, heads=2, dim_head=32, ff_mult=2, mel_dim=64, text_num_embeds=373, text_dim=32, text_mask_padding=False, conv_layers=2, dropout=0.0).eval()
# Upstream zero-initializes its modulation/output: randomize to exercise every block.
for name, param in reference.named_parameters():
    if name.endswith('weight') and param.ndim == 1:
        nn.init.uniform_(param, .7, 1.3)
    else:
        nn.init.normal_(param, std=.08)
native = YingMusic(dim=64, depth=2, heads=2, text_dim=32, text_depth=2, dtype=torch.float32, device='cpu', operations=comfy.ops.manual_cast)
native.load_state_dict(reference.state_dict(), strict=True)
for frames in (7, 29):
    x, cond = torch.randn(1, frames, 64), torch.randn(1, frames, 64)
    midi, tokens = torch.randn(1, frames, 128), torch.randint(0, 373, (1, frames))
    time = torch.tensor([.37])
    for null in (False, True):
        expected, _ = reference(x, cond, tokens, time, midi=midi, drop_audio_cond=null, drop_text=null, drop_midi=null)
        context = native.preprocess_conditioning(tokens, midi, cond.transpose(1, 2), unconditioned=null)
        actual = -native(x.transpose(1, 2), time, context).transpose(1, 2)
        error = (actual - expected).abs().max().item()
        print(dict(frames=frames, null=null, max_error=error), flush=True)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
from comfy.audio_encoders.yingmusic import MelodyEncoder, prepare_melody
# Extract the upstream mathematical definitions without its package-level training dependencies.
definitions(root.parent / 'melody/Gconform.py', {'GLU','conform_conv','Attention','conform_ffn','conform_blocke','Gcf','Gmidi_conform'})
reference = namespace['Gmidi_conform'](lay=2, dim=64, indim=80, outdim=128, use_lay_skip=True, attention_heads=2, attention_heads_dim=32).eval()
native = MelodyEncoder(dim=64, depth=2, heads=2, dtype=torch.float32, device='cpu', operations=comfy.ops.manual_cast).eval()
native.load_state_dict({**reference.state_dict(), "mel_basis": torch.empty(80, 1025)}, strict=True)
x = torch.randn(1, 37, 80)
a, b = reference(x), native(x)
for name, expected, actual in zip(('midi', 'boundary'), a, b):
    print(dict(component=name, max_error=(actual - expected).abs().max().item()), flush=True)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
definitions(root.parent / 'melody/SmoothMelody.py', {'MIDIFuzzDisturb'})
expected = F.interpolate(a[0].transpose(1,2), size=29, mode='linear', align_corners=False).transpose(1,2)
expected = namespace['MIDIFuzzDisturb'](drop_prob=[1,9], noise_scale=0, blur_kernel=0, drop_type='equal_space')(expected)
expected[:, :5] = 0
torch.testing.assert_close(prepare_melody(a[0], 29, 5), expected)
