from types import SimpleNamespace
import importlib
import os
import sys
import types

import pytest
import torch
from torch import nn

from comfy.cli_args import args

args.cpu = True

from comfy import model_base, utils
from comfy.ldm.minimax import model as h3_model
from comfy.ldm.minimax.model import mask_row_values


@pytest.mark.parametrize("value, expected", [(0.0, 0.0), (0.5, 0.5), (0.9995, 4094 / 4096), (0.9997, 4095 / 4096), (1.0, 1.0)])
def test_mask_grid_preserves_near_one_conditions(value, expected):
    model = model_base.MiniMaxH3.__new__(model_base.MiniMaxH3)
    model.diffusion_model = SimpleNamespace(patch_size=(1, 2, 2))
    video = torch.full((1, 1, 1, 2, 2), value)
    audio = torch.full((1, 1, 2, 1), value)
    packed, shapes = utils.pack_latents((video, audio))

    grids = model._token_grid_masks(packed, shapes)
    conditions = model._denoise_mask_values(packed, shapes)
    for grid in grids:
        torch.testing.assert_close(grid, torch.full_like(grid, expected), rtol=0, atol=0)
    if value == 1.0:
        assert conditions == {}
        assert mask_row_values(grids[0][0, 0], 1, 2, 2) is None
    else:
        assert set(conditions) == {"denoise_mask", "audio_denoise_mask"}
        torch.testing.assert_close(mask_row_values(grids[0][0, 0], 1, 2, 2), torch.tensor([expected]), rtol=0, atol=0)


@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("model_class", [model_base.BaseModel, model_base.MiniMaxH3])
def test_condition_transport_keeps_h3_masks_float32(compute_dtype, model_class):
    captured = {}

    def capture(x, timestep, **kwargs):
        captured.update(kwargs)
        assert x.dtype == compute_dtype
        return x

    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.diffusion_model = capture
    model.current_patcher = None
    model.model_sampling = SimpleNamespace(calculate_input=lambda sigma, x: x, timestep=lambda t: t,
                                          calculate_denoised=lambda sigma, output, x: output)
    model.get_dtype_inference = lambda: compute_dtype
    model.process_timestep = lambda t, **kwargs: t
    mask = torch.tensor([4094 / 4096, 4095 / 4096]).reshape(1, 1, 1, 2)
    tags = torch.tensor([1, 2], dtype=torch.long)
    payload = {"mask": mask}
    model._apply_model(torch.zeros_like(mask), torch.ones(1), denoise_mask=mask, audio_denoise_mask=mask,
                       regular=mask, regular_list=[mask, tags], tags=tags, minimax_payload=payload)

    expected_dtype = torch.float32 if model_class is model_base.MiniMaxH3 else compute_dtype
    for name in ("denoise_mask", "audio_denoise_mask"):
        assert captured[name].dtype == expected_dtype
        torch.testing.assert_close(captured[name], mask.to(expected_dtype), rtol=0, atol=0)
    assert captured["regular"].dtype == compute_dtype
    assert captured["regular_list"][0].dtype == compute_dtype
    assert captured["regular_list"][1].dtype == torch.long
    assert captured["tags"].dtype == torch.long
    assert captured["minimax_payload"] is payload


@pytest.mark.parametrize("value", [4094 / 4096, 4095 / 4096, 1.0])
@pytest.mark.parametrize("stream", ["video", "audio", "both"])
def test_audio_and_video_use_fractional_row_timestep(value, stream):
    model = h3_model.MiniMaxH3Model(
        hidden_size=8, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8,
        latents_dim=2, audio_latents_dim=3, text_dim=5,
        timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
        dtype=torch.float32, device="cpu", operations=nn,
    )
    model.sigma_shift_video = model.sigma_shift_audio = 3.0
    captured = []
    handle = model.time_embedder.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].clone()))
    model._forward(
        [torch.zeros(1, 2, 1, 2, 2), torch.zeros(1, 3, 2, 1)],
        torch.tensor([500.0]), torch.zeros(1, 1, 8),
        denoise_mask=torch.full((1, 1, 1, 2, 2), value if stream in ("video", "both") else 1.0),
        audio_denoise_mask=torch.full((1, 1, 2, 1), value if stream in ("audio", "both") else 1.0),
    )
    handle.remove()
    expected = [0.5] if value == 1.0 else [0.5, 1.0 - value * 0.5]
    torch.testing.assert_close(captured[0], torch.tensor(expected), rtol=0, atol=0)


@pytest.mark.skipif(not os.environ.get("H3_MOTION_CONTEXT_PATH"), reason="Optional installed Motion-Context compatibility probe")
def test_installed_compatibility_probe_installs_no_patch(monkeypatch):
    package = types.ModuleType("h3_precision_fixture")
    package.__path__ = [os.environ["H3_MOTION_CONTEXT_PATH"]]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    probe = importlib.import_module(f"{package.__name__}.h3_mask_precision")
    owners = (model_base.MiniMaxH3, h3_model.MiniMaxH3Model)
    before = [{name: value for name, value in vars(owner).items() if callable(value)} for owner in owners]
    before_rows = h3_model.mask_row_values
    status = probe.ensure_h3_mask_precision()
    assert status["precision_ready"], status
    assert not any(value for name, value in status.items() if name.endswith("_patch_active")), status
    assert before == [{name: value for name, value in vars(owner).items() if callable(value)} for owner in owners]
    assert h3_model.mask_row_values is before_rows
