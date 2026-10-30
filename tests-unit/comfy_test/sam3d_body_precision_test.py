"""CPU loader regressions: preserve dense checkpoint head precision and native ownership."""
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args
args.cpu = True

import comfy.memory_management
import comfy.ops
import comfy_extras.nodes_sam3d_body as nodes


class TinyBody(torch.nn.Module):
    """Two real native projections stand in for the large checkpoint's components."""
    def __init__(self, dtype, operations, backbone_dtype=None):
        super().__init__()
        self.backbone = operations.Linear(2, 2, bias=False, dtype=backbone_dtype)
        self.head = operations.Linear(2, 1, bias=False, dtype=dtype)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("backbone_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("quantized", [False, True])
def test_loader_preserves_fp32_head_values_and_native_residency(monkeypatch, lazy, backbone_dtype, quantized):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", lazy)
    monkeypatch.setattr(nodes, "SAM3DBody", TinyBody)
    monkeypatch.setattr(nodes.folder_paths, "get_full_path_or_raise", lambda *_: "fixture.safetensors")
    expected_dtype = torch.float32 if backbone_dtype == torch.bfloat16 and not quantized else backbone_dtype
    weights = torch.tensor([[1.0012, -1.0]], dtype=torch.float32)
    sd = {"backbone.weight": torch.eye(2, dtype=backbone_dtype), "head.weight": weights}
    monkeypatch.setattr(nodes.comfy.utils, "load_torch_file", lambda *_, **__: sd)
    quant_config = {"backbone": {"format": "fixture"}} if quantized else None
    monkeypatch.setattr(nodes.comfy.utils, "detect_layer_quantization", lambda *_: quant_config)
    quant_calls = []
    def quant_ops(config, dtype):
        quant_calls.append((config, dtype))
        return comfy.ops.disable_weight_init
    monkeypatch.setattr(nodes.comfy.ops, "mixed_precision_ops", quant_ops)
    monkeypatch.setattr(nodes.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes.comfy.model_management, "unet_dtype", lambda **_: backbone_dtype)
    monkeypatch.setattr(nodes.comfy.model_management, "unet_manual_cast", lambda *_: None)
    monkeypatch.setattr(nodes.comfy.model_management, "unet_offload_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes.comfy.ops, "pick_operations", lambda *_, **__: comfy.ops.disable_weight_init)
    monkeypatch.setattr(nodes.comfy.storage, "state_dict_fast_disk", lambda _: False)
    captured = {}

    def patcher(model, **options):
        captured.update(model=model, **options)
        return SimpleNamespace(model=model)

    monkeypatch.setattr(nodes.comfy.model_patcher, "CoreModelPatcher", patcher)
    nodes.SAM3DBody_Loader.execute("fixture.safetensors")
    model = captured["model"]
    # This is the dtype consumed by CoreModelPatcher's AIMDO geometry setup.
    resident_dtype = getattr(model.head, "weight_comfy_model_dtype", model.head.weight.dtype)
    assert resident_dtype == expected_dtype
    actual = torch.nn.functional.linear(torch.ones(1, 2, dtype=resident_dtype), model.head.weight.to(resident_dtype))
    expected = torch.nn.functional.linear(torch.ones(1, 2, dtype=expected_dtype), weights.to(expected_dtype))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.backbone_dtype == backbone_dtype
    assert getattr(model.backbone, "weight_comfy_model_dtype", model.backbone.weight.dtype) == backbone_dtype
    assert captured["load_device"] == captured["offload_device"] == torch.device("cpu")
    assert captured["fast_disk"] is False
    assert quant_calls == ([(quant_config, backbone_dtype)] if quantized else [])


def test_constructor_keeps_backbone_dtype_separate_on_meta():
    model = nodes.SAM3DBody(device="meta", dtype=torch.float32, backbone_dtype=torch.bfloat16, operations=comfy.ops.disable_weight_init)
    assert next(model.backbone.parameters()).dtype == torch.bfloat16
    assert next(model.decoder.parameters()).dtype == torch.float32
    assert next(model.head_camera.parameters()).dtype == torch.float32


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("supports_half", [False, True])
def test_device_default_selects_dense_checkpoint_precision(monkeypatch, quantized, supports_half):
    # Exercise the real selection policy: mocking unet_dtype itself previously
    # hid its FP16-first CUDA default, bypassing the dense BF16 head policy.
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
    monkeypatch.setattr(nodes, "SAM3DBody", TinyBody)
    monkeypatch.setattr(nodes.folder_paths, "get_full_path_or_raise", lambda *_: "fixture.safetensors")
    sd = {"backbone.weight": torch.eye(2, dtype=torch.bfloat16),
          "head.weight": torch.tensor([[1.0012, -1.0]], dtype=torch.float32)}
    monkeypatch.setattr(nodes.comfy.utils, "load_torch_file", lambda *_, **__: sd)
    monkeypatch.setattr(nodes.comfy.utils, "detect_layer_quantization", lambda *_: {"fixture": {}} if quantized else None)
    monkeypatch.setattr(nodes.comfy.ops, "mixed_precision_ops", lambda *_: comfy.ops.disable_weight_init)
    monkeypatch.setattr(nodes.comfy.ops, "pick_operations", lambda *_, **__: comfy.ops.disable_weight_init)
    monkeypatch.setattr(nodes.comfy.model_management, "get_torch_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(nodes.comfy.model_management, "should_use_fp16", lambda **_: supports_half)
    monkeypatch.setattr(nodes.comfy.model_management, "should_use_bf16", lambda *_, **__: supports_half)
    monkeypatch.setattr(nodes.comfy.model_management, "unet_manual_cast", lambda *_: None)
    monkeypatch.setattr(nodes.comfy.model_management, "unet_offload_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes.comfy.storage, "state_dict_fast_disk", lambda _: False)
    captured = {}
    def patcher(model, **options):
        captured['model'] = model
        return SimpleNamespace(model=model)
    monkeypatch.setattr(nodes.comfy.model_patcher, "CoreModelPatcher", patcher)
    nodes.SAM3DBody_Loader.execute("fixture.safetensors")
    model = captured['model']
    assert model.backbone_dtype == (torch.float32 if not supports_half else torch.float16 if quantized else torch.bfloat16)
    assert model.head.weight.dtype == (torch.float16 if quantized and supports_half else torch.float32)
