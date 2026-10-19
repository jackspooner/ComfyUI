"""Offline RVC export validation and weight-normalization materialization."""

import math
import re

import torch


def voice_config(checkpoint):
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("weight"), dict) or not isinstance(checkpoint.get("config"), (list, tuple)):
        raise ValueError("Expected an RVC inference export with weight/config; training checkpoints are unsupported.")
    config = list(checkpoint["config"])
    version = checkpoint.get("version", "v1")
    use_f0 = checkpoint.get("f0", 1)
    if len(config) != 18 or version not in ("v1", "v2") or use_f0 not in (0, 1):
        raise ValueError("Expected an RVC v1/v2 inference configuration.")
    if checkpoint.get("vocoder", "HiFi-GAN") != "HiFi-GAN" or config[9] != "1":
        raise ValueError("This native RVC path supports HiFi-GAN resblock 1 exports; MRF and RefineGAN need different architectures.")
    if config[-1] not in (32000, 40000, 48000) or math.prod(config[12]) != config[-1] // 100:
        raise ValueError("Expected 32/40/48 kHz synthesis with a 100 Hz content grid.")
    state = checkpoint["weight"]
    if "emb_g.weight" not in state or "enc_p.emb_phone.weight" not in state:
        raise ValueError("RVC export lacks speaker or content embeddings.")
    config[-3] = state["emb_g.weight"].shape[0]
    return config, version, bool(use_f0)


def fold_weights(state, positional=False):
    result, transforms = {}, {}
    consumed = set()
    for key, value in state.items():
        if key in consumed:
            continue
        match = re.match(r"(.*)\.(?:weight_g|parametrizations\.weight\.original0)$", key)
        if match:
            prefix = match[1]
            partner = prefix + (".weight_v" if key.endswith("weight_g") else ".parametrizations.weight.original1")
            if partner not in state:
                raise ValueError(f"Incomplete weight normalization: {key}")
            v, g = state[partner].float(), value.float()
            dim = 2 if positional and prefix == "encoder.pos_conv_embed.conv" else 0
            result[prefix + ".weight"] = torch._weight_norm(v, g, dim)
            transforms[prefix + ".weight"] = [key, partner, f"weight_norm dim={dim}, FP32 materialization"]
            consumed.add(partner)
        elif key.endswith((".weight_v", ".parametrizations.weight.original1")):
            partner = key.replace(".weight_v", ".weight_g").replace(".original1", ".original0")
            if partner not in state:
                raise ValueError(f"Incomplete weight normalization: {key}")
        else:
            mapped = re.sub(r"(norm_layers_[12]\.\d+)\.gamma$", r"\1.norm.weight", key)
            mapped = re.sub(r"(norm_layers_[12]\.\d+)\.beta$", r"\1.norm.bias", mapped)
            result[mapped] = value
            if mapped != key:
                transforms[mapped] = [key, "rename channel LayerNorm affine parameter"]
    return result, transforms
