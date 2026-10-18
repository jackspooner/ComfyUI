#!/usr/bin/env python3
"""Run pinned upstream inference offline in a separate validation environment."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import OmegaConf


def run(args):
    sys.path.insert(0, str(args.upstream))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from soulxsinger.models.soulxsinger import SoulXSinger
    from soulxsinger.utils.data_processor import DataProcessor
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = OmegaConf.load(args.models / "config.yaml")
    example = args.upstream / "example/audio"
    if args.task == "svc":
        from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
        from soulxsinger.models.modules import whisper_encoder as wh
        feature_loader = wh.WhisperFeatureExtractor.from_pretrained
        model_loader = wh.WhisperModel.from_pretrained
        with patch.object(wh.WhisperFeatureExtractor, "from_pretrained", side_effect=lambda *a, **kw: feature_loader(str(args.whisper_source), local_files_only=True)), \
             patch.object(wh.WhisperModel, "from_pretrained", side_effect=lambda *a, **kw: model_loader(str(args.whisper_source), local_files_only=True)), \
             patch("soulxsinger.models.soulxsinger_svc.WhisperEncoder", side_effect=lambda: wh.WhisperEncoder(device=args.device)):
            model = SoulXSingerSVC(cfg)
        filename = "model-svc.pt"
    else:
        model = SoulXSinger(cfg)
        filename = "model.pt"
    state = torch.load(args.models / filename, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)
    del state
    if args.precision == "float16":
        model.half()
        model.mel.float()
    processor = DataProcessor(480, 24000, str(args.upstream / "soulxsinger/utils/phoneme/phone_set.json"), args.device)
    pt_meta = json.loads((example / "en_prompt.json").read_text())[0]
    gt_meta = json.loads((example / "en_target.json").read_text())[0]
    # Decode independently of torchaudio's codec backend; model DSP is unchanged.
    waveform, rate = sf.read(example / "en_prompt.mp3", dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T).mean(0, keepdim=True)
    waveform = torchaudio.functional.resample(waveform, rate, 24000)
    pt = processor.process(dict(pt_meta))
    gt = processor.process(dict(gt_meta))
    pt["waveform"] = waveform.to(args.device)[..., :pt["mel2note"].shape[1] * 480]
    captured = {}
    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().float().cpu().numpy()
        return hook
    model.mel.register_forward_hook(capture("reference_mel"))
    model.cfm_decoder.register_forward_hook(capture("unused_training_forward"))
    original = model.cfm_decoder.reverse_diffusion
    def capture_mel(*pos, **kw):
        output = original(*pos, **kw)
        captured["generated_mel"] = output.detach().float().cpu().numpy()
        return output
    model.cfm_decoder.reverse_diffusion = capture_mel
    torch.manual_seed(args.seed)
    started = time.monotonic()
    if args.task == "svc":
        target_waveform, rate = sf.read(example / "en_target.mp3", dtype="float32", always_2d=True)
        target_waveform = torchaudio.functional.resample(torch.from_numpy(target_waveform.T).mean(0, keepdim=True), rate, 24000).to(args.device)
        target_waveform = target_waveform[..., :gt["f0"].shape[-1] * 480]
        audio, _ = model.infer(pt["waveform"], target_waveform, pt["f0"], gt["f0"],
                               auto_shift=False, pitch_shift=0, n_steps=args.steps, cfg=3, use_fp16=args.precision == "float16")
    else:
        audio = model.infer({"prompt": pt, "target": gt}, auto_shift=False, pitch_shift=0,
                            n_steps=args.steps, cfg=3, control=args.control, use_fp16=args.precision == "float16")
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    waveform = audio.squeeze().float().cpu().numpy()
    sf.write(args.output / "upstream.wav", waveform, 24000, subtype="FLOAT")
    np.savez(args.output / "tensors.npz", **captured, waveform=waveform)
    request = vars(args) | {"seconds": elapsed, "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                            "source_revision": "81aeb3ae772c70093c3de74dc23c92d983801ae4", "input": "upstream English fixture"}
    (args.output / "request.json").write_text(json.dumps(request, default=str, indent=2) + "\n")
    print(json.dumps(request, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("upstream", "models", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--control", choices=["melody", "score"], default="melody")
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--task", choices=["svs", "svc"], default="svs")
    parser.add_argument("--whisper-source", type=Path)
    run(parser.parse_args())
