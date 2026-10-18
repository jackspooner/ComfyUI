#!/usr/bin/env python3
"""Check an explicitly configured ComfyUI runtime for worktree development."""
import os
import subprocess
import sys
from pathlib import Path

runtime = os.environ.get("COMFYUI_PYTHON")
if not runtime or not Path(runtime).is_file():
    sys.exit("Set COMFYUI_PYTHON to the Python executable of a prepared ComfyUI environment (Python >=3.10, torch, einops, safetensors). This command never modifies that environment or starts ComfyUI.")
check = "import sys, torch, einops, safetensors; assert sys.version_info >= (3,10), 'Python >=3.10 required'; print('Ready:', sys.version.split()[0], 'torch', torch.__version__, 'einops', einops.__version__, 'safetensors', safetensors.__version__)"
subprocess.run([runtime, "-c", check], check=True)
if "--yingmusic" in sys.argv:
    subprocess.run([runtime, "-c", "import phonemizer, inflect, pypinyin, cn2an, jieba, onnxruntime; from phonemizer.backend import EspeakBackend; assert EspeakBackend.is_available(), 'Install the espeak-ng shared library'; print('YingMusic lyric prerequisites ready')"], check=True)
if "--paddleocr" in sys.argv:
    subprocess.run([runtime, "-c", "import cv2, numpy, torchvision; print('PaddleOCR prerequisites ready:', 'OpenCV', cv2.__version__, 'torchvision', torchvision.__version__)"], check=True)
if "--trellis-mesh" in sys.argv:
    subprocess.run([runtime, str(Path(__file__).with_name("build_trellis2_voxel.py"))], check=True)
print("Use the configured interpreter for tests. No model, production configuration, dependencies or tracked files were changed.")
