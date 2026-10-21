"""Local, immutable image attachments for Generate Text histories."""
import hashlib
import os
import re
import tempfile

import numpy as np
from PIL import Image, ImageOps
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
import torch

import folder_paths
import node_helpers


def image_id(image):
    digest = hashlib.sha256()
    digest.update(f"{image.dtype}:{tuple(image.shape)}:".encode())
    digest.update(image.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def attachment_path(identifier):
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("Invalid text history image_id; expected a SHA-256 attachment ID.")
    root = os.path.join(folder_paths.get_system_user_directory(), "text_history")
    path = os.path.join(root, "images", identifier + ".safetensors")
    if not folder_paths.is_within_directory(root, path):
        raise ValueError("Text history image path escapes its attachment directory.")
    return path


def load_attachment(identifier):
    path = attachment_path(identifier)
    if not os.path.isfile(path):
        raise ValueError(f"Text history image {identifier} is missing. Restore its attachment file.")
    try:
        tensors = load_file(path, device="cpu")
    except SafetensorError as error:
        raise ValueError(f"Text history image {identifier} is corrupt. Restore its attachment file.") from error
    if set(tensors) != {"image"}:
        raise ValueError(f"Text history image {identifier} has invalid contents.")
    image = tensors["image"]
    if image.ndim != 4 or image.shape[0] != 1 or image_id(image) != identifier:
        raise ValueError(f"Text history image {identifier} is corrupt. Restore its attachment file.")
    return image


def input_image_path(file):
    filename, subfolder = file["filename"], file["subfolder"]
    if not filename or os.path.basename(filename) != filename or "\\" in filename or os.path.isabs(subfolder) or "\\" in subfolder:
        raise ValueError("Image history file requires a filename and relative input subfolder.")
    name = os.path.join(subfolder, filename)
    path = folder_paths.get_annotated_filepath(name, folder_paths.get_input_directory())
    if not folder_paths.is_within_directory(folder_paths.get_input_directory(), path):
        raise ValueError("Image history files must stay inside the ComfyUI input directory.")
    return path


def load_input_image(file):
    with node_helpers.pillow(Image.open, input_image_path(file)) as source:
        image = node_helpers.pillow(ImageOps.exif_transpose, source).convert("RGB")
        pixels = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def resolve_images(messages, image=None):
    resolved, stored, pending = [], [], {}
    for index, message in enumerate(messages):
        content = message["content"]
        if image is not None and index == len(messages) - 1:
            content = [{"type": "image", "data": item.unsqueeze(0)} for item in image] + [{"type": "text", "text": content}]
        if isinstance(content, list) and all(part["type"] == "text" for part in content):
            content = "".join(part["text"] for part in content)
        if isinstance(content, str):
            resolved.append({"role": message["role"], "content": content})
            stored.append({"role": message["role"], "content": content})
            continue
        inputs, references = [], []
        for part in content:
            if part["type"] == "text":
                inputs.append(dict(part))
                references.append(dict(part))
                continue
            if "image_id" in part:
                identifier = part["image_id"]
                pixels = load_attachment(identifier)
            else:
                pixels = part["data"] if "data" in part else load_input_image(part["file"])
                # Own the slice: neither the snapshot nor replay should retain a larger batch.
                pixels = pixels.detach().to(device="cpu").clone().contiguous()
                identifier = image_id(pixels)
                pending[identifier] = pixels
            inputs.append({"type": "image", "data": pixels})
            references.append({"type": "image", "image_id": identifier})
        resolved.append({"role": message["role"], "content": inputs})
        stored.append({"role": message["role"], "content": references})
    return resolved, stored, pending


def publish_images(pending):
    for identifier, pixels in pending.items():
        path = attachment_path(identifier)
        if os.path.exists(path):
            load_attachment(identifier)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = attachment_path(identifier)
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix=".tmp", delete=False) as output:
            temporary = output.name
        try:
            save_file({"image": pixels}, temporary)
            try:
                os.link(temporary, path)
            except FileExistsError:
                load_attachment(identifier)
        finally:
            os.unlink(temporary)


def fingerprint(messages):
    digest = hashlib.sha256()
    for message in messages:
        if isinstance(message["content"], str):
            continue
        for part in message["content"]:
            if part["type"] != "image":
                continue
            path = attachment_path(part["image_id"]) if "image_id" in part else input_image_path(part["file"])
            with open(path, "rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()
