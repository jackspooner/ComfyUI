"""Image-only external references; generic annotated file containment stays unchanged."""
import os
import stat

from PIL import Image

import folder_paths
from comfy.cli_args import args


def resolve_image_path(reference: str) -> str:
    if not os.path.isabs(reference):
        return folder_paths.get_annotated_filepath(reference)
    if not args.allow_external_image_paths:
        raise PermissionError("External image paths require --allow-external-image-paths")
    filename = os.path.realpath(reference, strict=True)
    if not stat.S_ISREG(os.stat(filename).st_mode):
        raise ValueError("External image path must name a regular image file")
    with Image.open(filename) as image:
        image.verify()
    return filename
