"""CPU-only extraction benchmark. Run with PYTHONPATH pointing at the checkout under test."""
import argparse
import cProfile
import json
import resource
import sys
import time

import av
import numpy as np

from comfy_api.input_impl.video_types import VideoFromFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--create", action="store_true", help="Create a synthetic H.264 fixture instead of benchmarking")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--mode", choices=["default", "compact", "selected", "metadata"], default="default")
    parser.add_argument("--profile", help="Optional cProfile output path")
    args = parser.parse_args()
    if args.create:
        with av.open(args.path, "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = args.width, args.height, "yuv420p"
            stream.options = {"preset": "ultrafast", "crf": "18"}
            pixels = np.random.default_rng(0).integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
            for index in range(args.frames):
                image = np.roll(pixels, index, axis=1)
                container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
            container.mux(stream.encode(None))
        return
    video = VideoFromFile(args.path)
    profile = cProfile.Profile() if args.profile else None
    if profile:
        profile.enable()
    start = time.perf_counter()
    if args.mode == "default":
        components = video.get_components()
    else:
        options = {"compact": {"compact": True}, "selected": {"include_audio": False, "frame_stride": 2, "max_size": 640}, "metadata": {"include_images": False, "include_audio": False}}
        components = video.get_components_selected(**options[args.mode])
    elapsed = time.perf_counter() - start
    if profile:
        profile.disable()
        profile.dump_stats(args.profile)
    sys.stdout.write(json.dumps({"mode": args.mode, "seconds": elapsed, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      "shape": list(components.images.shape), "fps": float(components.frame_rate),
                                 "output_mib": components.images.numel() * components.images.element_size() / 1024 ** 2}) + "\n")


if __name__ == "__main__":
    main()
