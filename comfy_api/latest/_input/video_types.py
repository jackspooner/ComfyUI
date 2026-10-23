from __future__ import annotations
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Optional, Union, IO
import io
import av
import torch
import torch.nn.functional as F
from .._util import VideoContainer, VideoCodec, VideoComponents, normalize_crop_rect

class VideoInput(ABC):
    """
    Abstract base class for video input types.
    """

    @abstractmethod
    def get_components(self) -> VideoComponents:
        """
        Abstract method to get the video components (images, audio, and frame rate).

        Returns:
            VideoComponents containing images, audio, and frame rate
        """
        pass

    def get_components_selected(
        self, *, include_images: bool = True, include_audio: bool = True,
        frame_stride: int = 1, max_size: int = 0, compact: bool = False,
    ) -> VideoComponents:
        """Extract selected components; file inputs avoid decoding unused streams.

        Keep every frame_stride-th frame and divide FPS accordingly. max_size
        limits the longest image edge without enlarging. compact permits 8-bit
        RGB conversion of 8-bit SDR files. Defaults match get_components().
        Other implementations retain their existing get_components() signature
        and use this materializing fallback unless they override it.
        """
        if frame_stride < 1 or max_size < 0:
            raise ValueError("frame_stride must be >= 1 and max_size must be >= 0")
        components = self.get_components()
        images = components.images if include_images else torch.zeros(0, 0, 0, 3)
        alpha = components.alpha if include_images else None
        if include_images:
            if frame_stride > 1:
                images = images[::frame_stride].clone()
                alpha = alpha[::frame_stride].clone() if alpha is not None else None
            if max_size and images.shape[0]:
                height, width = images.shape[1:3]
                if max(height, width) > max_size:
                    size = (max(1, round(height * max_size / max(height, width))),
                            max(1, round(width * max_size / max(height, width))))
                    images = F.interpolate(images.movedim(-1, 1), size=size, mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
                    if alpha is not None:
                        alpha = F.interpolate(alpha.movedim(-1, 1), size=size, mode="bilinear", align_corners=False, antialias=True).movedim(1, -1)
        return VideoComponents(
            images=images, audio=components.audio if include_audio else None,
            frame_rate=components.frame_rate / frame_stride if include_images else components.frame_rate,
            alpha=alpha, metadata=components.metadata,
        )

    @abstractmethod
    def save_to(
        self,
        path: Union[str, IO[bytes]],
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
        crf: float | None = None,
        color_space: str | None = None,
        preset: str | None = None,
    ):
        """
        Abstract method to save the video input to a file.

        bit_depth selects the encoded bit depth; None keeps the video's native depth.
        crf selects the H.264 or AV1 constant rate factor; None uses 18 for H.264 or 24 for AV1 when encoding.
        preset selects the H.264 encoder speed/compression trade-off (e.g. "ultrafast");
        None uses the encoder default. Ignored for other codecs.
        color_space="sRGB" selects SDR BT.709/sRGB, "HDR" selects BT.2020/HLG, and "HDR PQ"
        selects BT.2020/PQ. Bit depth is selected independently.
        Tensor-created videos default to sRGB when color_space is None. Loaded videos keep matching recognized native color
        properties; other input pixels must already use the selected color space.
        """
        pass

    def get_color_space(self) -> str:
        """Return the video's color space as sRGB, HDR, HDR PQ, or auto when unspecified."""
        return "auto"

    @abstractmethod
    def as_trimmed(
        self,
        start_time: float | None = None,
        duration: float | None = None,
        strict_duration: bool = False,
    ) -> VideoInput | None:
        """
        Create a new VideoInput which is trimmed to have the corresponding start_time and duration

        Returns:
            A new VideoInput, or None if the result would have negative duration
        """
        pass

    def as_cropped(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> VideoInput:
        """
        Create a new VideoInput spatially cropped to the given pixel rectangle.

        The rectangle is clamped to the frame and even-aligned for encoder
        compatibility. An empty or full-frame rectangle returns the input
        unchanged.

        Default implementation materializes the video via get_components();
        subclasses should override with lazier strategies when possible.
        """
        components = self.get_components()
        rect = normalize_crop_rect(
            x, y, width, height, components.images.shape[2], components.images.shape[1]
        )
        if rect is None:
            return self
        from .._input_impl.video_types import VideoFromComponents

        cx, cy, cw, ch = rect
        return VideoFromComponents(
            VideoComponents(
                images=components.images[:, cy:cy + ch, cx:cx + cw, :].clone(),
                audio=components.audio,
                frame_rate=components.frame_rate,
                metadata=components.metadata,
                alpha=components.alpha[:, cy:cy + ch, cx:cx + cw].clone()
                if components.alpha is not None
                else None,
            ),
            bit_depth=self.get_bit_depth(),
        )

    def get_stream_source(self) -> Union[str, io.BytesIO]:
        """
        Get a streamable source for the video. This allows processing without
        loading the entire video into memory.

        Returns:
            Either a file path (str) or a BytesIO object that can be opened with av.

        Default implementation creates a BytesIO buffer, but subclasses should
        override this for better performance when possible.
        """
        buffer = io.BytesIO()
        self.save_to(buffer)
        buffer.seek(0)
        return buffer

    def get_active_trim_window(self) -> tuple[float, float]:
        """Return the active trim as ``(start_time, duration)`` in seconds (start_time normalized
        to ``>= 0``; ``duration == 0`` means "until the end"). Default: no trim; trimmable subclasses override.
        """
        return 0.0, 0.0

    # Provide a default implementation, but subclasses can provide optimized versions
    # if possible.
    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        components = self.get_components()
        return components.images.shape[2], components.images.shape[1]

    def get_bit_depth(self) -> int:
        """
        Returns the bit depth of the video (e.g. 8 or 10).

        Default implementation returns 8; subclasses report their real depth.
        """
        return 8

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        components = self.get_components()
        frame_count = components.images.shape[0]
        return float(frame_count / components.frame_rate)

    def get_frame_count(self) -> int:
        """
        Returns the number of frames in the video.

        Default implementation uses :meth:`get_components`, which may require
        loading all frames into memory. File-based implementations should
        override this method and use container/stream metadata instead.

        Returns:
            Total number of frames as an integer.
        """
        return int(self.get_components().images.shape[0])

    def get_frame_rate(self) -> Fraction:
        """
        Returns the frame rate of the video.

        Default implementation materializes the video into memory via
        `get_components()`. Subclasses that can inspect the underlying
        container (e.g. `VideoFromFile`) should override this with a more
        efficient implementation.

        Returns:
            Frame rate as a Fraction.
        """
        return self.get_components().frame_rate

    def get_container_format(self) -> str:
        """
        Returns the container format of the video (e.g., 'mp4', 'mov', 'avi').

        Returns:
            Container format as string
        """
        # Default implementation - subclasses should override for better performance
        source = self.get_stream_source()
        with av.open(source, mode="r") as container:
            return container.format.name
