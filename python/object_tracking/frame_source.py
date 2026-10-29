"""Uniform frame readers for conventional videos and TIFF image stacks."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TIFF_EXTENSIONS = {".tif", ".tiff"}


class VideoFrameSource:
    source_type = "video"

    def __init__(self, path: Path) -> None:
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or not np.isfinite(self.fps):
            self.fps = 30.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class TiffFrameSource:
    source_type = "tiff_stack"

    def __init__(self, path: Path, fps_override: float | None = None) -> None:
        self.image = Image.open(path)
        self.frame_count = int(getattr(self.image, "n_frames", 1))
        self.width, self.height = self.image.size
        self.fps = fps_override or _imagej_fps(self.image) or 10.0
        if self.fps <= 0 or not np.isfinite(self.fps):
            self.image.close()
            raise ValueError("TIFF FPS must be positive")
        self._index = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= self.frame_count:
            return False, None
        self.image.seek(self._index)
        self._index += 1
        array = np.asarray(self.image)
        if array.ndim == 2:
            gray = _to_uint8(array)
            return True, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rgb = np.asarray(self.image.convert("RGB"))
        return True, cv2.cvtColor(_to_uint8(rgb), cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        self.image.close()


def open_frame_source(
    path: Path, tiff_fps: float | None = None
) -> VideoFrameSource | TiffFrameSource:
    if path.suffix.lower() in TIFF_EXTENSIONS:
        return TiffFrameSource(path, fps_override=tiff_fps)
    return VideoFrameSource(path)


def _imagej_fps(image: Image.Image) -> float | None:
    description = image.tag_v2.get(270, "") if hasattr(image, "tag_v2") else ""
    if isinstance(description, bytes):
        description = description.decode("utf-8", errors="ignore")
    match = re.search(r"(?im)^finterval\s*=\s*([0-9.eE+-]+)", str(description))
    if not match:
        return None
    interval = float(match.group(1))
    return 1.0 / interval if interval > 0 else None


def _to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)
    minimum = float(np.min(array[finite]))
    maximum = float(np.max(array[finite]))
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array.astype(np.float32) - minimum) * (255.0 / (maximum - minimum))
    scaled[~finite] = 0
    return np.clip(scaled, 0, 255).astype(np.uint8)
