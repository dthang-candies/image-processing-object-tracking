"""Foreground segmentation and connected-object detection with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .models import Detection


DeviceMode = Literal["auto", "cpu", "gpu"]


@dataclass(slots=True)
class DetectorConfig:
    history: int = 300
    variance_threshold: float = 24.0
    detect_shadows: bool = True
    min_area: float = 400.0
    max_area: float | None = None
    min_width: int = 8
    min_height: int = 8
    kernel_size: int = 5
    open_iterations: int = 1
    close_iterations: int = 2
    learning_rate: float = -1.0
    histogram_h_bins: int = 12
    histogram_s_bins: int = 8
    max_foreground_ratio: float = 0.65
    device: DeviceMode = "auto"


class ForegroundDetector:
    """Segment moving foreground with MOG2 and extract contour centroids."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        if self.config.kernel_size <= 0 or self.config.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.config.histogram_h_bins <= 0 or self.config.histogram_s_bins <= 0:
            raise ValueError("histogram bin counts must be positive")
        if not 0 < self.config.max_foreground_ratio <= 1:
            raise ValueError("max_foreground_ratio must be in (0, 1]")
        if self.config.device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
        self.device, self.device_name = self._select_device(self.config.device)
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.variance_threshold,
            detectShadows=self.config.detect_shadows,
        )
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.kernel_size, self.config.kernel_size)
        )

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        processing_frame = cv2.UMat(frame) if self.device == "gpu" else frame
        mask = self.subtractor.apply(
            processing_frame, learningRate=self.config.learning_rate
        )
        # MOG2 encodes shadows as 127; keep only certain foreground (255).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        if self.config.open_iterations:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self.kernel,
                iterations=self.config.open_iterations,
            )
        if self.config.close_iterations:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self.kernel,
                iterations=self.config.close_iterations,
            )

        foreground_pixels = float(cv2.countNonZero(mask))
        mask = mask.get() if isinstance(mask, cv2.UMat) else mask
        foreground_ratio = foreground_pixels / mask.size
        if foreground_ratio > self.config.max_foreground_ratio:
            return mask, []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_area:
                continue
            if self.config.max_area is not None and area > self.config.max_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < self.config.min_width or height < self.config.min_height:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) > 1e-9:
                centroid = (
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                )
            else:
                centroid = (x + width / 2.0, y + height / 2.0)
            appearance = self._appearance_histogram(hsv, contour, (x, y, width, height))
            detections.append(
                Detection(
                    (x, y, width, height),
                    centroid,
                    area,
                    contour,
                    appearance,
                )
            )

        detections.sort(key=lambda detection: (detection.centroid[0], detection.centroid[1]))
        return mask, detections

    @staticmethod
    def _select_device(requested: DeviceMode) -> tuple[str, str]:
        """Select CPU or OpenCL GPU without silently faking acceleration."""
        if requested == "cpu":
            cv2.ocl.setUseOpenCL(False)
            return "cpu", "CPU"

        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            device = cv2.ocl.Device_getDefault()
            is_gpu = bool(device.type() & cv2.ocl.Device_TYPE_GPU)
            if cv2.ocl.useOpenCL() and device.available() and is_gpu:
                return "gpu", f"OpenCL: {device.name()}"

        cv2.ocl.setUseOpenCL(False)
        if requested == "gpu":
            raise RuntimeError(
                "GPU requested, but OpenCV cannot find an OpenCL GPU. "
                "Use --device cpu or install an OpenCV build with GPU support."
            )
        return "cpu", "CPU (GPU unavailable)"

    def _appearance_histogram(
        self,
        hsv_frame: np.ndarray,
        contour: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Build an L1-normalized hue/saturation histogram inside a contour."""
        x, y, width, height = bbox
        roi = hsv_frame[y : y + height, x : x + width]
        local_contour = contour - np.array([[[x, y]]], dtype=contour.dtype)
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [local_contour], -1, 255, -1)
        histogram = cv2.calcHist(
            [roi],
            [0, 1],
            contour_mask,
            [self.config.histogram_h_bins, self.config.histogram_s_bins],
            [0, 180, 0, 256],
        ).reshape(-1)
        total = float(histogram.sum())
        if total > 0:
            histogram /= total
        return histogram.astype(np.float32, copy=False)
