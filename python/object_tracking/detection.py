"""Foreground segmentation and connected-object detection with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .models import Detection


DeviceMode = Literal["auto", "cpu", "gpu"]
SegmentationMode = Literal["mog2", "bright_spots"]


@dataclass(slots=True)
class DetectorConfig:
    segmentation_mode: SegmentationMode = "mog2"
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
    spot_threshold: int = 15
    spot_background_size: int = 9
    adaptive_min_area: bool = False
    adaptive_spot_threshold: bool = False


class ForegroundDetector:
    """Segment moving foreground or bright spots and extract centroids."""

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
        if self.config.segmentation_mode not in {"mog2", "bright_spots"}:
            raise ValueError("segmentation_mode must be 'mog2' or 'bright_spots'")
        if not 0 <= self.config.spot_threshold <= 255:
            raise ValueError("spot_threshold must be between 0 and 255")
        if self.config.spot_background_size <= 1 or self.config.spot_background_size % 2 == 0:
            raise ValueError("spot_background_size must be an odd integer greater than 1")
        self.device, self.device_name = self._select_device(self.config.device)
        self.subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=self.config.history,
                varThreshold=self.config.variance_threshold,
                detectShadows=self.config.detect_shadows,
            )
            if self.config.segmentation_mode == "mog2"
            else None
        )
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.kernel_size, self.config.kernel_size)
        )
        self.fragment_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.spot_background_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.spot_background_size, self.config.spot_background_size),
        )
        self._minimum_area = self.config.min_area
        self._spot_threshold = float(self.config.spot_threshold)

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        processing_frame = cv2.UMat(frame) if self.device == "gpu" else frame
        # A small edge-preserving filter suppresses low-amplitude noise without
        # the strong loss of tiny, faint spots caused by a Gaussian blur.
        processing_frame = cv2.bilateralFilter(processing_frame, 3, 3.0, 1.0)
        if self.config.segmentation_mode == "bright_spots":
            gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
            response = cv2.morphologyEx(
                gray, cv2.MORPH_TOPHAT, self.spot_background_kernel
            )
            self._spot_threshold = float(self.config.spot_threshold)
            if self.config.adaptive_spot_threshold:
                gray_array = gray.get() if isinstance(gray, cv2.UMat) else gray
                background = cv2.GaussianBlur(gray_array, (5, 5), 1.0)
                residual = (
                    gray_array.astype(np.int16) - background.astype(np.int16)
                )[::2, ::2]
                median = float(np.median(residual))
                mad = float(np.median(np.abs(residual - median)))
                noise_sigma = mad / 0.6745
                self._spot_threshold = min(
                    255.0, max(self._spot_threshold, 4.5 * noise_sigma)
                )
            _, mask = cv2.threshold(
                response,
                self._spot_threshold,
                255,
                cv2.THRESH_BINARY,
            )
        else:
            assert self.subtractor is not None
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

        candidates = self._find_candidates(mask, self.config.min_area)
        median_area = (
            float(np.median([item[1] for item in candidates])) if candidates else 0.0
        )

        if self.config.adaptive_min_area:
            target_area = self.config.min_area
            if len(candidates) >= 10:
                areas = np.asarray([item[1] for item in candidates], dtype=float)
                area_percentile = float(np.percentile(areas, 90))
                area_cap = max(self.config.min_area, min(64.0, mask.size * 0.0001))
                target_area = max(
                    self.config.min_area, min(area_cap, 0.08 * area_percentile)
                )
                if self.config.segmentation_mode == "mog2":
                    large_area = max(100.0, mask.size * 0.0004)
                    large_components = areas[areas >= large_area]
                    required_count = max(4, int(np.ceil(len(areas) * 0.02)))
                    if len(large_components) >= required_count:
                        large_floor = 0.08 * float(np.percentile(large_components, 25))
                        target_area = max(target_area, min(area_cap, large_floor))
            # Raise the threshold gradually, but return promptly to the small
            # object threshold when the observed scale becomes smaller.
            rate = 0.5 if target_area < self._minimum_area else 0.2
            self._minimum_area += rate * (target_area - self._minimum_area)
            if abs(self._minimum_area - target_area) < 0.5:
                self._minimum_area = target_area
        else:
            self._minimum_area = self.config.min_area

        candidates = [item for item in candidates if item[1] >= self._minimum_area]
        clean_mask = np.zeros_like(mask)
        if candidates:
            cv2.drawContours(clean_mask, [item[0] for item in candidates], -1, 255, -1)
        mask = cv2.bitwise_and(mask, clean_mask)

        # In large-object scenes, reconnect close foreground pieces and remove
        # small boxes almost entirely contained in a larger box. The scale
        # gate leaves dense scenes dominated by tiny spots untouched.
        if (
            self.config.adaptive_min_area
            and self._minimum_area >= 20.0
            and median_area >= 8.0
            and len(candidates) >= 2
        ):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.fragment_kernel)
            candidates = _remove_nested_fragments(
                self._find_candidates(mask, self._minimum_area)
            )
            clean_mask = np.zeros_like(mask)
            if candidates:
                cv2.drawContours(clean_mask, [item[0] for item in candidates], -1, 255, -1)
            mask = cv2.bitwise_and(mask, clean_mask)
        hsv = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if self.config.segmentation_mode == "mog2"
            else None
        )
        detections: list[Detection] = []
        for contour, area, (x, y, width, height) in candidates:
            moments = cv2.moments(contour)
            if abs(moments["m00"]) > 1e-9:
                centroid = (
                    float(moments["m10"] / moments["m00"]),
                    float(moments["m01"] / moments["m00"]),
                )
            else:
                centroid = (x + width / 2.0, y + height / 2.0)
            appearance = (
                self._appearance_histogram(hsv, contour, (x, y, width, height))
                if hsv is not None
                else None
            )
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

    def _find_candidates(
        self, mask: np.ndarray, min_area: float
    ) -> list[tuple[np.ndarray, float, tuple[int, int, int, int]]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            if self.config.max_area is not None and area > self.config.max_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < self.config.min_width or height < self.config.min_height:
                continue
            candidates.append((contour, area, (x, y, width, height)))
        return candidates

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
                "Set device to 'cpu' or install an OpenCV build with GPU support."
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


def _remove_nested_fragments(
    candidates: list[tuple[np.ndarray, float, tuple[int, int, int, int]]],
) -> list[tuple[np.ndarray, float, tuple[int, int, int, int]]]:
    """Discard small boxes mostly contained inside a much larger detection."""
    if len(candidates) < 2:
        return candidates
    boxes = np.asarray([candidate[2] for candidate in candidates], dtype=float)
    x, y, width, height = boxes.T
    areas = width * height
    intersection_width = np.maximum(
        0.0, np.minimum((x + width)[:, None], (x + width)[None, :])
        - np.maximum(x[:, None], x[None, :])
    )
    intersection_height = np.maximum(
        0.0, np.minimum((y + height)[:, None], (y + height)[None, :])
        - np.maximum(y[:, None], y[None, :])
    )
    contained_fraction = intersection_width * intersection_height / areas[:, None]
    nested = (
        (contained_fraction >= 0.9)
        & (areas[None, :] >= 4.0 * areas[:, None])
        & (areas[None, :] >= 400.0)
    )
    return [
        candidate
        for index, candidate in enumerate(candidates)
        if not np.any(nested[index])
    ]
