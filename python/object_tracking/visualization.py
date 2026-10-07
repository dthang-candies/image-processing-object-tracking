"""Video annotation functions."""

from __future__ import annotations

import cv2
import numpy as np

from .models import Detection, TrackSnapshot


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 7919)
    values = rng.integers(70, 256, size=3)
    return int(values[0]), int(values[1]), int(values[2])


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    tracks: list[TrackSnapshot],
    algorithm: str,
) -> np.ndarray:
    canvas = frame.copy()
    for detection in detections:
        if detection.contour is not None:
            cv2.drawContours(canvas, [detection.contour], -1, (80, 180, 80), 1)
        cx, cy = (int(round(value)) for value in detection.centroid)
        cv2.circle(canvas, (cx, cy), 3, (0, 255, 255), -1)

    for track in tracks:
        if not track.confirmed:
            continue
        color = _color(track.track_id)
        x, y, width, height = track.bbox
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
        label = f"ID {track.track_id}" + (f" P{track.missed}" if track.missed else "")
        cv2.putText(
            canvas,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        if len(track.trail) >= 2:
            points = np.asarray(track.trail, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{algorithm} | objects: {sum(track.confirmed for track in tracks)}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas

