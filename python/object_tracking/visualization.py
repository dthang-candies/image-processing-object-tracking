"""Video annotation functions."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .models import Detection, TrackSnapshot


@lru_cache(maxsize=4096)
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
    visible_tracks = [track for track in tracks if track.confirmed and track.missed == 0]
    visible_boxes = {track.bbox for track in visible_tracks}
    for detection in detections:
        if detection.bbox not in visible_boxes:
            continue
        if detection.contour is not None:
            cv2.drawContours(canvas, [detection.contour], -1, (80, 180, 80), 1)
        cx, cy = (int(round(value)) for value in detection.centroid)
        radius = 1 if max(detection.bbox[2:]) <= 10 else 2
        cv2.circle(canvas, (cx, cy), radius, (0, 255, 255), -1)

    frame_height, frame_width = canvas.shape[:2]
    occupied_labels = np.zeros((frame_height, frame_width), dtype=np.uint8)
    occupied_labels[: min(frame_height, 30), : min(frame_width, 350)] = 1
    label_budget = max(8, frame_height * frame_width // 4000)
    labels_drawn = 0
    for track in visible_tracks:
        color = _color(track.track_id)
        x, y, box_width, box_height = track.bbox
        cv2.rectangle(canvas, (x, y), (x + box_width, y + box_height), color, 1)
        label = f"ID {track.track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
        )
        if (
            labels_drawn < label_budget
            and text_width + 2 <= frame_width
            and text_height + baseline + 2 <= frame_height
        ):
            text_x = max(0, min(x, frame_width - text_width - 1))
            text_y = min(frame_height - baseline - 1, max(text_height + 1, y - 4))
            y0 = max(0, text_y - text_height - 1)
            y1 = min(frame_height, text_y + baseline + 1)
            x1 = min(frame_width, text_x + text_width + 1)
            label_area = occupied_labels[y0:y1, text_x:x1]
            if label_area.size and not label_area.any():
                cv2.putText(
                    canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, color, 1, cv2.LINE_AA,
                )
                label_area[:] = 1
                labels_drawn += 1
        if len(track.trail) >= 2:
            points = np.asarray(track.trail, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{algorithm} | visible tracks: {len(visible_tracks)}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas

