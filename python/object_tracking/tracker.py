"""Multi-object tracking with selectable data association and Kalman filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .assignment import (
    hungarian_assignment,
    nearest_neighbor_assignment,
    pairwise_distances,
)
from .kalman import KalmanFilter2D
from .models import BBox, Detection, TrackSnapshot, TrajectoryPoint


AssociationMethod = Literal["nearest", "hungarian"]


@dataclass(slots=True)
class _Track:
    track_id: int
    filter: KalmanFilter2D
    bbox: BBox
    area: float
    age: int = 1
    hits: int = 1
    missed: int = 0
    history: list[TrajectoryPoint] = field(default_factory=list)
    appearance: np.ndarray | None = None


class MultiObjectTracker:
    """Centroid tracker that keeps one constant-velocity Kalman filter per ID."""

    def __init__(
        self,
        association: AssociationMethod = "hungarian",
        max_distance: float = 60.0,
        max_missed: int = 15,
        min_hits: int = 2,
        process_variance: float = 5.0,
        measurement_variance: float = 25.0,
        initial_velocity_variance: float = 10_000.0,
        iou_weight: float = 0.25,
        area_weight: float = 0.10,
        max_area_ratio: float = 4.0,
        appearance_weight: float = 0.50,
        appearance_momentum: float = 0.90,
    ) -> None:
        if association not in {"nearest", "hungarian"}:
            raise ValueError("association must be 'nearest' or 'hungarian'")
        if max_distance <= 0 or max_missed < 0 or min_hits <= 0:
            raise ValueError("invalid tracker thresholds")
        if (
            iou_weight < 0
            or area_weight < 0
            or max_area_ratio < 1
            or appearance_weight < 0
            or not 0 <= appearance_momentum < 1
        ):
            raise ValueError("invalid association weights")
        self.association = association
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        self.initial_velocity_variance = float(initial_velocity_variance)
        self.iou_weight = float(iou_weight)
        self.area_weight = float(area_weight)
        self.max_area_ratio = float(max_area_ratio)
        self.appearance_weight = float(appearance_weight)
        self.appearance_momentum = float(appearance_momentum)
        self._active: list[_Track] = []
        self._finished: list[_Track] = []
        self._next_id = 1

    def _new_track(
        self, detection: Detection, frame_index: int, time_seconds: float
    ) -> _Track:
        track = _Track(
            track_id=self._next_id,
            filter=KalmanFilter2D(
                detection.centroid,
                process_variance=self.process_variance,
                measurement_variance=self.measurement_variance,
                initial_velocity_variance=self.initial_velocity_variance,
            ),
            bbox=detection.bbox,
            area=detection.area,
            appearance=(
                np.asarray(detection.appearance, dtype=float).copy()
                if detection.appearance is not None
                else None
            ),
        )
        self._next_id += 1
        track.history.append(
            TrajectoryPoint(
                frame_index,
                time_seconds,
                detection.centroid,
                detection.bbox,
                True,
                track.filter.velocity,
            )
        )
        return track

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        time_seconds: float = 0.0,
        dt: float = 1.0,
    ) -> list[TrackSnapshot]:
        predictions = np.asarray(
            [track.filter.predict(dt) for track in self._active], dtype=float
        ).reshape((-1, 2))
        measurements = np.asarray(
            [detection.centroid for detection in detections], dtype=float
        ).reshape((-1, 2))
        pair_costs, valid_mask = self._association_costs(
            predictions, detections
        )
        assign = (
            nearest_neighbor_assignment
            if self.association == "nearest"
            else hungarian_assignment
        )
        matches, unmatched_tracks, unmatched_detections = assign(
            predictions,
            measurements,
            self.max_distance,
            pair_costs=pair_costs,
            valid_mask=valid_mask,
        )

        for track_index, detection_index in matches:
            track = self._active[track_index]
            detection = detections[detection_index]
            position = track.filter.update(detection.centroid)
            track.bbox = detection.bbox
            track.area = detection.area
            track.age += 1
            track.hits += 1
            track.missed = 0
            self._update_appearance(track, detection)
            track.history.append(
                TrajectoryPoint(
                    frame_index,
                    time_seconds,
                    position,
                    detection.bbox,
                    True,
                    track.filter.velocity,
                )
            )

        for track_index in unmatched_tracks:
            track = self._active[track_index]
            predicted = track.filter.position
            track.age += 1
            track.missed += 1
            x, y, width, height = track.bbox
            predicted_bbox = (
                int(round(predicted[0] - width / 2)),
                int(round(predicted[1] - height / 2)),
                width,
                height,
            )
            track.bbox = predicted_bbox
            track.history.append(
                TrajectoryPoint(
                    frame_index,
                    time_seconds,
                    predicted,
                    predicted_bbox,
                    False,
                    track.filter.velocity,
                )
            )

        for detection_index in unmatched_detections:
            self._active.append(
                self._new_track(detections[detection_index], frame_index, time_seconds)
            )

        survivors: list[_Track] = []
        for track in self._active:
            confirmed = track.hits >= self.min_hits
            # A tentative track that immediately disappears is usually a
            # foreground-segmentation artifact, so do not retain it.
            should_delete = (
                track.missed > self.max_missed
                if confirmed
                else track.missed > 0
            )
            if should_delete and confirmed:
                self._finished.append(track)
            elif should_delete:
                continue
            else:
                survivors.append(track)
        self._active = survivors
        return self.snapshots()

    def snapshots(self, trail_length: int = 30) -> list[TrackSnapshot]:
        return [
            TrackSnapshot(
                track_id=track.track_id,
                centroid=track.filter.position,
                bbox=track.bbox,
                age=track.age,
                hits=track.hits,
                missed=track.missed,
                confirmed=track.hits >= self.min_hits,
                trail=tuple(point.centroid for point in track.history[-trail_length:]),
                velocity=track.filter.velocity,
            )
            for track in self._active
        ]

    @property
    def trajectories(self) -> dict[int, list[TrajectoryPoint]]:
        """Return only confirmed active and completed trajectory histories."""
        tracks = sorted(self._finished + self._active, key=lambda track: track.track_id)
        return {
            track.track_id: list(track.history)
            for track in tracks
            if track.hits >= self.min_hits
        }

    def _association_costs(
        self, predictions: np.ndarray, detections: list[Detection]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine centroid motion, overlap, and relative object area."""
        measurements = np.asarray(
            [detection.centroid for detection in detections], dtype=float
        ).reshape((-1, 2))
        distances = pairwise_distances(predictions, measurements)
        if not self._active or not detections:
            return distances, np.ones(distances.shape, dtype=bool)

        iou_cost = np.empty_like(distances)
        area_cost = np.empty_like(distances)
        appearance_cost = np.zeros_like(distances)
        valid_area = np.empty(distances.shape, dtype=bool)
        for track_index, track in enumerate(self._active):
            predicted_bbox = _bbox_at_centroid(track.bbox, predictions[track_index])
            for detection_index, detection in enumerate(detections):
                iou_cost[track_index, detection_index] = 1.0 - _bbox_iou(
                    predicted_bbox, detection.bbox
                )
                small_area = max(min(track.area, detection.area), 1e-9)
                large_area = max(track.area, detection.area)
                area_ratio = large_area / small_area
                valid_area[track_index, detection_index] = (
                    area_ratio <= self.max_area_ratio
                )
                area_cost[track_index, detection_index] = np.log(area_ratio)
                appearance_cost[track_index, detection_index] = _appearance_distance(
                    track.appearance, detection.appearance
                )

        costs = (
            distances
            + self.iou_weight * self.max_distance * iou_cost
            + self.area_weight * self.max_distance * area_cost
            + self.appearance_weight * self.max_distance * appearance_cost
        )
        return costs, valid_area

    def _update_appearance(self, track: _Track, detection: Detection) -> None:
        if detection.appearance is None:
            return
        observed = np.asarray(detection.appearance, dtype=float)
        if track.appearance is None or track.appearance.shape != observed.shape:
            track.appearance = observed.copy()
            return
        track.appearance = (
            self.appearance_momentum * track.appearance
            + (1.0 - self.appearance_momentum) * observed
        )
        total = float(track.appearance.sum())
        if total > 0:
            track.appearance /= total


def _bbox_at_centroid(bbox: BBox, centroid: np.ndarray) -> BBox:
    _, _, width, height = bbox
    return (
        int(round(float(centroid[0]) - width / 2)),
        int(round(float(centroid[1]) - height / 2)),
        width,
        height,
    )


def _bbox_iou(first: BBox, second: BBox) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _appearance_distance(first: object | None, second: object | None) -> float:
    """Hellinger distance between normalized histograms, in [0, 1]."""
    if first is None or second is None:
        return 0.0
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        return 0.0
    first_sum = float(first_array.sum())
    second_sum = float(second_array.sum())
    if first_sum <= 0 or second_sum <= 0:
        return 0.0
    coefficient = np.sqrt(
        (first_array / first_sum) * (second_array / second_sum)
    ).sum()
    return float(np.sqrt(max(0.0, 1.0 - min(float(coefficient), 1.0))))

