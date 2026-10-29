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
    required_hits: int
    motion_anchor: tuple[float, float]
    previous_observation: tuple[float, float]
    has_moved: bool
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
        mahalanobis_gate: float = 25.0,
        mahalanobis_weight: float = 0.25,
        adaptive_confirmation: bool = False,
        min_motion_distance: float = 0.0,
        adaptive_process_noise: bool = False,
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
            or mahalanobis_gate <= 0
            or mahalanobis_weight < 0
            or min_motion_distance < 0
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
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.mahalanobis_weight = float(mahalanobis_weight)
        self.adaptive_confirmation = adaptive_confirmation
        self.min_motion_distance = float(min_motion_distance)
        self.adaptive_process_noise = adaptive_process_noise
        self._active: list[_Track] = []
        self._finished: list[_Track] = []
        self._next_id = 1

    def _new_track(
        self,
        detection: Detection,
        frame_index: int,
        time_seconds: float,
        required_hits: int,
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
            required_hits=required_hits,
            motion_anchor=detection.centroid,
            previous_observation=detection.centroid,
            has_moved=self.min_motion_distance == 0.0,
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
        if self.adaptive_process_noise:
            for track in self._active:
                vx, vy = track.filter.velocity
                track.filter.process_variance = self.process_variance + min(
                    100_000.0, 0.5 * (vx * vx + vy * vy)
                )
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
        association_gate = self.max_distance * min(np.sqrt(self.max_missed + 1.0), 1.5)
        matches, unmatched_tracks, unmatched_detections = assign(
            predictions,
            measurements,
            association_gate,
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
            self._update_motion(track, position)
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

        small_detection_limit = 0.0
        if self.adaptive_confirmation and len(detections) >= 10:
            areas = np.asarray([detection.area for detection in detections], dtype=float)
            large_scale = float(np.percentile(areas, 90))
            if large_scale >= 100.0 and float(np.median(areas)) >= 8.0:
                small_detection_limit = max(64.0, large_scale * 0.25)
        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            required_hits = self.min_hits + (
                2 if detection.area < small_detection_limit else 0
            )
            self._active.append(
                self._new_track(detection, frame_index, time_seconds, required_hits)
            )

        survivors: list[_Track] = []
        for track in self._active:
            confirmed = track.hits >= track.required_hits
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

    def snapshots(self, trail_length: int = 15) -> list[TrackSnapshot]:
        return [
            TrackSnapshot(
                track_id=track.track_id,
                centroid=track.filter.position,
                bbox=track.bbox,
                age=track.age,
                hits=track.hits,
                missed=track.missed,
                confirmed=track.hits >= track.required_hits and track.has_moved,
                trail=_observed_trail(track.history, trail_length),
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
            if track.hits >= track.required_hits and track.has_moved
        }

    def _update_motion(self, track: _Track, position: tuple[float, float]) -> None:
        """Confirm displacement using averaged observations, then retain it."""
        previous = track.previous_observation
        if track.hits == 2:
            track.motion_anchor = (
                (track.motion_anchor[0] + position[0]) / 2.0,
                (track.motion_anchor[1] + position[1]) / 2.0,
            )
        elif track.hits >= 4 and not track.has_moved:
            current = ((previous[0] + position[0]) / 2.0, (previous[1] + position[1]) / 2.0)
            displacement = np.hypot(
                current[0] - track.motion_anchor[0], current[1] - track.motion_anchor[1]
            )
            if displacement >= self.min_motion_distance:
                track.has_moved = True
        track.previous_observation = position

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

        track_boxes = np.asarray(
            [
                _bbox_at_centroid(track.bbox, predictions[index])
                for index, track in enumerate(self._active)
            ],
            dtype=float,
        )
        detection_boxes = np.asarray(
            [detection.bbox for detection in detections], dtype=float
        )
        intersection_x1 = np.maximum(
            track_boxes[:, None, 0], detection_boxes[None, :, 0]
        )
        intersection_y1 = np.maximum(
            track_boxes[:, None, 1], detection_boxes[None, :, 1]
        )
        intersection_x2 = np.minimum(
            track_boxes[:, None, 0] + track_boxes[:, None, 2],
            detection_boxes[None, :, 0] + detection_boxes[None, :, 2],
        )
        intersection_y2 = np.minimum(
            track_boxes[:, None, 1] + track_boxes[:, None, 3],
            detection_boxes[None, :, 1] + detection_boxes[None, :, 3],
        )
        intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
            0.0, intersection_y2 - intersection_y1
        )
        track_box_areas = track_boxes[:, 2] * track_boxes[:, 3]
        detection_box_areas = detection_boxes[:, 2] * detection_boxes[:, 3]
        union = track_box_areas[:, None] + detection_box_areas[None, :] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        iou_cost = 1.0 - iou

        track_areas = np.asarray([track.area for track in self._active])
        detection_areas = np.asarray([detection.area for detection in detections])
        small_areas = np.maximum(
            np.minimum(track_areas[:, None], detection_areas[None, :]), 1e-9
        )
        area_ratio = np.maximum(
            track_areas[:, None], detection_areas[None, :]
        ) / small_areas
        area_cost = np.log(area_ratio)
        valid_area = area_ratio <= self.max_area_ratio

        mahalanobis_squared = np.vstack(
            [track.filter.mahalanobis_squared_many(measurements) for track in self._active]
        )
        valid_motion = mahalanobis_squared <= self.mahalanobis_gate
        motion_cost = np.minimum(
            np.sqrt(mahalanobis_squared / self.mahalanobis_gate), 1.0
        )
        appearance_cost = (
            _appearance_cost_matrix(self._active, detections)
            if self.appearance_weight
            else np.zeros_like(distances)
        )

        # A limited expansion helps reacquire missed detections without letting
        # long gaps connect tracks to distant objects in crowded scenes.
        missed_counts = np.asarray([track.missed for track in self._active])
        distance_gates = self.max_distance * np.minimum(
            np.sqrt(missed_counts[:, None] + 1.0), 1.5
        )
        valid_distance = distances <= distance_gates

        costs = (
            distances
            + self.iou_weight * self.max_distance * iou_cost
            + self.area_weight * self.max_distance * area_cost
            + self.appearance_weight * self.max_distance * appearance_cost
            + self.mahalanobis_weight * self.max_distance * motion_cost
        )
        return costs, valid_area & valid_motion & valid_distance

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


def _observed_trail(history: list[TrajectoryPoint], limit: int) -> tuple[tuple[float, float], ...]:
    """Draw only the latest uninterrupted sequence of observed positions."""
    if limit <= 0:
        return ()
    points = []
    for point in reversed(history[-limit:]):
        if not point.matched:
            break
        points.append(point.centroid)
    return tuple(reversed(points))


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


def _appearance_cost_matrix(
    tracks: list[_Track], detections: list[Detection]
) -> np.ndarray:
    """Compute Hellinger histogram costs in batches for matching shapes."""
    costs = np.zeros((len(tracks), len(detections)), dtype=float)
    track_groups: dict[tuple[int, ...], list[tuple[int, np.ndarray]]] = {}
    detection_groups: dict[tuple[int, ...], list[tuple[int, np.ndarray]]] = {}

    for index, track in enumerate(tracks):
        if track.appearance is None:
            continue
        histogram = np.asarray(track.appearance, dtype=float)
        total = float(histogram.sum())
        if total > 0:
            track_groups.setdefault(histogram.shape, []).append(
                (index, np.sqrt(np.maximum(histogram, 0.0) / total).reshape(-1))
            )
    for index, detection in enumerate(detections):
        if detection.appearance is None:
            continue
        histogram = np.asarray(detection.appearance, dtype=float)
        total = float(histogram.sum())
        if total > 0:
            detection_groups.setdefault(histogram.shape, []).append(
                (index, np.sqrt(np.maximum(histogram, 0.0) / total).reshape(-1))
            )

    for shape in track_groups.keys() & detection_groups.keys():
        track_entries = track_groups[shape]
        detection_entries = detection_groups[shape]
        track_indices = [index for index, _ in track_entries]
        detection_indices = [index for index, _ in detection_entries]
        track_roots = np.stack([histogram for _, histogram in track_entries])
        detection_roots = np.stack([histogram for _, histogram in detection_entries])
        coefficients = np.clip(track_roots @ detection_roots.T, 0.0, 1.0)
        costs[np.ix_(track_indices, detection_indices)] = np.sqrt(
            np.maximum(0.0, 1.0 - coefficients)
        )
    return costs

