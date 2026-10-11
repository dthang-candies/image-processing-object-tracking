"""Data structures shared by the detector, tracker, and exporters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BBox = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass(slots=True)
class Detection:
    """One segmented object in a video frame."""

    bbox: BBox
    centroid: Point
    area: float
    contour: Any | None = None
    appearance: Any | None = None


@dataclass(slots=True)
class TrajectoryPoint:
    """Observed or Kalman-predicted position of a track at one frame."""

    frame_index: int
    time_seconds: float
    centroid: Point
    bbox: BBox
    matched: bool
    velocity: Point = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """Read-only representation of an active track for visualization."""

    track_id: int
    centroid: Point
    bbox: BBox
    age: int
    hits: int
    missed: int
    confirmed: bool
    trail: tuple[Point, ...]
    velocity: Point

