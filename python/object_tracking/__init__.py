"""Classical detection, segmentation, and multi-object tracking package."""

from .assignment import hungarian_assignment, nearest_neighbor_assignment
from .kalman import KalmanFilter2D
from .models import Detection, TrackSnapshot, TrajectoryPoint
from .tracker import MultiObjectTracker

__all__ = [
    "Detection",
    "KalmanFilter2D",
    "MultiObjectTracker",
    "TrackSnapshot",
    "TrajectoryPoint",
    "hungarian_assignment",
    "nearest_neighbor_assignment",
]

