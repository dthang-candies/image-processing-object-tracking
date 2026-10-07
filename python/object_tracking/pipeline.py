"""End-to-end video processing pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from .detection import DetectorConfig, ForegroundDetector
from .export import save_trajectories
from .tracker import MultiObjectTracker
from .visualization import annotate_frame


@dataclass(slots=True)
class TrackerConfig:
    max_distance: float = 60.0
    max_missed: int = 15
    min_hits: int = 2
    process_variance: float = 5.0
    measurement_variance: float = 25.0
    initial_velocity_variance: float = 10_000.0
    iou_weight: float = 0.25
    area_weight: float = 0.10
    max_area_ratio: float = 4.0
    appearance_weight: float = 0.50
    appearance_momentum: float = 0.90


@dataclass(slots=True)
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    warmup_frames: int = 10
    save_mask: bool = True
    display: bool = False
    codec: str = "mp4v"
    max_frames: int | None = None


def process_video(
    input_path: Path,
    output_dir: Path,
    algorithms: tuple[str, ...] = ("hungarian",),
    config: PipelineConfig | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict:
    """Process one video once and run one or both association methods."""
    config = config or PipelineConfig()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")
    invalid = set(algorithms) - {"nearest", "hungarian"}
    if invalid:
        raise ValueError(f"Unsupported algorithms: {sorted(invalid)}")
    if not algorithms:
        raise ValueError("At least one association algorithm is required")
    if len(config.codec) != 4:
        raise ValueError("codec must contain exactly four characters")
    if config.warmup_frames < 0:
        raise ValueError("warmup_frames cannot be negative")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps != fps:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame_count = (
        min(source_frame_count, config.max_frames)
        if source_frame_count > 0 and config.max_frames is not None
        else config.max_frames or source_frame_count
    )
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid dimensions: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    detector = ForegroundDetector(config.detector)
    tracker_settings = asdict(config.tracker)
    trackers = {
        algorithm: MultiObjectTracker(association=algorithm, **tracker_settings)
        for algorithm in algorithms
    }
    fourcc = cv2.VideoWriter_fourcc(*config.codec)
    writers = {
        algorithm: cv2.VideoWriter(
            str(output_dir / f"annotated_{algorithm}.mp4"),
            fourcc,
            fps,
            (width, height),
        )
        for algorithm in algorithms
    }
    mask_writer = (
        cv2.VideoWriter(
            str(output_dir / "foreground_mask.mp4"), fourcc, fps, (width, height)
        )
        if config.save_mask
        else None
    )
    video_outputs_ready = all(writer.isOpened() for writer in writers.values())
    mask_output_ready = mask_writer is None or mask_writer.isOpened()
    if not video_outputs_ready or not mask_output_ready:
        capture.release()
        for writer in writers.values():
            writer.release()
        if mask_writer is not None:
            mask_writer.release()
        raise RuntimeError(f"Cannot create output video(s) with codec {config.codec!r}")

    frame_index = 0
    association_seconds = {algorithm: 0.0 for algorithm in algorithms}
    started = time.perf_counter()
    try:
        while True:
            if config.max_frames is not None and frame_index >= config.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            mask, detections = detector.apply(frame)
            effective_detections = detections if frame_index >= config.warmup_frames else []
            current_time = frame_index / fps
            for algorithm, tracker in trackers.items():
                association_started = time.perf_counter()
                snapshots = tracker.update(
                    effective_detections,
                    frame_index=frame_index,
                    time_seconds=current_time,
                    dt=1.0 / fps,
                )
                association_seconds[algorithm] += time.perf_counter() - association_started
                writers[algorithm].write(
                    annotate_frame(frame, effective_detections, snapshots, algorithm)
                )
            if mask_writer is not None:
                mask_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            if config.display:
                preview_algorithm = algorithms[0]
                preview = annotate_frame(
                    frame,
                    effective_detections,
                    trackers[preview_algorithm].snapshots(),
                    preview_algorithm,
                )
                cv2.imshow("Object tracking - press q to stop", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
            if progress_callback is not None:
                progress_callback(
                    frame_index,
                    target_frame_count,
                    time.perf_counter() - started,
                )
    finally:
        capture.release()
        for writer in writers.values():
            writer.release()
        if mask_writer is not None:
            mask_writer.release()
        if config.display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    results: dict = {
        "input": str(input_path),
        "processing_device": detector.device,
        "processing_device_name": detector.device_name,
        "frames": frame_index,
        "source_fps": fps,
        "source_frame_count": source_frame_count,
        "processing_seconds": elapsed,
        "processing_fps": frame_index / elapsed if elapsed else 0.0,
        "algorithms": {},
    }
    for algorithm, tracker in trackers.items():
        trajectories = tracker.trajectories
        csv_path, json_path = save_trajectories(
            trajectories, output_dir / f"trajectories_{algorithm}"
        )
        track_lengths = [len(points) for points in trajectories.values()]
        point_count = sum(track_lengths)
        matched_points = sum(
            point.matched for points in trajectories.values() for point in points
        )
        speeds = [
            (point.velocity[0] ** 2 + point.velocity[1] ** 2) ** 0.5
            for points in trajectories.values()
            for point in points
            if point.matched
        ]
        results["algorithms"][algorithm] = {
            "track_count": len(trajectories),
            "trajectory_points": point_count,
            "matched_points": matched_points,
            "predicted_points": point_count - matched_points,
            "matched_fraction": matched_points / point_count if point_count else 0.0,
            "mean_track_length": (
                sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
            ),
            "mean_speed_pixels_per_second": (
                sum(speeds) / len(speeds) if speeds else 0.0
            ),
            "association_seconds": association_seconds[algorithm],
            "trajectory_csv": str(csv_path),
            "trajectory_json": str(json_path),
            "annotated_video": str(output_dir / f"annotated_{algorithm}.mp4"),
        }
    return results
