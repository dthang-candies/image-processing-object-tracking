"""Command-line entry point for processing one video or a video directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from object_tracking.detection import DetectorConfig
from object_tracking.pipeline import PipelineConfig, TrackerConfig, process_video


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect, segment, and track moving objects in video."
    )
    parser.add_argument("input", type=Path, help="Input video or directory of videos")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--algorithm",
        choices=("nearest", "hungarian", "both"),
        default="both",
        help="Data association method (default: both, for comparison)",
    )
    parser.add_argument("--min-area", type=float, default=400.0)
    parser.add_argument("--max-area", type=float, default=None)
    parser.add_argument("--max-distance", type=float, default=60.0)
    parser.add_argument("--max-missed", type=int, default=15)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--iou-weight", type=float, default=0.25)
    parser.add_argument("--area-weight", type=float, default=0.10)
    parser.add_argument("--max-area-ratio", type=float, default=4.0)
    parser.add_argument("--appearance-weight", type=float, default=0.50)
    parser.add_argument("--appearance-momentum", type=float, default=0.90)
    parser.add_argument("--initial-velocity-variance", type=float, default=10_000.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--history", type=int, default=300)
    parser.add_argument("--variance-threshold", type=float, default=24.0)
    parser.add_argument("--learning-rate", type=float, default=-1.0)
    parser.add_argument("--max-foreground-ratio", type=float, default=0.65)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Segmentation device: auto, CPU, or OpenCL GPU (default: auto)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames (useful for a quick test)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N frames; use 0 to disable",
    )
    return parser


def discover_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            item for item in path.iterdir() if item.suffix.lower() in VIDEO_EXTENSIONS
        )
    return []


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    videos = discover_videos(args.input)
    if not videos:
        print(f"No supported videos found at: {args.input}", file=sys.stderr)
        return 2
    algorithms = (
        ("nearest", "hungarian")
        if args.algorithm == "both"
        else (args.algorithm,)
    )
    config = PipelineConfig(
        detector=DetectorConfig(
            history=args.history,
            variance_threshold=args.variance_threshold,
            min_area=args.min_area,
            max_area=args.max_area,
            learning_rate=args.learning_rate,
            max_foreground_ratio=args.max_foreground_ratio,
            device=args.device,
        ),
        tracker=TrackerConfig(
            max_distance=args.max_distance,
            max_missed=args.max_missed,
            min_hits=args.min_hits,
            iou_weight=args.iou_weight,
            area_weight=args.area_weight,
            max_area_ratio=args.max_area_ratio,
            appearance_weight=args.appearance_weight,
            appearance_momentum=args.appearance_momentum,
            initial_velocity_variance=args.initial_velocity_variance,
        ),
        warmup_frames=args.warmup_frames,
        save_mask=not args.no_mask,
        display=args.display,
        max_frames=args.max_frames,
    )

    all_results = []
    failed = False
    for video in videos:
        destination = args.output / video.stem
        print(f"Processing {video} -> {destination}")
        try:
            last_reported = 0

            def report_progress(frame: int, total: int, elapsed: float) -> None:
                nonlocal last_reported
                if args.progress_every <= 0:
                    return
                if frame - last_reported < args.progress_every and frame != total:
                    return
                last_reported = frame
                speed = frame / elapsed if elapsed > 0 else 0.0
                total_text = str(total) if total > 0 else "?"
                print(
                    f"  {frame}/{total_text} frames | {speed:.1f} FPS",
                    flush=True,
                )

            all_results.append(
                process_video(
                    video,
                    destination,
                    algorithms,
                    config,
                    progress_callback=report_progress,
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            failed = True
            print(f"Failed: {video}: {error}", file=sys.stderr)
            all_results.append({"input": str(video), "error": str(error)})
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done. Summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
