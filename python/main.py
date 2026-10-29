"""Run the tracking demo on every supported file in the dataset directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from object_tracking.detection import DetectorConfig
from object_tracking.frame_source import TIFF_EXTENSIONS
from object_tracking.pipeline import PipelineConfig, TrackerConfig, process_video


VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".flv",
    ".m2ts",
    ".m4v",
    ".mjpeg",
    ".mjpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ts",
    ".vob",
    ".webm",
    ".wmv",
}
INPUT_EXTENSIONS = VIDEO_EXTENSIONS | TIFF_EXTENSIONS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "python" / "outputs"


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Process every supported file in dataset with both tracking algorithms."
    )


def discover_videos(path: Path) -> list[Path]:
    """Return supported media files directly inside a file or directory."""
    if path.is_file():
        if path.suffix.lower() in INPUT_EXTENSIONS:
            return [path]
        return []
    if path.is_dir():
        return sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in INPUT_EXTENSIONS
        )
    return []


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    videos = discover_videos(DEFAULT_DATASET)
    if not videos:
        print(
            f"No supported media found at: {DEFAULT_DATASET}. "
            f"Supported video extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}; "
            "also supports .tif and .tiff stacks.",
            file=sys.stderr,
        )
        return 2
    algorithms = ("nearest", "hungarian")
    all_results = []
    failed = False
    stem_counts: dict[str, int] = {}
    for video in videos:
        normalized_stem = video.stem.casefold()
        stem_counts[normalized_stem] = stem_counts.get(normalized_stem, 0) + 1

    used_destinations: set[str] = set()
    print(f"Processing all {len(videos)} supported media file(s).")
    for video in videos:
        config = build_config(video)
        destination_name = video.stem
        if stem_counts[video.stem.casefold()] > 1:
            destination_name = f"{video.stem}_{video.suffix[1:].lower()}"
        base_name = destination_name
        suffix = 2
        while destination_name.casefold() in used_destinations:
            destination_name = f"{base_name}_{suffix}"
            suffix += 1
        used_destinations.add(destination_name.casefold())
        destination = DEFAULT_OUTPUT / destination_name
        print(
            f"Processing {video} -> {destination} "
            f"[algorithms={', '.join(algorithms)}]"
        )
        try:
            last_reported = 0

            def report_progress(frame: int, total: int, elapsed: float) -> None:
                nonlocal last_reported
                if frame - last_reported < 100 and frame != total:
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
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = DEFAULT_OUTPUT / "summary.json"
    summary_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done. Summary: {summary_path}")
    return 1 if failed else 0


def build_config(input_path: Path) -> PipelineConfig:
    microscopy = input_path.suffix.lower() in TIFF_EXTENSIONS
    return PipelineConfig(
        detector=DetectorConfig(
            segmentation_mode="bright_spots" if microscopy else "mog2",
            detect_shadows=not microscopy,
            min_area=4.0,
            min_width=2,
            min_height=2,
            kernel_size=3,
            open_iterations=0,
            close_iterations=1,
            max_foreground_ratio=0.90 if microscopy else 0.65,
            adaptive_min_area=True,
            adaptive_spot_threshold=True,
        ),
        tracker=TrackerConfig(
            max_distance=15.0 if microscopy else 60.0,
            max_missed=10 if microscopy else 15,
            min_hits=3,
            adaptive_confirmation=True,
            min_motion_distance=0.0 if microscopy else 2.0,
            adaptive_process_noise=True,
            iou_weight=0.05 if microscopy else 0.25,
            area_weight=0.05 if microscopy else 0.10,
            max_area_ratio=6.0 if microscopy else 4.0,
            appearance_weight=0.0 if microscopy else 0.50,
            mahalanobis_weight=0.75 if microscopy else 0.25,
        ),
        save_mask=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
