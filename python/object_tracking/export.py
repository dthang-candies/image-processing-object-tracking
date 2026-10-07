"""Trajectory serialization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import TrajectoryPoint


def save_trajectories(
    trajectories: dict[int, list[TrajectoryPoint]], output_prefix: Path
) -> tuple[Path, Path]:
    """Save trajectory records as analysis-friendly CSV and structured JSON."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "track_id",
                "frame",
                "time_seconds",
                "x",
                "y",
                "vx",
                "vy",
                "speed",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "matched",
            ]
        )
        for track_id, points in sorted(trajectories.items()):
            for point in points:
                writer.writerow(
                    [
                        track_id,
                        point.frame_index,
                        f"{point.time_seconds:.6f}",
                        f"{point.centroid[0]:.3f}",
                        f"{point.centroid[1]:.3f}",
                        f"{point.velocity[0]:.3f}",
                        f"{point.velocity[1]:.3f}",
                        f"{(point.velocity[0] ** 2 + point.velocity[1] ** 2) ** 0.5:.3f}",
                        *point.bbox,
                        int(point.matched),
                    ]
                )

    document = {
        "tracks": [
            {
                "track_id": track_id,
                "points": [
                    {
                        "frame": point.frame_index,
                        "time_seconds": round(point.time_seconds, 6),
                        "centroid": [round(value, 3) for value in point.centroid],
                        "velocity": [round(value, 3) for value in point.velocity],
                        "speed": round(
                            (point.velocity[0] ** 2 + point.velocity[1] ** 2) ** 0.5,
                            3,
                        ),
                        "bbox": list(point.bbox),
                        "matched": point.matched,
                    }
                    for point in points
                ],
            }
            for track_id, points in sorted(trajectories.items())
        ]
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
    return csv_path, json_path

