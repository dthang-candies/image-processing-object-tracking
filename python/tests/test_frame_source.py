from pathlib import Path
import json

import cv2
import numpy as np
import pytest
from PIL import Image

import main as demo
from object_tracking.detection import ForegroundDetector
from object_tracking.frame_source import TiffFrameSource, open_frame_source


def make_tiff_stack(path: Path) -> None:
    frames = [
        Image.fromarray(np.full((12, 16), value, dtype=np.uint8))
        for value in (10, 40, 90)
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        tiffinfo={270: "ImageJ=1.53\nimages=3\nframes=3\nfinterval=0.05\n"},
    )


def test_tiff_stack_reads_all_pages_as_bgr_and_uses_imagej_fps(tmp_path):
    path = tmp_path / "stack.tif"
    make_tiff_stack(path)

    source = TiffFrameSource(path)
    frames = []
    while True:
        ok, frame = source.read()
        if not ok:
            break
        frames.append(frame)
    source.release()

    assert source.frame_count == 3
    assert source.fps == 20.0
    assert all(frame.shape == (12, 16, 3) for frame in frames)
    assert [int(frame[0, 0, 0]) for frame in frames] == [10, 40, 90]


def test_tiff_fps_override_takes_priority(tmp_path):
    path = tmp_path / "stack.tiff"
    make_tiff_stack(path)

    source = open_frame_source(path, tiff_fps=7.5)
    try:
        assert source.source_type == "tiff_stack"
        assert source.fps == 7.5
    finally:
        source.release()


def test_discover_videos_lists_supported_files(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.tif"
    unsupported = tmp_path / "notes.txt"
    for path in (first, second, unsupported):
        path.touch()

    assert demo.discover_videos(tmp_path) == [first, second]


def test_default_video_detector_keeps_small_moving_objects():
    detector = ForegroundDetector(demo.build_config(Path("small.mp4")).detector)
    background = np.zeros((50, 50, 3), dtype=np.uint8)
    detector.apply(background)
    frame = background.copy()
    cv2.rectangle(frame, (20, 20), (24, 24), (255, 255, 255), -1)

    _, detections = detector.apply(frame)

    assert any(abs(item.centroid[0] - 22) < 1 for item in detections)


def test_demo_processes_all_formats_with_separate_outputs(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    output = tmp_path / "outputs"
    dataset.mkdir()
    make_tiff_stack(dataset / "sample.tif")
    make_tiff_stack(dataset / "sample_mp4.tif")
    writer = cv2.VideoWriter(
        str(dataset / "sample.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (16, 12)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot encode MP4 in this environment")
    for x in (2, 4, 6):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        cv2.rectangle(frame, (x, 4), (x + 3, 7), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    monkeypatch.setattr(demo, "DEFAULT_DATASET", dataset)
    monkeypatch.setattr(demo, "DEFAULT_OUTPUT", output)

    assert demo.main([]) == 0

    results = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(results) == 3
    assert all(result["frames"] == 3 for result in results)
    assert all(set(result["algorithms"]) == {"nearest", "hungarian"} for result in results)
    for name in ("sample_mp4", "sample_tif", "sample_mp4_2"):
        for algorithm in ("nearest", "hungarian"):
            assert (output / name / f"annotated_{algorithm}.mp4").is_file()
            assert (output / name / f"trajectories_{algorithm}.csv").is_file()
