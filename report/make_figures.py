"""Tạo các hình minh hoạ cho báo cáo (Chương 4-5) từ mã nguồn và dữ liệu thật.

    python report/make_figures.py        # ghi vào report/figures/
"""

from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python")]
from main import build_config  # noqa: E402
from object_tracking.detection import ForegroundDetector  # noqa: E402
from object_tracking.frame_source import open_frame_source  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def frame_at(path: Path, index: int) -> np.ndarray:
    source = open_frame_source(path)
    for _ in range(index + 1):
        ok, frame = source.read()
        assert ok
    source.release()
    return frame


# ------------------------------------------------------------------ Hình 4.7
def fiji_track_lengths() -> None:
    path = ROOT / "Báo cáo XLA/TrackMate_GUI_A/results/track_2D_A_0000_tracks.csv"
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header, data = rows[0], rows[4:]
    n = sorted((int(float(r[header.index("NUMBER_SPOTS")])) for r in data), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(1, len(n) + 1), n, color="#2b6cb0")
    ax.axhline(100, color="#c53030", ls="--", lw=1)
    ax.text(len(n), 102, "độ dài GT = 100 frame", color="#c53030", ha="right", fontsize=9)
    ax.set_xlabel("track (sắp xếp giảm dần)")
    ax.set_ylabel("số điểm trong track")
    ax.set_title(f"TrackMate trên ảnh A: {len(n)} track, tổng {sum(n)} điểm / 2000 điểm GT", fontsize=10)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(OUT / "fiji_track_lengths.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- Hình 5.1/5.6
def box(ax, xy, w, h, text, color="#ebf4ff", edge="#2b6cb0", size=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec=edge, lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, a, b, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#2d3748"))
    if text:
        ax.text((a[0] + b[0]) / 2 + 0.05, (a[1] + b[1]) / 2, text, fontsize=8, color="#2d3748")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, (0.2, 4.6), 2.4, 0.9, "main.py\nbuild_config()\nTham số theo loại dữ liệu", "#fffaf0", "#c05621")
    box(ax, (3.1, 4.6), 2.4, 0.9, "pipeline.py\nprocess_video()\nvòng lặp theo frame", "#fffaf0", "#c05621")
    box(ax, (6.0, 4.6), 3.6, 0.9, "frame_source.py\nVideoFrameSource (OpenCV)\nTiffFrameSource (Pillow, ImageJ fps)", "#f0fff4", "#2f855a")
    box(ax, (0.2, 2.9), 3.3, 1.0, "detection.py\nForegroundDetector: MOG2 / top-hat\nlọc nhiễu, gộp vùng, centroid", "#ebf4ff")
    box(ax, (3.9, 2.9), 3.0, 1.0, "tracker.py\nMultiObjectTracker\nquản lý track, phục hồi ID", "#ebf4ff")
    box(ax, (7.3, 2.9), 2.3, 1.0, "kalman.py\nKalmanFilter2D\n[x, y, vx, vy]", "#ebf4ff")
    box(ax, (3.9, 1.3), 3.0, 1.0, "assignment.py\nnearest_neighbor_assignment\nhungarian_assignment", "#ebf4ff")
    box(ax, (0.2, 1.3), 3.3, 1.0, "models.py\nDetection, TrajectoryPoint,\nTrackSnapshot", "#f7fafc", "#718096")
    box(ax, (7.3, 1.3), 2.3, 1.0, "visualization.py\nvẽ box, ID, quỹ đạo", "#f0fff4", "#2f855a")
    box(ax, (3.1, 0.05), 3.4, 0.8, "export.py  →  trajectories_*.csv / .json\nannotated_*.mp4", "#f0fff4", "#2f855a")
    arrow(ax, (2.6, 5.05), (3.1, 5.05))
    arrow(ax, (5.5, 5.05), (6.0, 5.05))
    arrow(ax, (4.3, 4.6), (2.2, 3.9), "frame")
    arrow(ax, (3.5, 3.4), (3.9, 3.4), "Detection")
    arrow(ax, (6.9, 3.4), (7.3, 3.4))
    arrow(ax, (5.4, 2.9), (5.4, 2.3))
    arrow(ax, (5.0, 1.3), (5.0, 0.85))
    arrow(ax, (7.3, 1.8), (6.5, 0.6))
    fig.tight_layout()
    fig.savefig(OUT / "py_architecture.png", dpi=160)
    plt.close(fig)


def flowchart() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 8.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    steps = [
        "Mở nguồn (video / TIFF), đọc fps",
        "Đọc frame t",
        "Lọc nhiễu giữ biên (bilateral)",
        "Phân đoạn: MOG2 (video)\nhoặc top-hat + ngưỡng thích nghi (TIFF)",
        "Hình thái học, contour, lọc diện tích;\ngộp mảnh gần nhau (video)",
        "Centroid, bbox, diện tích, histogram màu",
        "Kalman dự đoán vị trí mọi track",
        "Ma trận chi phí + cổng (khoảng cách,\nMahalanobis, tỉ lệ diện tích)",
        "Nearest (tham lam) hoặc Hungarian",
        "Cập nhật Kalman; tạo track mới;\nphục hồi ID; xoá track mất quá lâu",
        "Ghi quỹ đạo, vẽ kết quả",
    ]
    y = 13.0
    for i, text in enumerate(steps):
        box(ax, (2.0, y - 0.45), 6.0, 0.9, text, "#ebf4ff" if i not in (8,) else "#fefcbf", size=9)
        if i < len(steps) - 1:
            arrow(ax, (5.0, y - 0.45), (5.0, y - 0.95))
        y -= 1.2
    ax.plot([8.0, 9.2, 9.2], [1.15, 1.15, 12.55], color="#c53030", lw=1.2)
    ax.add_patch(FancyArrowPatch((9.2, 12.55), (8.0, 12.55), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#c53030"))
    ax.text(9.35, 7.0, "còn frame? lặp lại", color="#c53030", fontsize=8.5, rotation=90, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "py_flowchart.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Hình 5.3
def preprocessing_tiff() -> None:
    path = ROOT / "dataset/data3/track_2D_A_0000.tif"
    frame = frame_at(path, 60)
    cfg = build_config(path)
    detector = ForegroundDetector(cfg.detector)
    smooth = cv2.bilateralFilter(frame, 3, 3.0, 1.0)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    response = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, detector.spot_background_kernel)
    mask, detections = detector.apply(frame)
    overlay = frame.copy()
    for d in detections:
        x, y, w, h = d.bbox
        cv2.rectangle(overlay, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 1)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    gain = lambda a: np.clip(a.astype(float) * 255.0 / max(60, a.max() * 0.6), 0, 255).astype(np.uint8)  # noqa: E731
    panels = [
        ("(a) ảnh gốc (frame 60)", gain(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), "gray"),
        ("(b) sau lọc giữ biên + top-hat", gain(response), "gray"),
        (f"(c) mặt nạ (ngưỡng {detector._spot_threshold:.1f})", mask, "gray"),
        (f"(d) {len(detections)} phát hiện", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), None),
    ]
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "py_prep_tiff.png", dpi=160)
    plt.close(fig)


def merge_effect() -> None:
    path = ROOT / "dataset/data1/video1.mpg"
    cfg = build_config(path)
    results = {}
    for label, gaps in (("không gộp", (0, 0)), ("gộp (6, 16)", (6, 16))):
        det = ForegroundDetector(dataclasses.replace(cfg.detector, merge_gap_x=gaps[0], merge_gap_y=gaps[1]))
        source = open_frame_source(path)
        for index in range(161):
            ok, frame = source.read()
            mask, detections = det.apply(frame)
        source.release()
        shown = frame.copy()
        for d in detections:
            x, y, w, h = d.bbox
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(shown, (int(d.centroid[0]), int(d.centroid[1])), 2, (0, 0, 255), -1)
        results[label] = (mask, shown, len(detections))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) frame 160", fontsize=9)
    axes[1].imshow(cv2.cvtColor(results["không gộp"][1], cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"(b) không gộp: {results['không gộp'][2]} vùng", fontsize=9)
    axes[2].imshow(cv2.cvtColor(results["gộp (6, 16)"][1], cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"(c) gộp (6, 16): {results['gộp (6, 16)'][2]} vùng", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "py_merge_effect.png", dpi=160)
    plt.close(fig)
    print("detections per label:", {k: v[2] for k, v in results.items()})


def main() -> None:
    fiji_track_lengths()
    architecture()
    flowchart()
    preprocessing_tiff()
    merge_effect()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
