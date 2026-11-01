"""Nearest và Hungarian có bao giờ chọn phép ghép KHÁC nhau trên cùng ma trận chi phí không?

Chạy tracker Hungarian; ở mỗi bước ghép, gọi thêm Nearest trên đúng dữ liệu đó và so sánh.
  contested : số bước có >= 2 track cùng chọn một detection rẻ nhất (nơi có thể xảy ra xung đột)
  differ    : số bước hai thuật toán cho tập cặp khác nhau

    python experiments/compare_association.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python")]

import object_tracking.tracker as T  # noqa: E402
from main import build_config  # noqa: E402
from object_tracking.pipeline import process_video  # noqa: E402

stats: dict[str, int] = {}
_hungarian = T.hungarian_assignment


def _compare(pred, meas, gate, *, pair_costs=None, valid_mask=None):
    hung = _hungarian(pred, meas, gate, pair_costs=pair_costs, valid_mask=valid_mask)
    near = T.nearest_neighbor_assignment(pred, meas, gate, pair_costs=pair_costs, valid_mask=valid_mask)
    stats["steps"] += 1
    if len(pred) and len(meas):
        stats["nonempty"] += 1
        cost = np.where(valid_mask, pair_costs, np.inf)
        rows = np.isfinite(cost).any(axis=1)
        wanted = cost.argmin(axis=1)[rows]
        stats["contested"] += int(len(wanted) != len(set(wanted)))
    stats["differ"] += int(sorted(hung[0]) != sorted(near[0]))
    return hung


T.hungarian_assignment = _compare
for rel in ("dataset/data3/track_2D_A_0000.tif", "dataset/data4/track_2D_C_0002.tif", "dataset/data1/video1.mpg"):
    stats.update(steps=0, nonempty=0, contested=0, differ=0)
    path = ROOT / rel
    process_video(path, Path(tempfile.mkdtemp()), ("hungarian",), build_config(path))
    print(f"{path.stem:<18} {stats}")
