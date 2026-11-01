"""Ablation trên video1 (CAVIAR): mỗi cải tiến đóng góp bao nhiêu?

Ba cấu hình, cộng dồn:
  A0  không gộp vùng, không phục hồi ID      (merge_gap=0,0  recover_frames=0)
  A1  + gộp mảnh foreground thành một người  (merge_gap=6,16)
  A2  + phục hồi ID track bị mất             (recover_frames=60)   <- cấu hình cuối

Mỗi cấu hình chạy cả Nearest và Hungarian, chấm bằng evaluation/ (khớp IoU >= 0.3).
Báo cáo thêm hai nửa thời gian (nửa đầu / nửa sau) để thấy độ ổn định.

    python experiments/ablation_video1.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "evaluation"), str(Path(__file__).parent)]

from loaders import load_tracks  # noqa: E402
from main import build_config  # noqa: E402
from metrics import MatchConfig, evaluate  # noqa: E402
from object_tracking.pipeline import process_video  # noqa: E402
from tune_detector_tiff import apply_overrides, split_frames  # noqa: E402

VIDEO = ROOT / "dataset" / "data1" / "video1.mpg"
GT = ROOT / "dataset" / "data1" / "ground_truth_video1.xml"

STEPS = {
    "A0 baseline": {"det.merge_gap_x": 0, "det.merge_gap_y": 0, "trk.recover_frames": 0},
    "A1 +gop vung": {"det.merge_gap_x": 6, "det.merge_gap_y": 16, "trk.recover_frames": 0},
    "A2 +phuc hoi ID": {"det.merge_gap_x": 6, "det.merge_gap_y": 16, "trk.recover_frames": 60},
}
FIELDS = ["config", "method", "precision", "recall", "F1", "MOTA", "IDF1", "IDSW", "FM",
          "MT", "PT", "ML", "pred_tracks", "FP", "FN",
          "first_F1", "first_IDF1", "first_IDSW", "second_F1", "second_IDF1", "second_IDSW"]


def main() -> None:
    gt = load_tracks(GT)
    match = MatchConfig("iou", iou_threshold=0.3)
    ranges = split_frames(gt)
    base = build_config(VIDEO)
    rows = []
    work = Path(tempfile.mkdtemp(prefix="abl_"))
    try:
        for name, overrides in STEPS.items():
            out = work / name.split()[0]
            process_video(VIDEO, out, ("nearest", "hungarian"), apply_overrides(base, overrides))
            for method in ("nearest", "hungarian"):
                pred = load_tracks(out / f"trajectories_{method}.csv")
                per = {seg: evaluate(gt.restrict_frames(*ranges[seg]), pred.restrict_frames(*ranges[seg]), match).metrics
                       for seg in ("all", "tune", "test")}
                m = per["all"]
                row = {"config": name, "method": method}
                row.update({k: m[k] for k in ("precision", "recall", "F1", "MOTA", "IDF1", "IDSW", "FM",
                                              "MT", "PT", "ML", "pred_tracks", "FP", "FN")})
                for seg, label in (("tune", "first"), ("test", "second")):
                    row.update({f"{label}_F1": per[seg]["F1"], f"{label}_IDF1": per[seg]["IDF1"],
                                f"{label}_IDSW": per[seg]["IDSW"]})
                rows.append(row)
                print(f"{name:<18}{method:<10} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['F1']:.3f} "
                      f"MOTA={m['MOTA']:.3f} IDF1={m['IDF1']:.3f} IDSW={m['IDSW']} tracks={m['pred_tracks']}", flush=True)
            shutil.rmtree(out, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    target = ROOT / "experiments" / "results" / "video1_ablation.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"Đã ghi {target}")


if __name__ == "__main__":
    main()
