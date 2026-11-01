"""Quét tham số detector/tracker cho ảnh TIFF (hạt sáng) và chấm bằng evaluation/.

Mỗi cấu hình: chạy pipeline Python (Hungarian) -> đọc trajectories -> so với GT
(dịch GT -1,-1 như ``evaluate.py``) -> ghi một dòng vào CSV.

Ví dụ::

    # quét từng tham số một quanh cấu hình gốc (nhanh, ~20 lần chạy)
    python experiments/tune_detector_tiff.py --data dataset/data4/track_2D_C_0002

    # lưới đầy đủ trên vài tham số chính
    python experiments/tune_detector_tiff.py --data dataset/data4/track_2D_C_0002 --mode grid

    # kiểm tra chéo: cấu hình tìm được trên C chạy lại trên A
    python experiments/tune_detector_tiff.py --data dataset/data3/track_2D_A_0000 --mode one-at-a-time

    # chạy riêng vài cấu hình (kèm baseline) trên một bộ khác
    python experiments/tune_detector_tiff.py --data dataset/data3/track_2D_A_0000 \\
        --config "adaptive_threshold_sigma=5.0,min_area=6.0"

Thiết kế (xem experiments/README.md): chia THEO THỜI GIAN, nửa đầu chọn cấu hình
(tune), nửa sau kiểm (test); cấu hình thắng phải kiểm chéo trên bộ còn lại.

Kết quả: ``experiments/results/<stem>_<mode>.csv`` (sắp xếp theo IDF1 tune giảm dần),
cấu hình gốc luôn là dòng ``baseline``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "evaluation")]

from evaluate import apply_gt_offset, crop_point_gt, media_size  # noqa: E402
from loaders import load_tracks  # noqa: E402
from main import build_config  # noqa: E402
from metrics import MatchConfig, evaluate  # noqa: E402
from object_tracking.pipeline import process_video  # noqa: E402

# Giá trị thử cho từng tham số. Tên có tiền tố ``det.`` thuộc DetectorConfig,
# ``trk.`` thuộc TrackerConfig. Giá trị đầu tiên không có ý nghĩa đặc biệt;
# baseline lấy từ main.build_config.
SPACE: dict[str, list] = {
    "det.adaptive_threshold_sigma": [3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
    "det.spot_threshold": [5, 10, 15, 25],
    "det.spot_background_size": [5, 7, 9, 11, 15],
    "det.kernel_size": [3, 5],
    "det.close_iterations": [0, 1, 2],
    "det.min_area": [2.0, 4.0, 6.0, 9.0, 12.0],
    "trk.min_hits": [2, 3, 4, 5],
    "trk.max_missed": [5, 10, 15, 20],
    "trk.measurement_variance": [0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
    "trk.process_variance": [5.0, 20.0, 50.0, 100.0, 300.0, 1000.0],
    "trk.mahalanobis_gate": [9.0, 25.0, 100.0, 400.0, 1000.0],
    "trk.max_distance": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0],
}

# Lưới đầy đủ (mode=grid): chỉ những tham số ảnh hưởng nhiều nhất.
# (min_area, min_hits, max_missed, spot_threshold không đổi kết quả ở quét 1 chiều
# nên bỏ khỏi lưới.)
GRIDS: dict[str, dict[str, list]] = {}
GRIDS["tracker"] = {  # H5: lọc Kalman chặt hơn nhưng vẫn cho hạt nhảy xa
    "trk.measurement_variance": [1.0, 2.0, 5.0],
    "trk.process_variance": [50.0, 300.0, 1000.0],
    "trk.mahalanobis_gate": [25.0, 100.0, 400.0],
    "trk.max_distance": [10.0, 15.0, 25.0],
}
GRID_SPACE: dict[str, list] = {
    "trk.max_distance": [4.0, 5.0, 6.0, 8.0, 10.0],
    "det.spot_background_size": [5, 7, 9],
    "det.adaptive_threshold_sigma": [4.0, 4.5, 5.0],
    "det.close_iterations": [0, 1],
}

SEGMENTS = ("all", "tune", "test")  # toàn bộ / nửa đầu / nửa sau
FIELDS = (
    ["name", "precision", "recall", "F1", "RMSE", "MOTA", "IDF1",
     "IDSW", "FM", "MT", "PT", "ML", "pred_tracks"]
    + [f"{seg}_{k}" for seg in ("tune", "test") for k in ("F1", "IDF1", "FM")]
    + ["seconds"]
)


def split_frames(gt) -> dict[str, tuple[int, int]]:
    first, last = gt.frame_range
    mid = (first + last) // 2
    return {"all": (first, last), "tune": (first, mid), "test": (mid + 1, last)}


def parse_config(text: str) -> dict:
    """``"sigma=5.0,min_area=6"`` -> {'det.adaptive_threshold_sigma': 5.0, ...}."""
    known = {k.split(".", 1)[1]: k for k in SPACE}
    out = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        name, value = item.split("=", 1)
        if name not in known:
            raise SystemExit(f"tham số lạ {name!r}; chọn trong {sorted(known)}")
        out[known[name]] = type(SPACE[known[name]][0])(value)
    return out


def apply_overrides(config, overrides: dict):
    detector = dataclasses.replace(
        config.detector,
        **{k[4:]: v for k, v in overrides.items() if k.startswith("det.")},
    )
    tracker = dataclasses.replace(
        config.tracker,
        **{k[4:]: v for k, v in overrides.items() if k.startswith("trk.")},
    )
    return dataclasses.replace(config, detector=detector, tracker=tracker, save_mask=False)


def label(overrides: dict) -> str:
    if not overrides:
        return "baseline"
    return ",".join(f"{k.split('.', 1)[1]}={v}" for k, v in overrides.items())


def candidates(mode: str, grid: str = "detector") -> list[dict]:
    runs: list[dict] = [{}]
    if mode == "one-at-a-time":
        for key, values in SPACE.items():
            runs += [{key: v} for v in values]
    else:
        space = GRIDS[grid]
        keys = list(space)
        runs += [dict(zip(keys, c)) for c in itertools.product(*(space[k] for k in keys))]
    seen, unique = set(), []
    for run in runs:
        marker = tuple(sorted(run.items()))
        if marker not in seen:
            seen.add(marker)
            unique.append(run)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path,
                        help="đường dẫn không đuôi, vd dataset/data4/track_2D_C_0002 (cần .tif và .csv)")
    parser.add_argument("--mode", choices=("one-at-a-time", "grid"), default="one-at-a-time")
    parser.add_argument("--grid", choices=("detector", "tracker"), default="detector",
                        help="bộ tham số của --mode grid")
    parser.add_argument("--config", action="append", default=[], metavar="K=V,K=V",
                        help="chỉ chạy cấu hình này (lặp được) cùng baseline, bỏ qua --mode")
    parser.add_argument("--legacy-tracker", action="store_true",
                        help="baseline = tham số Kalman TRƯỚC thí nghiệm (variance 25, process 5, Mahalanobis 25)")
    parser.add_argument("--keep-outside-gt", action="store_true",
                        help="không loại điểm GT nằm ngoài khung ảnh")
    parser.add_argument("--gate", type=float, default=5.0, help="ngưỡng ghép cặp (px)")
    parser.add_argument("--algorithm", choices=("nearest", "hungarian"), default="hungarian")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results")
    args = parser.parse_args()

    data = args.data if args.data.is_absolute() else ROOT / args.data
    tif, gt_path = data.with_suffix(".tif"), data.with_suffix(".csv")
    for path in (tif, gt_path):
        if not path.exists():
            parser.error(f"không thấy {path}")

    gt, offset = apply_gt_offset(load_tracks(gt_path), gt_path, "auto")
    excluded = 0
    if not args.keep_outside_gt:
        gt, excluded = crop_point_gt(gt, gt_path, media_size(tif))
    match = MatchConfig("distance", args.gate)
    base = build_config(tif)
    if args.legacy_tracker:
        base = apply_overrides(base, {"trk.measurement_variance": 25.0, "trk.process_variance": 5.0,
                                      "trk.mahalanobis_gate": 25.0})
    runs = [{}] + [parse_config(c) for c in args.config] if args.config else candidates(args.mode, args.grid)
    ranges = split_frames(gt)
    print(f"{data.stem}: {len(runs)} cấu hình, GT offset {offset}, GT ngoài khung bị loại: {excluded}")

    rows = []
    work = Path(tempfile.mkdtemp(prefix="tune_"))
    try:
        for index, overrides in enumerate(runs, 1):
            started = time.time()
            out = work / f"run{index}"
            try:
                process_video(tif, out, (args.algorithm,), apply_overrides(base, overrides))
                pred = load_tracks(out / f"trajectories_{args.algorithm}.csv")
                per = {
                    seg: evaluate(
                        gt.restrict_frames(*ranges[seg]), pred.restrict_frames(*ranges[seg]), match
                    ).metrics
                    for seg in SEGMENTS
                }
                m = per["all"]
            except Exception as error:  # cấu hình không hợp lệ không được làm hỏng cả đợt
                print(f"[{index}/{len(runs)}] {label(overrides)}: LỖI {error}")
                continue
            finally:
                shutil.rmtree(out, ignore_errors=True)
            row = {
                "name": label(overrides),
                "precision": m["precision"], "recall": m["recall"], "F1": m["F1"],
                "RMSE": m["RMSE_px"], "MOTA": m["MOTA"], "IDF1": m["IDF1"],
                "IDSW": m["IDSW"], "FM": m["FM"], "MT": m["MT"], "PT": m["PT"], "ML": m["ML"],
                "pred_tracks": m["pred_tracks"],
                **{f"{seg}_{k}": per[seg][k] for seg in ("tune", "test") for k in ("F1", "IDF1", "FM")},
                "seconds": round(time.time() - started, 1),
            }
            rows.append(row)
            print(f"[{index}/{len(runs)}] {row['name']:<60} P={row['precision']:.3f} "
                  f"R={row['recall']:.3f} F1={row['F1']:.3f} IDF1={row['IDF1']:.3f} tune={row['tune_F1']:.3f} "
                  f"test={row['test_F1']:.3f} FM={row['FM']} ({row['seconds']}s)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    rows.sort(key=lambda r: (r["name"] != "baseline", -r["tune_IDF1"], -r["tune_F1"]))
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / (f"{data.stem}_{args.grid}-grid.csv" if args.mode == "grid" and not args.config else f"{data.stem}_{args.mode}.csv")
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in FIELDS})
    print(f"\nĐã ghi {target}")
    baseline = next((r for r in rows if r["name"] == "baseline"), None)
    others = [r for r in rows if r["name"] != "baseline"]
    if baseline and others:
        best = max(others, key=lambda r: (r["tune_IDF1"], r["tune_F1"]))  # chọn CHỈ theo nửa tune
        print(f"Chọn theo nửa tune: {best['name']}")
        for seg in ("tune", "test"):
            print(f"  {seg:<4} IDF1 {baseline[seg + '_IDF1']:.3f} -> {best[seg + '_IDF1']:.3f}   "
                  f"F1 {baseline[seg + '_F1']:.3f} -> {best[seg + '_F1']:.3f}   "
                  f"FM {baseline[seg + '_FM']} -> {best[seg + '_FM']}")
        print("  Nếu test không tăng theo tune thì đó là quá khớp, đừng dùng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
