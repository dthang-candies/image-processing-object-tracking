"""Evaluate Python and Fiji/TrackMate tracking results against ground truth.

Single comparison::

    python evaluation/evaluate.py \\
        --gt dataset/data3/track_2D_A_0000.csv \\
        --pred Hungarian=python/outputs/track_2D_A_0000/trajectories_hungarian.csv \\
        --pred TrackMate="Báo cáo XLA/TrackMate_GUI_A/results/track_2D_A_0000_spots.csv" \\
        --out evaluation/results/track_2D_A_0000

Everything in ``dataset/`` at once (finds the GT, the Python outputs in
``python/outputs`` and the Fiji exports automatically)::

    python evaluation/evaluate.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loaders import FORMATS, TrackSet, detect_format, load_tracks  # noqa: E402
from metrics import MatchConfig, Result, evaluate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "dataset"
PYTHON_OUTPUTS = PROJECT_ROOT / "python" / "outputs"
FIJI_ROOTS = (PROJECT_ROOT / "fiji", PROJECT_ROOT / "Báo cáo XLA")
RESULTS = PROJECT_ROOT / "evaluation" / "results"
MEDIA = {".tif", ".tiff", ".mpg", ".mpeg", ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}

# columns shown in the comparison tables (key, header, format)
TABLE = (
    ("precision", "Precision", "{:.3f}"), ("recall", "Recall", "{:.3f}"), ("F1", "F1", "{:.3f}"),
    ("RMSE_px", "RMSE(px)", "{:.2f}"), ("MOTA", "MOTA", "{:.3f}"), ("MOTP", "MOTP", "{:.3f}"),
    ("IDF1", "IDF1", "{:.3f}"), ("IDSW", "IDSW", "{}"), ("FM", "FM", "{}"),
    ("MT", "MT", "{}"), ("PT", "PT", "{}"), ("ML", "ML", "{}"),
    ("gt_tracks", "GT tracks", "{}"), ("pred_tracks", "Pred tracks", "{}"),
)


# ------------------------------------------------------------------ helpers
def track_stats(tracks: TrackSet) -> dict:
    """Descriptive statistics that need no ground truth."""
    if not len(tracks):
        return {"tracks": 0, "points": 0}
    lengths, speeds = [], []
    for track_id in np.unique(tracks.track_id):
        sel = tracks.track_id == track_id
        order = np.argsort(tracks.frame[sel])
        frames, x, y = tracks.frame[sel][order], tracks.x[sel][order], tracks.y[sel][order]
        lengths.append(len(frames))
        step = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(frames), 1)
        speeds.extend(step.tolist())
    lengths = np.array(lengths)
    return {
        "tracks": int(len(lengths)),
        "points": int(len(tracks)),
        "mean_track_length": float(lengths.mean()),
        "median_track_length": float(np.median(lengths)),
        "tracks_ge_10_frames": int((lengths >= 10).sum()),
        "mean_speed_px_per_frame": float(np.mean(speeds)) if speeds else 0.0,
    }


def fmt(value, pattern: str) -> str:
    return "-" if value is None else pattern.format(value)


def markdown_table(rows: list[tuple[str, dict]], first_header: str = "Method") -> str:
    head = "| " + " | ".join([first_header] + [h for _, h, _ in TABLE]) + " |"
    sep = "|" + "---|" * (len(TABLE) + 1)
    lines = [head, sep]
    for name, metrics in rows:
        cells = [fmt(metrics.get(k), p) for k, _, p in TABLE]
        lines.append("| " + " | ".join([name] + cells) + " |")
    return "\n".join(lines)


def print_table(rows: list[tuple[str, dict]]) -> None:
    width = max([len(n) for n, _ in rows] + [6])
    headers = [h for _, h, _ in TABLE]
    print(f"{'Method':<{width}} " + " ".join(f"{h:>11}" for h in headers))
    for name, metrics in rows:
        cells = [fmt(metrics.get(k), p) for k, _, p in TABLE]
        print(f"{name:<{width}} " + " ".join(f"{c:>11}" for c in cells))


def apply_gt_offset(gt: TrackSet, gt_path: Path, setting: str) -> tuple[TrackSet, tuple[float, float]]:
    """Shift point GT by ``--gt-offset``.

    ``auto`` = (-1, -1) for ``TrackID,x,y,frame`` CSVs: this GT uses 1-based
    pixel coordinates (MATLAB style) while OpenCV/TrackMate use 0-based ones.
    The measured bias of *every* method on data3/data4 is about -1 px in x and
    y, which is the evidence for this choice. Use ``--gt-offset 0,0`` to turn
    it off.
    """
    if setting == "auto":
        offset = (-1.0, -1.0) if detect_format(gt_path) == "gt-csv" else (0.0, 0.0)
    else:
        dx, dy = (float(v) for v in setting.split(","))
        offset = (dx, dy)
    return (gt.shifted(*offset) if any(offset) else gt), offset


# ------------------------------------------------------------------ one run
def run_comparison(
    gt: TrackSet,
    predictions: dict[str, TrackSet],
    config: MatchConfig,
    out_dir: Path | None,
    title: str,
    plots: bool = True,
    cross_reference: tuple[str, TrackSet] | None = None,
    gt_offset: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """Score every prediction against the GT and write JSON/CSV/plots."""
    results: dict[str, Result] = {}
    for name, tracks in predictions.items():
        results[name] = evaluate(gt, tracks, config, name)
    report = {
        "dataset": title,
        "ground_truth": {"objects": len(gt), "applied_offset_px": list(gt_offset), **track_stats(gt)},
        "methods": {n: {**r.metrics, "stats": track_stats(predictions[n])} for n, r in results.items()},
    }
    if cross_reference:
        ref_name, ref_tracks = cross_reference
        report["agreement_with_" + ref_name] = {
            n: evaluate(ref_tracks, t, config, n).metrics
            for n, t in predictions.items() if t is not ref_tracks
        }
    print(f"\n=== {title}  (GT: {len(gt)} objects, {report['ground_truth']['tracks']} tracks) ===")
    print_table([(n, r.metrics) for n, r in results.items()])
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        write_csv(out_dir / "metrics.csv", results)
        for name, result in results.items():
            write_frames(out_dir / f"frames_{slug(name)}.csv", result)
        (out_dir / "metrics.md").write_text(
            f"## {title}\n\n" + markdown_table([(n, r.metrics) for n, r in results.items()]) + "\n",
            encoding="utf-8")
        if plots:
            make_plots(out_dir, gt, predictions, results, title)
    return report


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def write_csv(path: Path, results: dict[str, Result]) -> None:
    keys = ["method"] + list(next(iter(results.values())).metrics) if results else ["method"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        for name, result in results.items():
            writer.writerow([name] + [result.metrics.get(k, "") for k in keys[1:]])


def write_frames(path: Path, result: Result) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "gt", "pred", "tp", "fp", "fn", "idsw"])
        for f in result.frames:
            writer.writerow([f.frame, f.gt, f.pred, f.tp, f.fp, f.fn, f.idsw])


def make_plots(out_dir: Path, gt: TrackSet, predictions: dict[str, TrackSet],
               results: dict[str, Result], title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots", file=sys.stderr)
        return
    names = list(results)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1) bar chart of the headline scores
    keys = [("precision", "Precision"), ("recall", "Recall"), ("F1", "F1"),
            ("MOTA", "MOTA"), ("IDF1", "IDF1")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        values = [max(results[name].metrics.get(k) or 0.0, 0.0) for k, _ in keys]
        bars = ax.bar(np.arange(len(keys)) + i * width, values, width, label=name, color=colors[i % 10])
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(np.arange(len(keys)) + width * (len(names) - 1) / 2, [l for _, l in keys])
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{title} - detection and tracking scores (higher is better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scores.png", dpi=150)
    plt.close(fig)

    # 2) per-frame TP / FP / FN of every method
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for i, name in enumerate(names):
        frames = [f.frame for f in results[name].frames]
        for ax, attr in zip(axes, ("tp", "fp", "fn")):
            ax.plot(frames, [getattr(f, attr) for f in results[name].frames],
                    label=name, color=colors[i % 10], lw=1)
    for ax, label in zip(axes, ("TP", "FP", "FN")):
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"{title} - per-frame matches")
    fig.tight_layout()
    fig.savefig(out_dir / "per_frame.png", dpi=150)
    plt.close(fig)

    # 3) trajectories: GT next to each method
    panels = [("Ground truth", gt)] + [(n, predictions[n]) for n in names]
    cols = min(3, len(panels))
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, tracks) in zip(axes.ravel(), panels):
        ax.axis("on")
        for track_id in np.unique(tracks.track_id):
            sel = tracks.track_id == track_id
            order = np.argsort(tracks.frame[sel])
            ax.plot(tracks.x[sel][order], tracks.y[sel][order], lw=0.8)
        ax.set_title(f"{label} ({len(np.unique(tracks.track_id))} tracks)", fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ batch discovery
def find_gt(media: Path) -> Path | None:
    folder = media.parent
    for candidate in (folder / f"{media.stem}.csv", folder / f"ground_truth_{media.stem}.xml"):
        if candidate.exists():
            return candidate
    for candidate in sorted(folder.glob("*.xml")):
        if candidate.suffix == ".xml" and detect_format(candidate) == "caviar":
            return candidate
    return None


def find_fiji(stem: str, media: Path) -> dict[str, Path]:
    """Fiji/TrackMate exports for one sequence, keyed by display name."""
    found: dict[str, Path] = {}
    for root in FIJI_ROOTS:
        if not root.exists():
            continue
        csvs = sorted(root.rglob(f"{stem}_spots.csv"))
        xmls = sorted(root.rglob(f"{stem}_TrackMate.xml"))
        if csvs:
            found["Fiji-TrackMate"] = csvs[0]
        elif xmls:
            found["Fiji-TrackMate"] = xmls[0]
    dataset_xml = media.with_name(media.name + ".xml")
    if dataset_xml.exists() and detect_format(dataset_xml) == "trackmate-xml":
        found["TrackMate-dataset-XML"] = dataset_xml
    return found


def run_all(config: MatchConfig, out_root: Path, plots: bool, pixel_size: float | None,
            include_predicted: bool, gt_offset: str = "auto") -> int:
    summary: list[tuple[str, str, dict]] = []
    medias = sorted(p for p in DATASET.rglob("*") if p.suffix.lower() in MEDIA
                    and not any(part.startswith(".") for part in p.relative_to(DATASET).parts))
    for media in medias:
        stem = media.stem
        gt_path = find_gt(media)
        predictions: dict[str, TrackSet] = {}
        for algo in ("nearest", "hungarian"):
            path = PYTHON_OUTPUTS / stem / f"trajectories_{algo}.csv"
            if path.exists():
                predictions[f"Python-{algo.capitalize()}"] = load_tracks(
                    path, include_predicted=include_predicted)
        for name, path in find_fiji(stem, media).items():
            try:
                predictions[name] = load_tracks(path, pixel_size=pixel_size)
            except ValueError as error:
                print(f"[skip] {name}: {error}", file=sys.stderr)
        if not predictions:
            print(f"[skip] {stem}: no Python/Fiji result found")
            continue
        if gt_path is None:
            print(f"\n=== {stem}: no ground truth - descriptive statistics only ===")
            for name, tracks in predictions.items():
                print(f"  {name:<24}", {k: round(v, 2) if isinstance(v, float) else v
                                       for k, v in track_stats(tracks).items()})
            out = out_root / stem
            out.mkdir(parents=True, exist_ok=True)
            (out / "stats.json").write_text(json.dumps(
                {n: track_stats(t) for n, t in predictions.items()}, indent=2), encoding="utf-8")
            continue
        gt, offset = apply_gt_offset(load_tracks(gt_path), gt_path, gt_offset)
        fiji_ref = next(((n, t) for n, t in predictions.items() if n.startswith("Fiji")), None)
        report = run_comparison(gt, predictions, config, out_root / stem, stem, plots,
                                fiji_ref, offset)
        for name, metrics in report["methods"].items():
            summary.append((stem, name, metrics))
    if summary:
        write_summary(out_root, summary)
    return 0 if summary else 1


def write_summary(out_root: Path, summary: list[tuple[str, str, dict]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _, _ in TABLE]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "method", "match_mode"] + keys)
        for stem, name, metrics in summary:
            writer.writerow([stem, name, metrics["match_mode"]] + [metrics.get(k, "") for k in keys])
    parts = ["# Evaluation summary\n"]
    for stem in dict.fromkeys(s for s, _, _ in summary):
        rows = [(n, m) for s, n, m in summary if s == stem]
        parts.append(f"## {stem}  (matching: {rows[0][1]['match_mode']})\n")
        parts.append(markdown_table(rows) + "\n")
    (out_root / "summary.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"\nSummary written to {out_root / 'summary.md'}")


# ----------------------------------------------------------------------- CLI
def parse_pred(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name, Path(path)
    return Path(text).stem, Path(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="evaluate every sequence in dataset/")
    p.add_argument("--gt", type=Path, help="ground-truth file")
    p.add_argument("--gt-format", choices=FORMATS, default="auto")
    p.add_argument("--pred", action="append", default=[], metavar="NAME=PATH",
                   help="result file to score (repeatable)")
    p.add_argument("--pred-format", choices=FORMATS, default="auto")
    p.add_argument("--match", choices=("auto", "distance", "iou", "inbox"), default="auto")
    p.add_argument("--gate", type=float, default=5.0, help="max centroid distance in px (point GT)")
    p.add_argument("--iou", type=float, default=0.3, help="min IoU for box GT")
    p.add_argument("--gt-offset", default="auto", metavar="DX,DY",
                   help="shift the GT in px; 'auto' = -1,-1 for 1-based GT CSV (see README)")
    p.add_argument("--pixel-size", type=float, help="um per pixel for TrackMate CSV (auto from XML)")
    p.add_argument("--include-predicted", action="store_true",
                   help="also score Kalman-predicted (matched=0) Python points")
    p.add_argument("--keep-untracked", action="store_true",
                   help="TrackMate: keep spots without track as one-point tracks")
    p.add_argument("--out", type=Path, help="output folder (single mode)")
    p.add_argument("--out-root", type=Path, default=RESULTS, help="output root for --all")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = MatchConfig(args.match, args.gate, args.iou)
    if args.all:
        return run_all(config, args.out_root, not args.no_plots, args.pixel_size,
                       args.include_predicted, args.gt_offset)
    if not args.gt or not args.pred:
        build_parser().error("give --all, or --gt together with at least one --pred")
    gt, offset = apply_gt_offset(load_tracks(args.gt, args.gt_format), args.gt, args.gt_offset)
    predictions = {}
    for item in args.pred:
        name, path = parse_pred(item)
        predictions[name] = load_tracks(
            path, args.pred_format, pixel_size=args.pixel_size,
            include_predicted=args.include_predicted, keep_untracked=args.keep_untracked)
    run_comparison(gt, predictions, config, args.out, args.gt.stem, not args.no_plots,
                   gt_offset=offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
