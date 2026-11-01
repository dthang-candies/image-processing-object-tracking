"""Quantitative tracking metrics (detection, CLEAR-MOT, identity, track level).

All metrics compare a *prediction* (Python or Fiji/TrackMate) to a ground truth
(GT) and use the same matching rule, so the two pipelines are scored fairly.

Matching rule (``match="auto"``)
    * both have boxes   -> IoU  >= ``iou_threshold``      (CAVIAR video)
    * GT box, pred point-> prediction inside the GT box   (TrackMate on video)
    * otherwise         -> centroid distance <= ``gate`` px (particle stacks)

Definitions
    Detection:  P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R)
    CLEAR-MOT:  MOTA = 1 - (FN + FP + IDSW) / #GT,  MOTP = mean match quality
                (Bernardin & Stiefelhagen 2008; a GT->pred pairing is kept
                while it stays valid, so IDSW counts real identity changes)
    Identity:   IDTP from one global GT-track <-> pred-track assignment,
                IDP = IDTP/#pred, IDR = IDTP/#GT, IDF1 = 2 IDTP/(#GT + #pred)
                (Ristani et al. 2016)
    Track level: MT >= 80 % tracked, ML <= 20 %, PT otherwise; FM = number of
                 times a GT track is interrupted and tracked again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from loaders import TrackSet

BIG = 1e9


@dataclass
class MatchConfig:
    match: str = "auto"  # auto | distance | iou | inbox
    gate: float = 5.0  # pixels, for distance matching
    iou_threshold: float = 0.3


def resolve_mode(gt: TrackSet, pred: TrackSet, config: MatchConfig) -> str:
    if config.match != "auto":
        return config.match
    if gt.has_boxes and pred.has_boxes:
        return "iou"
    if gt.has_boxes:
        return "inbox"
    return "distance"


def pair_cost(
    gt: TrackSet, gi: np.ndarray, pred: TrackSet, pi: np.ndarray, mode: str, config: MatchConfig
) -> np.ndarray:
    """Cost matrix (lower is better); invalid pairs are ``BIG``."""
    gx, gy = gt.x[gi][:, None], gt.y[gi][:, None]
    px, py = pred.x[pi][None, :], pred.y[pi][None, :]
    dist = np.hypot(gx - px, gy - py)
    if mode == "distance":
        return np.where(dist <= config.gate, dist, BIG)
    gw, gh = gt.w[gi][:, None], gt.h[gi][:, None]
    if mode == "inbox":
        inside = (np.abs(gx - px) <= gw / 2) & (np.abs(gy - py) <= gh / 2)
        return np.where(inside, dist / np.hypot(gw, gh), BIG)
    if mode == "iou":
        pw, ph = pred.w[pi][None, :], pred.h[pi][None, :]
        ix = np.maximum(0, np.minimum(gx + gw / 2, px + pw / 2) - np.maximum(gx - gw / 2, px - pw / 2))
        iy = np.maximum(0, np.minimum(gy + gh / 2, py + ph / 2) - np.maximum(gy - gh / 2, py - ph / 2))
        inter = ix * iy
        union = gw * gh + pw * ph - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        return np.where(iou >= config.iou_threshold, 1.0 - iou, BIG)
    raise ValueError(f"unknown match mode {mode!r}")


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < BIG]


@dataclass
class FrameLog:
    frame: int
    gt: int
    pred: int
    tp: int
    fp: int
    fn: int
    idsw: int


@dataclass
class Result:
    name: str
    mode: str
    metrics: dict = field(default_factory=dict)
    frames: list[FrameLog] = field(default_factory=list)


def evaluate(
    gt: TrackSet, pred: TrackSet, config: MatchConfig | None = None, name: str | None = None
) -> Result:
    """Score ``pred`` against ``gt`` over the GT frame range."""
    config = config or MatchConfig()
    first, last = gt.frame_range
    pred = pred.restrict_frames(first, last)
    mode = resolve_mode(gt, pred, config)
    gt_frames, pred_frames = gt.by_frame(), pred.by_frame()
    empty = np.array([], dtype=int)

    n_gt = n_pred = tp = fp = fn = idsw = 0
    quality: list[float] = []
    err_x: list[float] = []
    err_y: list[float] = []
    err_d: list[float] = []
    last_match: dict[int, int] = {}  # gt id -> pred id (CLEAR-MOT memory)
    pair_frames: dict[tuple[int, int], int] = {}  # (gt id, pred id) -> co-valid frames
    gt_len: dict[int, int] = {}
    gt_tracked: dict[int, list[bool]] = {}
    frame_logs: list[FrameLog] = []
    det_tp = det_fp = det_fn = 0

    for frame in range(first, last + 1):
        gi = gt_frames.get(frame, empty)
        pi = pred_frames.get(frame, empty)
        n_gt += len(gi)
        n_pred += len(pi)
        cost = pair_cost(gt, gi, pred, pi, mode, config) if len(gi) and len(pi) else np.empty((len(gi), len(pi)))
        gt_ids = gt.track_id[gi]
        pred_ids = pred.track_id[pi]
        for g_id in gt_ids:
            gt_len[int(g_id)] = gt_len.get(int(g_id), 0) + 1

        # --- detection-only matching (no identity memory)
        det_pairs = _assign(cost)
        det_tp += len(det_pairs)
        det_fp += len(pi) - len(det_pairs)
        det_fn += len(gi) - len(det_pairs)
        for a, b in det_pairs:
            err_x.append(float(pred.x[pi[b]] - gt.x[gi[a]]))
            err_y.append(float(pred.y[pi[b]] - gt.y[gi[a]]))
            err_d.append(float(np.hypot(err_x[-1], err_y[-1])))

        # --- IDF1 pair statistics: every co-valid (gt, pred) pair in this frame
        if cost.size:
            for a, b in zip(*np.nonzero(cost < BIG)):
                key = (int(gt_ids[a]), int(pred_ids[b]))
                pair_frames[key] = pair_frames.get(key, 0) + 1

        # --- CLEAR-MOT matching: keep previous pairings that are still valid
        pairs: list[tuple[int, int]] = []
        used_g, used_p = set(), set()
        pred_pos = {int(pid): b for b, pid in enumerate(pred_ids)}
        for a, g_id in enumerate(gt_ids):
            prev = last_match.get(int(g_id))
            if (prev is not None and prev in pred_pos and pred_pos[prev] not in used_p
                    and cost[a, pred_pos[prev]] < BIG):
                pairs.append((a, pred_pos[prev]))
                used_g.add(a)
                used_p.add(pred_pos[prev])
        rest_g = [a for a in range(len(gi)) if a not in used_g]
        rest_p = [b for b in range(len(pi)) if b not in used_p]
        if rest_g and rest_p:
            sub = cost[np.ix_(rest_g, rest_p)]
            pairs += [(rest_g[r], rest_p[c]) for r, c in _assign(sub)]
        frame_idsw = 0
        matched_g = set()
        for a, b in pairs:
            g_id, p_id = int(gt_ids[a]), int(pred_ids[b])
            if g_id in last_match and last_match[g_id] != p_id:
                frame_idsw += 1
            last_match[g_id] = p_id
            matched_g.add(g_id)
            c = cost[a, b]
            quality.append(1.0 - c if mode == "iou" else float(np.hypot(
                pred.x[pi[b]] - gt.x[gi[a]], pred.y[pi[b]] - gt.y[gi[a]])))
        for g_id in gt_ids:
            gt_tracked.setdefault(int(g_id), []).append(int(g_id) in matched_g)
        tp += len(pairs)
        fp += len(pi) - len(pairs)
        fn += len(gi) - len(pairs)
        idsw += frame_idsw
        frame_logs.append(
            FrameLog(frame, len(gi), len(pi), len(pairs), len(pi) - len(pairs),
                     len(gi) - len(pairs), frame_idsw)
        )

    metrics = _summarise(
        gt, pred, mode, config, n_gt, n_pred, tp, fp, fn, idsw, quality,
        (det_tp, det_fp, det_fn), (err_x, err_y, err_d), pair_frames, gt_len, gt_tracked,
    )
    return Result(name or pred.name, mode, metrics, frame_logs)


def _safe(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _summarise(gt, pred, mode, config, n_gt, n_pred, tp, fp, fn, idsw, quality,
               det, errors, pair_frames, gt_len, gt_tracked) -> dict:
    det_tp, det_fp, det_fn = det
    precision = _safe(det_tp, det_tp + det_fp)
    recall = _safe(det_tp, det_tp + det_fn)
    f1 = _safe(2 * precision * recall, precision + recall)

    # identity measures: global one-to-one assignment of GT tracks to pred tracks
    g_ids = sorted(gt_len)
    p_ids = sorted({int(i) for i in np.unique(pred.track_id)})
    idtp = 0
    if g_ids and p_ids and pair_frames:
        g_pos = {g: i for i, g in enumerate(g_ids)}
        p_pos = {p: i for i, p in enumerate(p_ids)}
        overlap = np.zeros((len(g_ids), len(p_ids)))
        for (g, p), count in pair_frames.items():
            overlap[g_pos[g], p_pos[p]] = count
        rows, cols = linear_sum_assignment(-overlap)
        idtp = int(overlap[rows, cols].sum())
    idp, idr = _safe(idtp, n_pred), _safe(idtp, n_gt)
    idf1 = _safe(2 * idtp, n_gt + n_pred)

    mostly_tracked = mostly_lost = partial = fragments = 0
    for g_id, flags in gt_tracked.items():
        ratio = sum(flags) / len(flags)
        if ratio >= 0.8:
            mostly_tracked += 1
        elif ratio <= 0.2:
            mostly_lost += 1
        else:
            partial += 1
        fragments += count_fragments(flags)
    err_x, err_y, err_d = (np.array(e) for e in errors)

    pred_lengths = (
        np.bincount(np.unique(pred.track_id, return_inverse=True)[1]) if len(pred) else np.array([])
    )
    metrics = {
        "match_mode": mode,
        "gate_px": config.gate if mode == "distance" else None,
        "iou_threshold": config.iou_threshold if mode == "iou" else None,
        "gt_objects": n_gt,
        "pred_objects": n_pred,
        "gt_tracks": len(g_ids),
        "pred_tracks": len(p_ids),
        # detection
        "TP": det_tp, "FP": det_fp, "FN": det_fn,
        "precision": precision, "recall": recall, "F1": f1,
        # localisation (matched detections)
        "RMSE_px": float(np.sqrt(np.mean(err_d ** 2))) if err_d.size else None,
        "mean_error_px": float(err_d.mean()) if err_d.size else None,
        "bias_x_px": float(err_x.mean()) if err_x.size else None,
        "bias_y_px": float(err_y.mean()) if err_y.size else None,
        # CLEAR-MOT
        "MOTA": 1.0 - _safe(fn + fp + idsw, n_gt) if n_gt else None,
        "MOTP": float(np.mean(quality)) if quality else None,
        "MOTP_unit": "IoU" if mode == "iou" else "px",
        "IDSW": idsw,
        "MOT_FP": fp, "MOT_FN": fn, "MOT_TP": tp,
        # identity
        "IDTP": idtp, "IDP": idp, "IDR": idr, "IDF1": idf1,
        # track level
        "MT": mostly_tracked, "PT": partial, "ML": mostly_lost, "FM": fragments,
        "track_count_ratio": _safe(len(p_ids), len(g_ids)),
        "mean_pred_track_length": float(pred_lengths.mean()) if pred_lengths.size else 0.0,
    }
    return metrics


def count_fragments(flags: list[bool]) -> int:
    """Number of tracked->lost->tracked interruptions of one GT track."""
    state, count, lost_after_track = False, 0, False
    for tracked in flags:
        if tracked:
            if lost_after_track:
                count += 1
                lost_after_track = False
            state = True
        elif state:
            lost_after_track = True
    return count
