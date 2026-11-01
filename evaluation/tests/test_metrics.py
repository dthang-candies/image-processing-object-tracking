import numpy as np
import pytest

from loaders import TrackSet
from metrics import MatchConfig, count_fragments, evaluate


def line(track_id, frames, x0=0.0, y0=0.0, dx=1.0):
    return [(f, track_id, x0 + dx * f, y0) for f in frames]


def make(rows, name="t"):
    return TrackSet.from_rows(name, rows)


def test_perfect_prediction():
    gt = make(line(0, range(10)) + line(1, range(10), y0=50))
    m = evaluate(gt, make(line(7, range(10)) + line(9, range(10), y0=50))).metrics
    assert (m["precision"], m["recall"], m["F1"]) == (1, 1, 1)
    assert m["MOTA"] == 1 and m["IDF1"] == 1 and m["IDSW"] == 0 and m["MT"] == 2


def test_missing_half_of_detections():
    gt = make(line(0, range(10)))
    m = evaluate(gt, make(line(0, range(5)))).metrics
    assert m["recall"] == 0.5 and m["precision"] == 1
    assert m["MOTA"] == pytest.approx(0.5)
    assert m["PT"] == 1


def test_false_positives_lower_precision_and_mota():
    gt = make(line(0, range(10)))
    pred = make(line(0, range(10)) + line(5, range(10), y0=100))
    m = evaluate(gt, pred).metrics
    assert m["precision"] == 0.5 and m["recall"] == 1
    assert m["MOTA"] == pytest.approx(0.0)


def test_id_switch_is_counted_once():
    gt = make(line(0, range(10)))
    pred = make(line(1, range(5)) + line(2, range(5, 10), x0=0))  # new id mid-way
    m = evaluate(gt, pred).metrics
    assert m["IDSW"] == 1
    assert m["MOTA"] == pytest.approx(1 - 1 / 10)
    assert m["IDF1"] == pytest.approx(0.5)  # only one of the two ids can be matched


def test_gate_rejects_far_predictions():
    gt = make(line(0, range(5)))
    pred = make([(f, 0, f + 20.0, 0.0) for f in range(5)])
    assert evaluate(gt, pred, MatchConfig(gate=5)).metrics["TP"] == 0
    assert evaluate(gt, pred, MatchConfig(gate=25)).metrics["TP"] == 5


def test_localisation_error_and_bias():
    gt = make(line(0, range(4)))
    pred = make([(f, 0, f + 2.0, 0.0) for f in range(4)])
    m = evaluate(gt, pred).metrics
    assert m["RMSE_px"] == pytest.approx(2.0) and m["bias_x_px"] == pytest.approx(2.0)


def test_iou_matching_and_motp():
    gt = make([(0, 0, 50.0, 50.0, 20.0, 40.0)])
    assert evaluate(gt, make([(0, 3, 50.0, 50.0, 20.0, 40.0)])).metrics["MOTP"] == pytest.approx(1.0)
    assert evaluate(gt, make([(0, 3, 90.0, 50.0, 20.0, 40.0)])).metrics["TP"] == 0


def test_point_prediction_inside_gt_box():
    gt = make([(0, 0, 50.0, 50.0, 20.0, 40.0)])
    assert evaluate(gt, make([(0, 1, 55.0, 60.0)])).metrics["match_mode"] == "inbox"
    assert evaluate(gt, make([(0, 1, 55.0, 60.0)])).metrics["TP"] == 1
    assert evaluate(gt, make([(0, 1, 80.0, 50.0)])).metrics["TP"] == 0


def test_one_prediction_cannot_serve_two_gt_objects():
    gt = make(line(0, range(5)) + line(1, range(5), y0=1))
    m = evaluate(gt, make(line(0, range(5)))).metrics
    assert m["MOT_FP"] >= 0 and m["TP"] == 5


def test_fragmentation_count():
    assert count_fragments([True, True, False, True, False, False, True]) == 2
    assert count_fragments([False, True, True]) == 0
    assert count_fragments([True, False, False]) == 0
