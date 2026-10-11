import numpy as np

from object_tracking.assignment import (
    _hungarian_rows,
    hungarian_assignment,
    nearest_neighbor_assignment,
)


def test_nearest_neighbor_respects_gate_and_unique_pairs():
    predictions = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
    measurements = np.array([[1.0, 0.0], [9.0, 0.0]])

    matches, unmatched_tracks, unmatched_detections = nearest_neighbor_assignment(
        predictions, measurements, max_distance=3.0
    )

    assert set(matches) == {(0, 0), (1, 1)}
    assert unmatched_tracks == [2]
    assert unmatched_detections == []


def test_hungarian_finds_better_global_assignment_than_greedy():
    predictions = np.array([[0.0, 0.0], [2.0, 0.0]])
    measurements = np.array([[1.1, 0.0], [3.0, 0.0]])

    nearest_matches, _, _ = nearest_neighbor_assignment(
        predictions, measurements, max_distance=2.0
    )
    hungarian_matches, unmatched_tracks, unmatched_detections = hungarian_assignment(
        predictions, measurements, max_distance=2.0
    )

    assert len(nearest_matches) == 1
    assert set(hungarian_matches) == {(0, 0), (1, 1)}
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_hungarian_handles_empty_and_out_of_gate_inputs():
    matches, unmatched_tracks, unmatched_detections = hungarian_assignment(
        np.array([[0.0, 0.0]]), np.array([[50.0, 0.0]]), max_distance=10.0
    )
    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_custom_cost_and_validity_mask_influence_assignment():
    predictions = np.array([[0.0, 0.0], [10.0, 0.0]])
    measurements = np.array([[1.0, 0.0], [9.0, 0.0]])
    custom_cost = np.array([[100.0, 1.0], [1.0, 100.0]])
    validity = np.array([[True, False], [False, True]])

    matches, _, _ = hungarian_assignment(
        predictions,
        measurements,
        max_distance=20.0,
        pair_costs=custom_cost,
        valid_mask=validity,
    )

    assert set(matches) == {(0, 0), (1, 1)}

    matches, unmatched_tracks, unmatched_detections = hungarian_assignment(
        np.empty((0, 2)), np.array([[1.0, 1.0]]), max_distance=10.0
    )
    assert matches == []
    assert unmatched_tracks == []
    assert unmatched_detections == [0]


def test_rectangular_hungarian_assigns_each_row_once():
    assignment = _hungarian_rows(
        np.array([[4.0, 1.0, 8.0, 9.0], [2.0, 5.0, 3.0, 7.0]])
    )

    assert assignment == [1, 0]
