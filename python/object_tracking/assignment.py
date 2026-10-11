"""Data association algorithms used by the multi-object tracker."""

from __future__ import annotations

import numpy as np


Match = tuple[int, int]


def pairwise_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance matrix between two sets of 2-D points."""
    first = np.asarray(first, dtype=float).reshape((-1, 2))
    second = np.asarray(second, dtype=float).reshape((-1, 2))
    if first.size == 0 or second.size == 0:
        return np.empty((len(first), len(second)), dtype=float)
    return np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)


def nearest_neighbor_assignment(
    predictions: np.ndarray,
    measurements: np.ndarray,
    max_distance: float,
    *,
    pair_costs: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[list[Match], list[int], list[int]]:
    """Associate globally cheapest pairs greedily under a distance gate.

    ``pair_costs`` can add appearance/shape information while Euclidean
    centroid distance remains the hard safety gate. ``valid_mask`` allows the
    caller to reject implausible pairs, for example objects with very
    different areas.
    """
    distances = pairwise_distances(predictions, measurements)
    track_count, detection_count = distances.shape
    costs, valid = _prepare_pair_costs(
        distances, max_distance, pair_costs=pair_costs, valid_mask=valid_mask
    )
    candidates = [
        (float(costs[i, j]), i, j)
        for i in range(track_count)
        for j in range(detection_count)
        if valid[i, j]
    ]
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[Match] = []
    for _, track_index, detection_index in candidates:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        matches.append((track_index, detection_index))
        used_tracks.add(track_index)
        used_detections.add(detection_index)

    return (
        matches,
        [i for i in range(track_count) if i not in used_tracks],
        [j for j in range(detection_count) if j not in used_detections],
    )


def _hungarian_square(cost: np.ndarray) -> list[int]:
    """Solve a finite square minimum-cost assignment using Kuhn-Munkres.

    The return value maps every row to one column. This implementation uses
    the O(n^3) shortest augmenting path form of the Hungarian algorithm.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("Hungarian cost matrix must be square")
    return _hungarian_rows(cost)


def _hungarian_rows(cost: np.ndarray) -> list[int]:
    """Assign every row of a rectangular matrix to a distinct column.

    The shortest-augmenting-path implementation runs in
    ``O(rows**2 * columns)`` and requires ``rows <= columns``.  Keeping the
    matrix rectangular avoids the much larger ``(tracks + detections)^2``
    allocation previously used by the tracker.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("Hungarian cost matrix must be two-dimensional")
    if not np.all(np.isfinite(cost)):
        raise ValueError("Hungarian cost matrix must contain finite values")

    row_count, column_count = cost.shape
    if row_count > column_count:
        raise ValueError("Hungarian matrix must have at least as many columns as rows")
    if row_count == 0:
        return []
    row_potential = np.zeros(row_count + 1, dtype=float)
    col_potential = np.zeros(column_count + 1, dtype=float)
    column_row = np.zeros(column_count + 1, dtype=int)
    predecessor = np.zeros(column_count + 1, dtype=int)

    for row in range(1, row_count + 1):
        column_row[0] = row
        min_values = np.full(column_count + 1, np.inf)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0

        while True:
            used[column] = True
            current_row = column_row[column]
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    cost[current_row - 1, candidate_column - 1]
                    - row_potential[current_row]
                    - col_potential[candidate_column]
                )
                if reduced < min_values[candidate_column]:
                    min_values[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if min_values[candidate_column] < delta:
                    delta = min_values[candidate_column]
                    next_column = candidate_column

            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[column_row[candidate_column]] += delta
                    col_potential[candidate_column] -= delta
                else:
                    min_values[candidate_column] -= delta
            column = next_column
            if column_row[column] == 0:
                break

        while True:
            previous_column = predecessor[column]
            column_row[column] = column_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if column_row[column] != 0:
            assignment[column_row[column] - 1] = column - 1
    return assignment


def hungarian_assignment(
    predictions: np.ndarray,
    measurements: np.ndarray,
    max_distance: float,
    *,
    pair_costs: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[list[Match], list[int], list[int]]:
    """Find a globally optimal gated association with the Hungarian algorithm.

    Dummy rows and columns allow either side to remain unmatched. A real pair
    is accepted only when its Euclidean distance is inside ``max_distance``.
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    distances = pairwise_distances(predictions, measurements)
    track_count, detection_count = distances.shape
    if track_count == 0 or detection_count == 0:
        return [], list(range(track_count)), list(range(detection_count))

    costs, valid = _prepare_pair_costs(
        distances, max_distance, pair_costs=pair_costs, valid_mask=valid_mask
    )
    valid_costs = costs[valid]
    unmatched_cost = max(
        max_distance,
        float(np.max(valid_costs)) if valid_costs.size else max_distance,
    )
    forbidden = max(
        1e9,
        unmatched_cost * (track_count + detection_count + 1) * 1000.0,
    )
    # A track chooses a real detection or one of ``track_count`` dummy
    # columns. Real detections that no row selects are naturally unmatched,
    # so dummy rows are unnecessary.
    augmented = np.full(
        (track_count, detection_count + track_count),
        unmatched_cost,
        dtype=float,
    )
    augmented[:, :detection_count] = np.where(valid, costs, forbidden)

    assignment = _hungarian_rows(augmented)
    matches = [
        (track_index, column)
        for track_index, column in enumerate(assignment[:track_count])
        if column < detection_count and valid[track_index, column]
    ]
    matched_tracks = {track for track, _ in matches}
    matched_detections = {detection for _, detection in matches}
    return (
        matches,
        [i for i in range(track_count) if i not in matched_tracks],
        [j for j in range(detection_count) if j not in matched_detections],
    )


def _prepare_pair_costs(
    distances: np.ndarray,
    max_distance: float,
    pair_costs: np.ndarray | None,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    costs = distances if pair_costs is None else np.asarray(pair_costs, dtype=float)
    if costs.shape != distances.shape:
        raise ValueError("pair_costs must have the same shape as the distance matrix")
    if not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("pair_costs must contain finite non-negative values")
    valid = distances <= max_distance
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.shape != distances.shape:
            raise ValueError("valid_mask must have the same shape as the distance matrix")
        valid &= supplied_mask
    return costs, valid
