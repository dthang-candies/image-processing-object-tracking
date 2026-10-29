import pytest

from object_tracking.models import Detection
from object_tracking.tracker import MultiObjectTracker


def detection(x: float, y: float = 20.0, appearance=None) -> Detection:
    return Detection(
        (int(x - 5), int(y - 5), 10, 10),
        (x, y),
        100.0,
        appearance=appearance,
    )


def test_tracker_keeps_id_across_one_missing_detection():
    tracker = MultiObjectTracker(
        association="hungarian", max_distance=20.0, max_missed=2, min_hits=1
    )
    first = tracker.update([detection(10)], frame_index=0)
    second = tracker.update([detection(15)], frame_index=1)
    missing = tracker.update([], frame_index=2)
    recovered = tracker.update([detection(25)], frame_index=3)

    assert first[0].track_id == 1
    assert second[0].track_id == 1
    assert missing[0].track_id == 1 and missing[0].missed == 1
    assert recovered[0].track_id == 1 and recovered[0].missed == 0
    assert [point.matched for point in tracker.trajectories[1]] == [True, True, False, True]
    assert recovered[0].trail == (recovered[0].centroid,)


def test_tracker_finishes_track_after_missed_limit():
    tracker = MultiObjectTracker(max_missed=1, min_hits=1)
    tracker.update([detection(10)], frame_index=0)
    tracker.update([], frame_index=1)
    assert len(tracker.snapshots()) == 1
    tracker.update([], frame_index=2)
    assert tracker.snapshots() == []
    assert 1 in tracker.trajectories


def test_unconfirmed_one_frame_noise_is_not_exported():
    tracker = MultiObjectTracker(min_hits=2, max_missed=10)
    tracker.update([detection(10)], frame_index=0)
    tracker.update([], frame_index=1)

    assert tracker.snapshots() == []
    assert tracker.trajectories == {}


def test_area_gate_prevents_implausible_match():
    tracker = MultiObjectTracker(
        min_hits=1, max_distance=30, max_area_ratio=2.0
    )
    tracker.update([detection(10)], frame_index=0)
    much_larger = Detection((5, 5, 50, 50), (12.0, 20.0), 2500.0)
    snapshots = tracker.update([much_larger], frame_index=1)

    assert {snapshot.track_id for snapshot in snapshots} == {1, 2}
    assert next(item for item in snapshots if item.track_id == 1).missed == 1


def test_appearance_keeps_identity_when_objects_swap_positions():
    red = [1.0, 0.0]
    blue = [0.0, 1.0]
    tracker = MultiObjectTracker(
        min_hits=1,
        max_distance=20,
        iou_weight=0,
        area_weight=0,
        appearance_weight=2.0,
    )
    tracker.update(
        [detection(0, appearance=red), detection(10, appearance=blue)],
        frame_index=0,
    )
    snapshots = tracker.update(
        [detection(1, appearance=blue), detection(9, appearance=red)],
        frame_index=1,
    )

    track_one = next(snapshot for snapshot in snapshots if snapshot.track_id == 1)
    track_two = next(snapshot for snapshot in snapshots if snapshot.track_id == 2)
    assert track_one.centroid[0] > track_two.centroid[0]


def test_missing_track_does_not_reacquire_far_away_detection():
    tracker = MultiObjectTracker(
        max_distance=10, max_missed=10, min_hits=1,
        measurement_variance=10_000, mahalanobis_gate=1_000_000,
    )
    tracker.update([detection(10)], frame_index=0)
    for frame in range(1, 4):
        tracker.update([], frame_index=frame)

    snapshots = tracker.update([detection(26)], frame_index=4)

    assert {snapshot.track_id for snapshot in snapshots} == {1, 2}
    assert next(snapshot for snapshot in snapshots if snapshot.track_id == 1).missed == 4


@pytest.mark.parametrize("association", ["nearest", "hungarian"])
def test_adaptive_confirmation_requires_small_fragments_to_persist(association):
    tracker = MultiObjectTracker(
        association=association, max_distance=20, min_hits=3,
        adaptive_confirmation=True,
    )
    small = [detection(index * 50) for index in range(12)]
    large = [
        Detection((1000 + index * 150, 0, 100, 100), (1050 + index * 150, 50), 10_000)
        for index in range(3)
    ]
    for frame in range(3):
        snapshots = tracker.update(small + large, frame_index=frame)

    assert sum(snapshot.confirmed for snapshot in snapshots) == 3
    for frame in range(3, 5):
        snapshots = tracker.update(small + large, frame_index=frame)
    assert sum(snapshot.confirmed for snapshot in snapshots) == 15


@pytest.mark.parametrize("association", ["nearest", "hungarian"])
def test_adaptive_confirmation_preserves_small_object_scene(association):
    tracker = MultiObjectTracker(
        association=association, max_distance=10, min_hits=3,
        adaptive_confirmation=True,
    )
    small = [
        Detection((index * 20, 10, 3, 3), (index * 20 + 1.5, 11.5), 4)
        for index in range(12)
    ]
    for frame in range(3):
        snapshots = tracker.update(small, frame_index=frame)

    assert sum(snapshot.confirmed for snapshot in snapshots) == 12


@pytest.mark.parametrize("association", ["nearest", "hungarian"])
def test_motion_gate_rejects_stationary_tracks_and_keeps_stopped_movers(association):
    tracker = MultiObjectTracker(
        association=association, max_distance=20, min_hits=2, min_motion_distance=2.0,
    )
    for frame in range(8):
        snapshots = tracker.update([detection(10)], frame_index=frame)

    assert not snapshots[0].confirmed
    assert tracker.trajectories == {}

    for frame in range(8, 14):
        snapshots = tracker.update([detection(10 + (frame - 7) * 2)], frame_index=frame)
    assert snapshots[0].confirmed
    assert list(tracker.trajectories) == [1]

    for frame in range(14, 19):
        snapshots = tracker.update([detection(22)], frame_index=frame)
    assert snapshots[0].confirmed


@pytest.mark.parametrize("association", ["nearest", "hungarian"])
def test_adaptive_process_noise_keeps_accelerating_object_id(association):
    tracker = MultiObjectTracker(
        association=association, max_distance=60, min_hits=1,
        adaptive_process_noise=True,
    )
    for frame in range(40):
        x = 10 + 3 * frame + 0.4 * frame * frame
        snapshots = tracker.update(
            [detection(x)], frame_index=frame, time_seconds=frame / 25, dt=1 / 25
        )
        observed = [snapshot for snapshot in snapshots if snapshot.missed == 0]
        assert len(observed) == 1
        assert observed[0].track_id == 1

