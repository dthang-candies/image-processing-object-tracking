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

