import cv2
import numpy as np

from object_tracking.models import Detection, TrackSnapshot
from object_tracking.visualization import annotate_frame


def snapshot(track_id, bbox, *, missed=0, confirmed=True):
    x, y, width, height = bbox
    return TrackSnapshot(
        track_id=track_id,
        centroid=(x + width / 2, y + height / 2),
        bbox=bbox,
        age=5,
        hits=3,
        missed=missed,
        confirmed=confirmed,
        trail=(),
        velocity=(0.0, 0.0),
    )


def test_draws_only_confirmed_current_tracks_and_detections(monkeypatch):
    frame = np.zeros((200, 420, 3), dtype=np.uint8)
    detections = [
        Detection((20, 100, 10, 10), (25.0, 105.0), 100.0),
        Detection((300, 100, 10, 10), (305.0, 105.0), 100.0),
    ]
    tracks = [
        snapshot(1, (20, 100, 10, 10)),
        snapshot(2, (190, 100, 10, 10), missed=2),
        snapshot(3, (300, 100, 10, 10), confirmed=False),
    ]
    labels = []
    put_text = cv2.putText

    def record_text(image, text, *args, **kwargs):
        labels.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    image = annotate_frame(frame, detections, tracks, "hungarian")

    assert "ID 1" in labels
    assert "ID 2" not in labels and "ID 3" not in labels
    assert "hungarian | visible tracks: 1" in labels
    assert image[100, 20].any()
    assert not image[100, 190].any()
    assert not image[105, 305].any()


def test_overlapping_id_labels_are_not_all_drawn(monkeypatch):
    frame = np.zeros((200, 420, 3), dtype=np.uint8)
    tracks = [snapshot(1, (20, 100, 10, 10)), snapshot(2, (25, 100, 10, 10))]
    labels = []
    put_text = cv2.putText

    def record_text(image, text, *args, **kwargs):
        labels.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    annotate_frame(frame, [], tracks, "hungarian")

    assert sum(label.startswith("ID ") for label in labels) == 1
    assert "hungarian | visible tracks: 2" in labels


def test_dense_scene_limits_number_of_labels(monkeypatch):
    frame = np.zeros((350, 350, 3), dtype=np.uint8)
    tracks = [
        snapshot(index + 1, (10 + (index % 10) * 33, 40 + (index // 10) * 35, 3, 3))
        for index in range(80)
    ]
    labels = []
    put_text = cv2.putText

    def record_text(image, text, *args, **kwargs):
        labels.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    annotate_frame(frame, [], tracks, "hungarian")

    assert 0 < sum(label.startswith("ID ") for label in labels) <= 30
    assert "hungarian | visible tracks: 80" in labels
