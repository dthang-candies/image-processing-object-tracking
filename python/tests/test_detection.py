import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from object_tracking.detection import (
    DetectorConfig,
    ForegroundDetector,
    _remove_nested_fragments,
)


def test_detector_segments_bright_moving_rectangle():
    detector = ForegroundDetector(
        DetectorConfig(
            history=20,
            detect_shadows=False,
            min_area=50,
            kernel_size=3,
            open_iterations=0,
            close_iterations=0,
            learning_rate=0.0,
        )
    )
    background = np.zeros((100, 140, 3), dtype=np.uint8)
    detector.apply(background)
    frame = background.copy()
    cv2.rectangle(frame, (40, 30), (60, 50), (255, 255, 255), -1)

    mask, detections = detector.apply(frame)

    assert mask[40, 50] == 255
    assert len(detections) == 1
    assert detections[0].centroid[0] == pytest.approx(50.0, abs=1.0)
    assert detections[0].centroid[1] == pytest.approx(40.0, abs=1.0)
    assert detections[0].appearance is not None
    assert detections[0].appearance.sum() == pytest.approx(1.0)


def test_detector_suppresses_abnormal_full_frame_foreground():
    detector = ForegroundDetector(
        DetectorConfig(
            detect_shadows=False,
            min_area=10,
            learning_rate=0.0,
            max_foreground_ratio=0.5,
        )
    )
    background = np.zeros((60, 80, 3), dtype=np.uint8)
    detector.apply(background)

    mask, detections = detector.apply(np.full_like(background, 255))

    assert cv2.countNonZero(mask) / mask.size > 0.5
    assert detections == []


def test_detector_can_be_forced_to_cpu():
    detector = ForegroundDetector(DetectorConfig(device="cpu"))

    assert detector.device == "cpu"
    assert detector.device_name == "CPU"


def test_detector_rejects_unknown_device():
    with pytest.raises(ValueError, match="device"):
        ForegroundDetector(DetectorConfig(device="quantum"))


def test_adaptive_filter_keeps_small_objects_when_they_dominate():
    detector = ForegroundDetector(
        DetectorConfig(
            device="cpu", detect_shadows=False, min_area=4, min_width=2,
            min_height=2, kernel_size=3, open_iterations=0, close_iterations=0,
            learning_rate=0.0, adaptive_min_area=True,
        )
    )
    background = np.zeros((400, 400, 3), dtype=np.uint8)
    detector.apply(background)
    frame = background.copy()
    for index in range(16):
        x, y = 10 + (index % 4) * 15, 10 + (index // 4) * 15
        cv2.rectangle(frame, (x, y), (x + 2, y + 2), (255, 255, 255), -1)

    _, detections = detector.apply(frame)

    assert len(detections) == 16


def test_adaptive_filter_removes_tiny_fragments_in_large_object_scene():
    detector = ForegroundDetector(
        DetectorConfig(
            device="cpu", detect_shadows=False, min_area=4, min_width=2,
            min_height=2, kernel_size=3, open_iterations=0, close_iterations=0,
            learning_rate=0.0, adaptive_min_area=True,
        )
    )
    background = np.zeros((400, 400, 3), dtype=np.uint8)
    detector.apply(background)
    frame = background.copy()
    for index in range(12):
        x, y = 10 + (index % 4) * 15, 10 + (index // 4) * 15
        cv2.rectangle(frame, (x, y), (x + 2, y + 2), (255, 255, 255), -1)
    for x in (100, 200, 300):
        cv2.rectangle(frame, (x, 100), (x + 25, 125), (255, 255, 255), -1)

    mask, detections = detector.apply(frame)

    assert len(detections) == 3
    assert mask[11, 11] == 0
    assert mask[110, 110] == 255


def test_edge_preserving_filter_keeps_faint_small_spot():
    detector = ForegroundDetector(
        DetectorConfig(
            device="cpu", segmentation_mode="bright_spots", min_area=4,
            min_width=2, min_height=2, kernel_size=3, open_iterations=0,
            close_iterations=0, spot_threshold=15, adaptive_min_area=True,
            adaptive_spot_threshold=True,
        )
    )
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (22, 22), (20, 20, 20), -1)

    _, detections = detector.apply(frame)

    assert len(detections) == 1


def test_large_scene_reconnects_nearby_fragments():
    detector = ForegroundDetector(
        DetectorConfig(
            device="cpu", detect_shadows=False, min_area=4, min_width=2,
            min_height=2, kernel_size=3, open_iterations=0, close_iterations=0,
            learning_rate=0.0, adaptive_min_area=True,
        )
    )
    background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    detector.apply(background)
    frame = background.copy()
    for index in range(12):
        x, y = 50 + (index % 4) * 100, 100 + (index // 4) * 80
        cv2.rectangle(frame, (x, y), (x + 25, y + 25), (255, 255, 255), -1)
    cv2.rectangle(frame, (79, 110), (87, 118), (255, 255, 255), -1)

    _, initial = detector.apply(frame)
    _, grouped = detector.apply(frame)

    assert len(initial) == 13
    assert len(grouped) == 12
    assert any(x <= 50 and x + width >= 88 for x, _, width, _ in (d.bbox for d in grouped))


def test_nested_fragment_filter_keeps_separate_boxes():
    contour = np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], dtype=np.int32)
    large = (contour, 1000.0, (10, 10, 40, 40))
    inside = (contour, 16.0, (20, 20, 5, 5))
    separate = (contour, 16.0, (100, 20, 5, 5))

    kept = _remove_nested_fragments([large, inside, separate])

    assert [item[2] for item in kept] == [large[2], separate[2]]


def test_nearby_tiny_objects_remain_separate():
    detector = ForegroundDetector(
        DetectorConfig(
            device="cpu", detect_shadows=False, min_area=4, min_width=2,
            min_height=2, kernel_size=3, open_iterations=0, close_iterations=0,
            learning_rate=0.0, adaptive_min_area=True,
        )
    )
    background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    detector.apply(background)
    frame = background.copy()
    for index in range(6):
        x = 10 + index * 30
        cv2.rectangle(frame, (x, 50), (x + 2, 52), (255, 255, 255), -1)
        cv2.rectangle(frame, (x + 6, 50), (x + 8, 52), (255, 255, 255), -1)

    detector.apply(frame)
    _, detections = detector.apply(frame)

    assert len(detections) == 12


def test_noise_aware_spot_threshold_keeps_real_spots():
    rng = np.random.default_rng(7)
    gray = rng.gamma(1.3, 6.0, (250, 250)).clip(0, 255).astype(np.uint8)
    spots = [(40, 40), (120, 90), (190, 190)]
    for point in spots:
        cv2.circle(gray, point, 3, 85, -1)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def detect(adaptive):
        detector = ForegroundDetector(
            DetectorConfig(
                device="cpu", segmentation_mode="bright_spots", min_area=4,
                min_width=2, min_height=2, kernel_size=3, open_iterations=0,
                close_iterations=1, adaptive_spot_threshold=adaptive,
            )
        )
        return detector.apply(frame)[1]

    before = detect(False)
    after = detect(True)
    assert len(before) > 50
    assert len(after) < 15
    assert all(
        any(np.linalg.norm(np.asarray(item.centroid) - point) < 5 for item in after)
        for point in spots
    )


def test_area_gate_resists_noise_burst_then_recovers_for_small_scene():
    detector = ForegroundDetector(
        DetectorConfig(
            device="cpu", detect_shadows=False, min_area=4, min_width=2,
            min_height=2, kernel_size=3, open_iterations=0, close_iterations=0,
            learning_rate=0.0, adaptive_min_area=True,
        )
    )
    background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    detector.apply(background)
    large = background.copy()
    for index in range(10):
        x, y = 50 + (index % 5) * 100, 100 + (index // 5) * 100
        cv2.rectangle(large, (x, y), (x + 25, y + 25), (255, 255, 255), -1)
    small = background.copy()
    for index in range(200):
        x, y = 20 + (index % 20) * 10, 500 + (index // 20) * 10
        cv2.rectangle(small, (x, y), (x + 2, y + 2), (255, 255, 255), -1)

    for _ in range(10):
        detector.apply(large)
    noisy = cv2.bitwise_or(large, small)
    for _ in range(10):
        _, detections = detector.apply(noisy)
    assert len(detections) == 10
    assert detector._minimum_area >= 20

    for _ in range(10):
        _, detections = detector.apply(small)
    assert detector._minimum_area == 4
    assert len(detections) == 200
