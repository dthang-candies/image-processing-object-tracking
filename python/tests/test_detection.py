import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from object_tracking.detection import DetectorConfig, ForegroundDetector


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
