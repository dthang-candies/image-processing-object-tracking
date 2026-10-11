import numpy as np

from object_tracking.kalman import KalmanFilter2D


def test_kalman_learns_constant_velocity_and_predicts_through_gap():
    kalman = KalmanFilter2D((0.0, 0.0), process_variance=0.1, measurement_variance=1.0)
    for x in range(1, 8):
        kalman.predict(dt=1.0)
        kalman.update((float(x), 0.0))

    predicted_x, predicted_y = kalman.predict(dt=1.0)
    assert np.isclose(predicted_x, 8.0, atol=0.35)
    assert np.isclose(predicted_y, 0.0, atol=0.1)


def test_kalman_rejects_non_positive_timestep():
    kalman = KalmanFilter2D((0.0, 0.0))
    try:
        kalman.predict(0.0)
    except ValueError as error:
        assert "dt" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_kalman_learns_pixel_per_second_velocity_with_video_timestep():
    kalman = KalmanFilter2D((0.0, 0.0), process_variance=5.0, measurement_variance=4.0)
    fps = 20.0
    for frame in range(1, 15):
        kalman.predict(dt=1.0 / fps)
        kalman.update((frame * 3.0, 0.0))

    assert np.isclose(kalman.velocity[0], 60.0, atol=3.0)

