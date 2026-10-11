"""A small constant-velocity Kalman filter for 2-D centroids."""

from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """Track state ``[x, y, vx, vy]`` from noisy centroid measurements."""

    def __init__(
        self,
        initial_position: tuple[float, float],
        process_variance: float = 5.0,
        measurement_variance: float = 25.0,
        initial_velocity_variance: float = 10_000.0,
    ) -> None:
        self.state = np.array(
            [initial_position[0], initial_position[1], 0.0, 0.0], dtype=float
        )
        self.covariance = np.diag(
            [measurement_variance, measurement_variance, initial_velocity_variance, initial_velocity_variance]
        ).astype(float)
        self.process_variance = float(process_variance)
        self.measurement_noise = np.eye(2, dtype=float) * float(measurement_variance)
        self.observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float
        )

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimated velocity in coordinate units per ``dt`` time unit."""
        return float(self.state[2]), float(self.state[3])

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        process_noise = self.process_variance * np.array(
            [[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2], [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]],
            dtype=float,
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        return self.position

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        measured = np.asarray(measurement, dtype=float)
        innovation = measured - self.observation @ self.state
        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T + self.measurement_noise
        )
        gain = np.linalg.solve(
            innovation_covariance.T, (self.covariance @ self.observation.T).T
        ).T
        self.state = self.state + gain @ innovation
        identity = np.eye(4, dtype=float)
        # Joseph form helps keep the covariance symmetric and positive.
        residual = identity - gain @ self.observation
        self.covariance = (
            residual @ self.covariance @ residual.T
            + gain @ self.measurement_noise @ gain.T
        )
        return self.position

