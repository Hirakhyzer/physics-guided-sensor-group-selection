from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import PhysicsSystem


@dataclass(frozen=True)
class EstimatorConfig:
    process_variance: float = 1e-3
    measurement_variance: float = 1e-2
    initial_variance: float = 1.0

    def __post_init__(self) -> None:
        if self.process_variance < 0 or self.measurement_variance <= 0 or self.initial_variance < 0:
            raise ValueError("estimator variances must be non-negative and measurement variance positive")


@dataclass(frozen=True)
class EstimationResult:
    estimates: np.ndarray
    covariances: np.ndarray
    rmse: float
    final_trace: float


def kalman_estimate(
    system: PhysicsSystem,
    sensor_names: tuple[str, ...],
    measurements: np.ndarray,
    *,
    truth: np.ndarray | None = None,
    config: EstimatorConfig = EstimatorConfig(),
    initial_state: np.ndarray | None = None,
) -> EstimationResult:
    """Estimate system state using only the selected sensor subset.

    Missing measurements may be represented by NaN. At each step the update is
    built from the currently available rows, allowing the same estimator to be
    used for dropout/fault experiments.
    """
    if not sensor_names:
        raise ValueError("at least one sensor is required for estimation")
    unknown = set(sensor_names) - set(system.sensors)
    if unknown:
        raise ValueError(f"unknown sensors: {sorted(unknown)}")

    y = np.asarray(measurements, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(sensor_names):
        raise ValueError("measurements must have shape [steps, selected sensors]")

    n = system.state_dim
    x = np.zeros(n, dtype=float) if initial_state is None else np.asarray(initial_state, dtype=float).copy()
    if x.shape != (n,):
        raise ValueError("initial_state has wrong dimension")

    P = np.eye(n) * config.initial_variance
    Q = np.eye(n) * config.process_variance
    full_C = system.measurement_matrix(sensor_names)
    identity = np.eye(n)
    estimates = []
    covariances = []

    for row in y:
        x_pred = system.A @ x
        P_pred = system.A @ P @ system.A.T + Q
        available = np.isfinite(row)

        if np.any(available):
            C = full_C[available]
            z = row[available]
            R = np.eye(len(z)) * config.measurement_variance
            innovation = z - C @ x_pred
            S = C @ P_pred @ C.T + R
            K = P_pred @ C.T @ np.linalg.pinv(S)
            x = x_pred + K @ innovation
            # Joseph form is more numerically stable than (I-KC)P.
            correction = identity - K @ C
            P = correction @ P_pred @ correction.T + K @ R @ K.T
        else:
            x, P = x_pred, P_pred

        estimates.append(x.copy())
        covariances.append(P.copy())

    estimate_array = np.asarray(estimates)
    covariance_array = np.asarray(covariances)
    rmse = float("nan")
    if truth is not None:
        truth_array = np.asarray(truth, dtype=float)
        if truth_array.shape != estimate_array.shape:
            raise ValueError("truth must match estimated state trajectory shape")
        rmse = float(np.sqrt(np.mean((estimate_array - truth_array) ** 2)))

    return EstimationResult(
        estimates=estimate_array,
        covariances=covariance_array,
        rmse=rmse,
        final_trace=float(np.trace(covariance_array[-1])) if len(covariance_array) else float("nan"),
    )
