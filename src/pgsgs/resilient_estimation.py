from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimation import EstimatorConfig
from .model import PhysicsSystem, finite_real


@dataclass(frozen=True)
class ResilientEstimationResult:
    estimates: np.ndarray
    rmse: float
    rejected_measurements: int
    rejection_counts: dict[str, int]
    accepted_fraction: float


def innovation_gated_estimate(
    system: PhysicsSystem,
    sensor_names: tuple[str, ...],
    measurements: np.ndarray,
    *,
    truth: np.ndarray | None = None,
    config: EstimatorConfig = EstimatorConfig(),
    gate_sigma: float = 4.0,
) -> ResilientEstimationResult:
    """Kalman estimation with per-sensor innovation consistency gating.

    The gate rejects measurements that are inconsistent with the predicted
    state and uncertainty. It is a defensive robustness mechanism for synthetic
    fault/attack studies, not a guarantee of attack detection.
    """
    if finite_real(gate_sigma, "gate_sigma") <= 0:
        raise ValueError("gate_sigma must be positive")
    if not sensor_names or len(set(sensor_names)) != len(sensor_names):
        raise ValueError("estimation sensor names must be nonempty and unique")
    unknown = set(sensor_names) - set(system.sensors)
    if unknown:
        raise ValueError(f"unknown sensors: {sorted(unknown)}")
    y = np.asarray(measurements, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(sensor_names) or y.shape[0] == 0:
        raise ValueError("measurements must have nonempty shape [steps, selected sensors]")
    if np.isinf(y).any():
        raise ValueError("measurements may use NaN for missing data but must not contain infinity")

    n = system.state_dim
    if truth is not None:
        truth_values = np.asarray(truth, dtype=float)
        if truth_values.shape != (len(y), n) or not np.isfinite(truth_values).all():
            raise ValueError("truth must be a finite trajectory matching the estimated state shape")
    C_all = system.measurement_matrix(sensor_names)
    x = np.zeros(n, dtype=float)
    P = np.eye(n) * config.initial_variance
    Q = np.eye(n) * config.process_variance
    I = np.eye(n)
    estimates = []
    rejection_counts = {name: 0 for name in sensor_names}
    rejected = 0
    finite_total = 0
    accepted_total = 0

    for row in y:
        x_pred = system.A @ x
        P_pred = system.A @ P @ system.A.T + Q
        accepted_indices = []

        for idx, value in enumerate(row):
            if not np.isfinite(value):
                continue
            finite_total += 1
            c = C_all[idx : idx + 1]
            innovation = float(value - (c @ x_pred)[0])
            variance = float((c @ P_pred @ c.T)[0, 0] + config.measurement_variance)
            normalized = abs(innovation) / max(np.sqrt(variance), 1e-12)
            if normalized <= gate_sigma:
                accepted_indices.append(idx)
                accepted_total += 1
            else:
                rejected += 1
                rejection_counts[sensor_names[idx]] += 1

        if accepted_indices:
            C = C_all[accepted_indices]
            z = row[accepted_indices]
            R = np.eye(len(accepted_indices)) * config.measurement_variance
            S = C @ P_pred @ C.T + R
            K = P_pred @ C.T @ np.linalg.pinv(S)
            x = x_pred + K @ (z - C @ x_pred)
            correction = I - K @ C
            P = correction @ P_pred @ correction.T + K @ R @ K.T
        else:
            x, P = x_pred, P_pred
        estimates.append(x.copy())

    estimate_array = np.asarray(estimates)
    rmse = float("nan")
    if truth is not None:
        truth_array = np.asarray(truth, dtype=float)
        if truth_array.shape != estimate_array.shape:
            raise ValueError("truth must match estimated state trajectory shape")
        rmse = float(np.sqrt(np.mean((estimate_array - truth_array) ** 2)))

    return ResilientEstimationResult(
        estimates=estimate_array,
        rmse=rmse,
        rejected_measurements=rejected,
        rejection_counts=rejection_counts,
        accepted_fraction=(accepted_total / finite_total) if finite_total else 0.0,
    )
