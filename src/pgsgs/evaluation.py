from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np

from .estimation import EstimatorConfig, kalman_estimate
from .model import PhysicsSystem
from .resilient_estimation import innovation_gated_estimate
from .scenarios import SensorDisturbance, SimulationTrace, simulate_measurements


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    rmse: float
    resilient_rmse: float
    rejected_measurements: int
    selected_sensor_count: int


@dataclass(frozen=True)
class EvaluationSummary:
    method: str
    sensors: tuple[str, ...]
    cases: tuple[EvaluationCase, ...]
    mean_rmse: float
    mean_resilient_rmse: float
    worst_rmse: float


def subset_trace(trace: SimulationTrace, sensor_names: tuple[str, ...]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(trace.sensor_names)}
    missing = set(sensor_names) - set(index)
    if missing:
        raise ValueError(f"trace does not contain sensors: {sorted(missing)}")
    return trace.measurements[:, [index[name] for name in sensor_names]]


def evaluate_sensor_set(
    system: PhysicsSystem,
    sensor_names: tuple[str, ...],
    scenarios: dict[str, tuple[SensorDisturbance, ...]],
    *,
    method: str,
    steps: int = 120,
    seed: int = 7,
    process_std: float = 0.02,
    measurement_std: float = 0.05,
    estimator_config: EstimatorConfig = EstimatorConfig(),
) -> EvaluationSummary:
    if not sensor_names:
        raise ValueError("cannot evaluate an empty sensor set")

    all_names = tuple(system.sensors)
    cases = []
    for scenario_name, disturbances in scenarios.items():
        full_trace = simulate_measurements(
            system,
            all_names,
            steps=steps,
            seed=seed,
            process_std=process_std,
            measurement_std=measurement_std,
            disturbances=disturbances,
        )
        measurements = subset_trace(full_trace, sensor_names)
        standard = kalman_estimate(
            system,
            sensor_names,
            measurements,
            truth=full_trace.truth,
            config=estimator_config,
        )
        resilient = innovation_gated_estimate(
            system,
            sensor_names,
            measurements,
            truth=full_trace.truth,
            config=estimator_config,
        )
        cases.append(
            EvaluationCase(
                name=scenario_name,
                rmse=standard.rmse,
                resilient_rmse=resilient.rmse,
                rejected_measurements=resilient.rejected_measurements,
                selected_sensor_count=len(sensor_names),
            )
        )

    rmse_values = [case.rmse for case in cases]
    resilient_values = [case.resilient_rmse for case in cases]
    return EvaluationSummary(
        method=method,
        sensors=sensor_names,
        cases=tuple(cases),
        mean_rmse=float(mean(rmse_values)),
        mean_resilient_rmse=float(mean(resilient_values)),
        worst_rmse=float(max(rmse_values)),
    )
