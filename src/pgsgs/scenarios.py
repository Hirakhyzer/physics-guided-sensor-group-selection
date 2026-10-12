from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .model import PhysicsSystem


class DisturbanceKind(str, Enum):
    DROPOUT = "dropout"
    BIAS = "bias"
    DRIFT = "drift"
    FREEZE = "freeze"


@dataclass(frozen=True)
class SensorDisturbance:
    sensor: str
    kind: DisturbanceKind
    start: int
    end: int
    magnitude: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("disturbance requires 0 <= start < end")


@dataclass(frozen=True)
class SimulationTrace:
    truth: np.ndarray
    measurements: np.ndarray
    sensor_names: tuple[str, ...]


def simulate_measurements(
    system: PhysicsSystem,
    sensor_names: tuple[str, ...],
    *,
    steps: int = 100,
    seed: int = 0,
    process_std: float = 0.02,
    measurement_std: float = 0.05,
    initial_state: np.ndarray | None = None,
    disturbances: tuple[SensorDisturbance, ...] = (),
) -> SimulationTrace:
    if steps < 1:
        raise ValueError("steps must be positive")
    if process_std < 0 or measurement_std < 0:
        raise ValueError("noise standard deviations must be non-negative")
    unknown = set(sensor_names) - set(system.sensors)
    if unknown:
        raise ValueError(f"unknown sensors: {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    n = system.state_dim
    x = np.linspace(0.2, 0.8, n) if initial_state is None else np.asarray(initial_state, dtype=float).copy()
    if x.shape != (n,):
        raise ValueError("initial_state has wrong dimension")

    C = system.measurement_matrix(sensor_names)
    truth = np.zeros((steps, n), dtype=float)
    measurements = np.zeros((steps, len(sensor_names)), dtype=float)

    for t in range(steps):
        x = system.A @ x + rng.normal(0.0, process_std, size=n)
        truth[t] = x
        measurements[t] = C @ x + rng.normal(0.0, measurement_std, size=len(sensor_names))

    index = {name: i for i, name in enumerate(sensor_names)}
    for disturbance in disturbances:
        if disturbance.sensor not in index:
            continue
        column = index[disturbance.sensor]
        start = min(disturbance.start, steps)
        end = min(disturbance.end, steps)
        if start >= end:
            continue
        if disturbance.kind == DisturbanceKind.DROPOUT:
            measurements[start:end, column] = np.nan
        elif disturbance.kind == DisturbanceKind.BIAS:
            measurements[start:end, column] += disturbance.magnitude
        elif disturbance.kind == DisturbanceKind.DRIFT:
            measurements[start:end, column] += np.linspace(0.0, disturbance.magnitude, end - start)
        elif disturbance.kind == DisturbanceKind.FREEZE:
            reference = measurements[max(0, start - 1), column]
            measurements[start:end, column] = reference

    return SimulationTrace(truth=truth, measurements=measurements, sensor_names=sensor_names)
