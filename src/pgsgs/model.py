from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Sensor:
    name: str
    state_index: int
    cost: float = 1.0
    energy: float = 1.0
    reliability: float = 1.0

    def __post_init__(self) -> None:
        if self.state_index < 0:
            raise ValueError("state_index must be non-negative")
        if self.cost < 0 or self.energy < 0:
            raise ValueError("cost and energy must be non-negative")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be within [0, 1]")


@dataclass(frozen=True)
class SensorGroup:
    name: str
    sensors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.sensors:
            raise ValueError("sensor group cannot be empty")
        if len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensor group cannot contain duplicate sensors")


@dataclass(frozen=True)
class PhysicsSystem:
    A: np.ndarray
    sensors: Mapping[str, Sensor]
    coupling: np.ndarray

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=float)
        coupling = np.asarray(self.coupling, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square state-transition matrix")
        if coupling.shape != A.shape:
            raise ValueError("coupling matrix must match A shape")
        if any(sensor.state_index >= A.shape[0] for sensor in self.sensors.values()):
            raise ValueError("sensor state index outside system dimension")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "coupling", coupling)

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    def measurement_matrix(self, sensor_names: tuple[str, ...]) -> np.ndarray:
        C = np.zeros((len(sensor_names), self.state_dim), dtype=float)
        for row, name in enumerate(sensor_names):
            sensor = self.sensors[name]
            C[row, sensor.state_index] = 1.0
        return C
