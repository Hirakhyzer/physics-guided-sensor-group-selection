"""Validated sensor and state-space definitions for synthetic CPS studies."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Mapping

import numpy as np


def finite_real(value: object, label: str) -> float:
    """Require a finite real value; booleans must not silently act as numbers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


@dataclass(frozen=True)
class Sensor:
    name: str
    state_index: int
    cost: float = 1.0
    energy: float = 1.0
    reliability: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("sensor name must be a nonempty string")
        if isinstance(self.state_index, bool) or not isinstance(self.state_index, int) or self.state_index < 0:
            raise ValueError("state_index must be a nonnegative integer")
        for name in ("cost", "energy", "reliability"):
            finite_real(getattr(self, name), name)
        if self.cost < 0 or self.energy < 0:
            raise ValueError("cost and energy must be non-negative")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be within [0, 1]")


@dataclass(frozen=True)
class SensorGroup:
    name: str
    sensors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("sensor group name must be a nonempty string")
        if not isinstance(self.sensors, (tuple, list)) or not self.sensors:
            raise ValueError("sensor group cannot be empty")
        if any(not isinstance(name, str) or not name.strip() for name in self.sensors):
            raise ValueError("sensor group members must have nonempty names")
        if len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensor group cannot contain duplicate sensors")
        object.__setattr__(self, "sensors", tuple(self.sensors))


@dataclass(frozen=True)
class PhysicsSystem:
    A: np.ndarray
    sensors: Mapping[str, Sensor]
    coupling: np.ndarray

    def __post_init__(self) -> None:
        A = np.array(self.A, dtype=float, copy=True)
        coupling = np.array(self.coupling, dtype=float, copy=True)
        if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a nonempty square state-transition matrix")
        if coupling.shape != A.shape:
            raise ValueError("coupling matrix must match A shape")
        if not np.isfinite(A).all() or not np.isfinite(coupling).all():
            raise ValueError("A and coupling matrix coefficients must be finite")
        if not isinstance(self.sensors, Mapping):
            raise ValueError("sensors must be a mapping of names to Sensor objects")
        for name, sensor in self.sensors.items():
            if not isinstance(sensor, Sensor) or name != sensor.name:
                raise ValueError("sensor mapping keys must match Sensor names")
            if sensor.state_index >= A.shape[0]:
                raise ValueError("sensor state index outside system dimension")
        A.flags.writeable = False
        coupling.flags.writeable = False
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "coupling", coupling)
        object.__setattr__(self, "sensors", dict(self.sensors))

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    def measurement_matrix(self, sensor_names: tuple[str, ...]) -> np.ndarray:
        C = np.zeros((len(sensor_names), self.state_dim), dtype=float)
        for row, name in enumerate(sensor_names):
            sensor = self.sensors[name]
            C[row, sensor.state_index] = 1.0
        return C
