from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .model import PhysicsSystem
from .physics import observability_matrix


@dataclass(frozen=True)
class DropoutCase:
    failed_sensors: tuple[str, ...]
    remaining_sensors: tuple[str, ...]
    observability_fraction: float


@dataclass(frozen=True)
class DropoutReport:
    baseline_observability: float
    worst_case_observability: float
    mean_observability: float
    cases: tuple[DropoutCase, ...]


def _observability_fraction(system: PhysicsSystem, sensors: tuple[str, ...]) -> float:
    if not sensors:
        return 0.0
    C = system.measurement_matrix(sensors)
    O = observability_matrix(system.A, C)
    import numpy as np
    return int(np.linalg.matrix_rank(O)) / system.state_dim


def evaluate_sensor_dropout(
    system: PhysicsSystem,
    selected_sensors: tuple[str, ...],
    *,
    failures: int = 1,
) -> DropoutReport:
    if failures < 1:
        raise ValueError("failures must be positive")
    if failures > len(selected_sensors):
        raise ValueError("failures cannot exceed selected sensor count")

    baseline = _observability_fraction(system, selected_sensors)
    cases = []
    for failed in combinations(selected_sensors, failures):
        failed_set = set(failed)
        remaining = tuple(name for name in selected_sensors if name not in failed_set)
        cases.append(
            DropoutCase(
                failed_sensors=tuple(failed),
                remaining_sensors=remaining,
                observability_fraction=_observability_fraction(system, remaining),
            )
        )

    values = [case.observability_fraction for case in cases]
    return DropoutReport(
        baseline_observability=baseline,
        worst_case_observability=min(values),
        mean_observability=sum(values) / len(values),
        cases=tuple(cases),
    )
