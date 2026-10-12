from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import PhysicsSystem, SensorGroup
from .physics import observability_matrix


@dataclass(frozen=True)
class BaselineSelection:
    method: str
    sensors: tuple[str, ...]
    total_cost: float
    total_energy: float
    observability_fraction: float


def _observability_fraction(system: PhysicsSystem, names: tuple[str, ...]) -> float:
    if not names:
        return 0.0
    C = system.measurement_matrix(names)
    return float(np.linalg.matrix_rank(observability_matrix(system.A, C)) / system.state_dim)


def _pack_by_order(
    system: PhysicsSystem,
    ordered_names: list[str],
    *,
    max_cost: float,
    max_energy: float,
    method: str,
) -> BaselineSelection:
    chosen = []
    cost = energy = 0.0
    for name in ordered_names:
        sensor = system.sensors[name]
        if cost + sensor.cost <= max_cost and energy + sensor.energy <= max_energy:
            chosen.append(name)
            cost += sensor.cost
            energy += sensor.energy
    names = tuple(chosen)
    return BaselineSelection(method, names, cost, energy, _observability_fraction(system, names))


def cheapest_first(system: PhysicsSystem, *, max_cost: float, max_energy: float) -> BaselineSelection:
    order = sorted(system.sensors, key=lambda n: (system.sensors[n].cost, system.sensors[n].energy, n))
    return _pack_by_order(system, order, max_cost=max_cost, max_energy=max_energy, method="cheapest-first")


def reliability_first(system: PhysicsSystem, *, max_cost: float, max_energy: float) -> BaselineSelection:
    order = sorted(
        system.sensors,
        key=lambda n: (-system.sensors[n].reliability, system.sensors[n].cost, n),
    )
    return _pack_by_order(system, order, max_cost=max_cost, max_energy=max_energy, method="reliability-first")


def observability_greedy(system: PhysicsSystem, *, max_cost: float, max_energy: float) -> BaselineSelection:
    chosen: list[str] = []
    remaining = set(system.sensors)
    cost = energy = 0.0
    current = 0.0
    while remaining:
        candidates = []
        for name in remaining:
            sensor = system.sensors[name]
            if cost + sensor.cost > max_cost or energy + sensor.energy > max_energy:
                continue
            trial = tuple([*chosen, name])
            obs = _observability_fraction(system, trial)
            gain = obs - current
            candidates.append((gain, sensor.reliability, -sensor.cost, name, obs))
        if not candidates:
            break
        gain, _, _, name, obs = max(candidates)
        if gain <= 0.0:
            break
        sensor = system.sensors[name]
        chosen.append(name)
        remaining.remove(name)
        cost += sensor.cost
        energy += sensor.energy
        current = obs
    return BaselineSelection("observability-greedy", tuple(chosen), cost, energy, current)


def individual_group_baseline(
    system: PhysicsSystem,
    groups: list[SensorGroup],
    *,
    max_cost: float,
    max_energy: float,
) -> BaselineSelection:
    """Ignore group structure and rank sensors by membership frequency."""
    frequency = {name: 0 for name in system.sensors}
    for group in groups:
        for name in group.sensors:
            frequency[name] += 1
    order = sorted(system.sensors, key=lambda n: (-frequency[n], -system.sensors[n].reliability, n))
    return _pack_by_order(system, order, max_cost=max_cost, max_energy=max_energy, method="membership-frequency")
