from __future__ import annotations

from dataclasses import dataclass

from .model import PhysicsSystem, SensorGroup
from .physics import GroupPhysicsScore, group_physics_score


@dataclass(frozen=True)
class SelectionWeights:
    observability: float = 0.4
    coupling: float = 0.2
    reliability: float = 0.2
    redundancy: float = 0.1
    cost: float = 0.05
    energy: float = 0.05


@dataclass(frozen=True)
class SelectionResult:
    selected_groups: tuple[str, ...]
    selected_sensors: tuple[str, ...]
    total_cost: float
    total_energy: float
    utility: float
    group_scores: tuple[tuple[str, float], ...]


def group_utility(score: GroupPhysicsScore, weights: SelectionWeights) -> float:
    benefit = (
        weights.observability * score.observability_fraction
        + weights.coupling * score.coupling_score
        + weights.reliability * score.reliability_score
        + weights.redundancy * score.redundancy_score
    )
    penalty = weights.cost * score.cost + weights.energy * score.energy
    return benefit - penalty


def select_groups(
    system: PhysicsSystem,
    groups: list[SensorGroup],
    *,
    max_cost: float,
    max_energy: float,
    max_groups: int | None = None,
    weights: SelectionWeights = SelectionWeights(),
) -> SelectionResult:
    if max_cost < 0 or max_energy < 0:
        raise ValueError("budgets must be non-negative")

    scored = []
    for group in groups:
        physics = group_physics_score(system, group)
        scored.append((group, physics, group_utility(physics, weights)))

    scored.sort(key=lambda item: (item[2], item[1].observability_fraction, item[1].reliability_score), reverse=True)

    selected_groups: list[str] = []
    selected_sensors: list[str] = []
    used_cost = 0.0
    used_energy = 0.0
    utility = 0.0

    for group, physics, score in scored:
        if max_groups is not None and len(selected_groups) >= max_groups:
            break

        new_sensor_names = [name for name in group.sensors if name not in selected_sensors]
        incremental_cost = sum(system.sensors[name].cost for name in new_sensor_names)
        incremental_energy = sum(system.sensors[name].energy for name in new_sensor_names)

        if used_cost + incremental_cost > max_cost:
            continue
        if used_energy + incremental_energy > max_energy:
            continue

        selected_groups.append(group.name)
        selected_sensors.extend(new_sensor_names)
        used_cost += incremental_cost
        used_energy += incremental_energy
        utility += score

    return SelectionResult(
        selected_groups=tuple(selected_groups),
        selected_sensors=tuple(selected_sensors),
        total_cost=used_cost,
        total_energy=used_energy,
        utility=utility,
        group_scores=tuple((group.name, score) for group, _, score in scored),
    )
