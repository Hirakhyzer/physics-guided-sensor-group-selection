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
    marginal_trace: tuple[tuple[str, float], ...] = ()


def group_utility(score: GroupPhysicsScore, weights: SelectionWeights) -> float:
    benefit = (
        weights.observability * score.observability_fraction
        + weights.coupling * score.coupling_score
        + weights.reliability * score.reliability_score
        + weights.redundancy * score.redundancy_score
    )
    penalty = weights.cost * score.cost + weights.energy * score.energy
    return benefit - penalty


def _sensor_set_score(
    system: PhysicsSystem,
    sensor_names: tuple[str, ...],
    weights: SelectionWeights,
) -> float:
    if not sensor_names:
        return 0.0
    synthetic = SensorGroup("selected-union", sensor_names)
    return group_utility(group_physics_score(system, synthetic), weights)


def select_groups(
    system: PhysicsSystem,
    groups: list[SensorGroup],
    *,
    max_cost: float,
    max_energy: float,
    max_groups: int | None = None,
    weights: SelectionWeights = SelectionWeights(),
    require_positive_gain: bool = True,
) -> SelectionResult:
    """Greedily select groups by *marginal* utility of their sensor union.

    The original prototype ranked each group independently. That can overvalue
    overlapping groups because already-selected sensors repeatedly contribute
    observability/coupling benefit. This implementation recomputes the utility
    of the union at every step, while cost and energy are charged only once per
    physical sensor.
    """
    if max_cost < 0 or max_energy < 0:
        raise ValueError("budgets must be non-negative")

    for group in groups:
        unknown = set(group.sensors) - set(system.sensors)
        if unknown:
            raise ValueError(f"group {group.name!r} references unknown sensors: {sorted(unknown)}")

    static_scores = tuple(
        sorted(
            (
                (group.name, group_utility(group_physics_score(system, group), weights))
                for group in groups
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    remaining = list(groups)
    selected_groups: list[str] = []
    selected_sensors: list[str] = []
    marginal_trace: list[tuple[str, float]] = []
    used_cost = 0.0
    used_energy = 0.0
    current_utility = 0.0

    while remaining and (max_groups is None or len(selected_groups) < max_groups):
        candidates = []
        for group in remaining:
            new_names = [name for name in group.sensors if name not in selected_sensors]
            incremental_cost = sum(system.sensors[name].cost for name in new_names)
            incremental_energy = sum(system.sensors[name].energy for name in new_names)
            if used_cost + incremental_cost > max_cost or used_energy + incremental_energy > max_energy:
                continue

            union = tuple(dict.fromkeys([*selected_sensors, *group.sensors]))
            union_utility = _sensor_set_score(system, union, weights)
            marginal_gain = union_utility - current_utility
            # Prefer gain, then observability of the resulting union, then lower
            # incremental resource demand for deterministic tie breaking.
            union_physics = group_physics_score(system, SensorGroup("candidate-union", union))
            candidates.append(
                (
                    marginal_gain,
                    union_physics.observability_fraction,
                    -(incremental_cost + incremental_energy),
                    group.name,
                    group,
                    new_names,
                    incremental_cost,
                    incremental_energy,
                    union_utility,
                )
            )

        if not candidates:
            break
        best = max(candidates, key=lambda item: item[:4])
        marginal_gain, _, _, _, group, new_names, inc_cost, inc_energy, union_utility = best
        if require_positive_gain and marginal_gain <= 0.0:
            break

        selected_groups.append(group.name)
        selected_sensors.extend(new_names)
        marginal_trace.append((group.name, float(marginal_gain)))
        used_cost += inc_cost
        used_energy += inc_energy
        current_utility = union_utility
        remaining.remove(group)

    return SelectionResult(
        selected_groups=tuple(selected_groups),
        selected_sensors=tuple(selected_sensors),
        total_cost=used_cost,
        total_energy=used_energy,
        utility=current_utility,
        group_scores=static_scores,
        marginal_trace=tuple(marginal_trace),
    )
