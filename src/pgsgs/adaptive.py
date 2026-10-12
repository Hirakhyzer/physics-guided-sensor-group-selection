from __future__ import annotations

from dataclasses import dataclass, replace

from .model import PhysicsSystem, SensorGroup
from .physics import group_physics_score
from .selection import SelectionResult, SelectionWeights, select_groups


@dataclass(frozen=True)
class ReconfigurationResult:
    selection: SelectionResult
    unavailable_sensors: tuple[str, ...]
    degraded_sensors: tuple[str, ...]
    observability_fraction: float
    meets_observability_target: bool


def reconfigure_selection(
    system: PhysicsSystem,
    groups: list[SensorGroup],
    *,
    max_cost: float,
    max_energy: float,
    unavailable_sensors: set[str] | None = None,
    reliability_updates: dict[str, float] | None = None,
    minimum_observability: float = 0.75,
    weights: SelectionWeights = SelectionWeights(),
) -> ReconfigurationResult:
    """Re-select groups after sensor loss or reliability degradation.

    Groups are projected onto currently available sensors. This allows a
    physically meaningful group to degrade gracefully instead of disappearing
    entirely when one member is unavailable. The result explicitly reports
    whether the requested observability target is still achieved.
    """
    if not 0.0 <= minimum_observability <= 1.0:
        raise ValueError("minimum_observability must be within [0, 1]")
    unavailable = unavailable_sensors or set()
    unknown = unavailable - set(system.sensors)
    if unknown:
        raise ValueError(f"unknown unavailable sensors: {sorted(unknown)}")

    updates = reliability_updates or {}
    if set(updates) - set(system.sensors):
        raise ValueError("reliability update references unknown sensor")
    if any(not 0.0 <= value <= 1.0 for value in updates.values()):
        raise ValueError("updated reliability must be within [0, 1]")

    sensors = {
        name: replace(sensor, reliability=updates.get(name, sensor.reliability))
        for name, sensor in system.sensors.items()
        if name not in unavailable
    }
    reduced_system = PhysicsSystem(A=system.A, sensors=sensors, coupling=system.coupling)

    projected_groups = []
    for group in groups:
        surviving = tuple(name for name in group.sensors if name in sensors)
        if surviving:
            projected_groups.append(SensorGroup(group.name, surviving))

    selection = select_groups(
        reduced_system,
        projected_groups,
        max_cost=max_cost,
        max_energy=max_energy,
        weights=weights,
    )
    if selection.selected_sensors:
        score = group_physics_score(
            reduced_system,
            SensorGroup("reconfigured-selection", selection.selected_sensors),
        )
        observability = score.observability_fraction
    else:
        observability = 0.0

    degraded = tuple(sorted(name for name, value in updates.items() if value < system.sensors[name].reliability))
    return ReconfigurationResult(
        selection=selection,
        unavailable_sensors=tuple(sorted(unavailable)),
        degraded_sensors=degraded,
        observability_fraction=observability,
        meets_observability_target=observability >= minimum_observability,
    )
