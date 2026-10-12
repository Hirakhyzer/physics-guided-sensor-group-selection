from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import PhysicsSystem, SensorGroup


@dataclass(frozen=True)
class GroupPhysicsScore:
    group: str
    observability_rank: int
    observability_fraction: float
    coupling_score: float
    redundancy_score: float
    reliability_score: float
    cost: float
    energy: float


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    blocks = [C]
    current = C.copy()
    for _ in range(1, A.shape[0]):
        current = current @ A
        blocks.append(current)
    return np.vstack(blocks)


def group_physics_score(system: PhysicsSystem, group: SensorGroup) -> GroupPhysicsScore:
    names = group.sensors
    C = system.measurement_matrix(names)
    O = observability_matrix(system.A, C)
    rank = int(np.linalg.matrix_rank(O))
    observability_fraction = rank / system.state_dim

    state_indices = [system.sensors[name].state_index for name in names]
    coupling_values = []
    for i, left in enumerate(state_indices):
        for right in state_indices[i + 1 :]:
            coupling_values.append(abs(float(system.coupling[left, right])))
    coupling_score = float(np.mean(coupling_values)) if coupling_values else 0.0

    covered = set(state_indices)
    redundancy_score = 1.0 - (len(covered) / len(state_indices)) if state_indices else 0.0
    reliability_score = float(np.mean([system.sensors[name].reliability for name in names]))
    cost = float(sum(system.sensors[name].cost for name in names))
    energy = float(sum(system.sensors[name].energy for name in names))

    return GroupPhysicsScore(
        group=group.name,
        observability_rank=rank,
        observability_fraction=observability_fraction,
        coupling_score=coupling_score,
        redundancy_score=redundancy_score,
        reliability_score=reliability_score,
        cost=cost,
        energy=energy,
    )
