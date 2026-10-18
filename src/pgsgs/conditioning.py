from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import PhysicsSystem
from .physics import observability_matrix


@dataclass(frozen=True)
class ObservabilityConditioning:
    rank: int
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    regularized_logdet: float


def observability_conditioning(
    system: PhysicsSystem,
    sensor_names: tuple[str, ...],
    *,
    regularization: float = 1e-9,
) -> ObservabilityConditioning:
    """Measure numerical observability quality for a selected sensor set.

    Rank answers whether the finite-horizon observability matrix spans the state
    space. Gramian eigenvalues additionally expose weakly observed directions
    that rank alone can hide.
    """
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if not sensor_names:
        return ObservabilityConditioning(0, 0.0, 0.0, float("inf"), system.state_dim * np.log(regularization))

    unknown = set(sensor_names) - set(system.sensors)
    if unknown:
        raise ValueError(f"unknown sensors: {sorted(unknown)}")

    C = system.measurement_matrix(sensor_names)
    O = observability_matrix(system.A, C)
    rank = int(np.linalg.matrix_rank(O))
    gramian = O.T @ O
    eigenvalues = np.linalg.eigvalsh(gramian)
    minimum = max(0.0, float(eigenvalues[0]))
    maximum = max(0.0, float(eigenvalues[-1]))
    condition = float("inf") if minimum <= regularization else maximum / minimum
    sign, logdet = np.linalg.slogdet(gramian + regularization * np.eye(system.state_dim))
    return ObservabilityConditioning(
        rank=rank,
        min_eigenvalue=minimum,
        max_eigenvalue=maximum,
        condition_number=float(condition),
        regularized_logdet=float(logdet if sign > 0 else -np.inf),
    )
