from __future__ import annotations

import numpy as np

from .model import PhysicsSystem, Sensor, SensorGroup


def coupled_process_benchmark() -> tuple[PhysicsSystem, list[SensorGroup]]:
    """Return a small coupled-process benchmark with interpretable groups."""
    A = np.array(
        [
            [0.92, 0.08, 0.00, 0.00],
            [0.05, 0.90, 0.10, 0.00],
            [0.00, 0.08, 0.88, 0.10],
            [0.00, 0.00, 0.07, 0.93],
        ],
        dtype=float,
    )
    coupling = np.abs(A - np.diag(np.diag(A)))
    sensors = {
        "pressure_in": Sensor("pressure_in", 0, cost=1.0, energy=0.8, reliability=0.98),
        "flow": Sensor("flow", 1, cost=1.2, energy=1.0, reliability=0.96),
        "level": Sensor("level", 2, cost=1.0, energy=0.7, reliability=0.99),
        "pressure_out": Sensor("pressure_out", 3, cost=1.1, energy=0.9, reliability=0.97),
        "flow_backup": Sensor("flow_backup", 1, cost=0.9, energy=0.7, reliability=0.94),
    }
    groups = [
        SensorGroup("hydraulic-core", ("pressure_in", "flow", "level")),
        SensorGroup("downstream-state", ("flow", "level", "pressure_out")),
        SensorGroup("redundant-flow", ("flow", "flow_backup")),
        SensorGroup("boundary-pressure", ("pressure_in", "pressure_out")),
        SensorGroup("low-power-state", ("flow_backup", "level")),
    ]
    return PhysicsSystem(A=A, sensors=sensors, coupling=coupling), groups
