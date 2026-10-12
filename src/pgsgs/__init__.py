"""Physics-guided sensor group selection research package."""

from .benchmarks import coupled_process_benchmark
from .model import PhysicsSystem, Sensor, SensorGroup
from .physics import GroupPhysicsScore, group_physics_score, observability_matrix
from .robustness import DropoutCase, DropoutReport, evaluate_sensor_dropout
from .selection import SelectionResult, SelectionWeights, group_utility, select_groups

__all__ = [
    "DropoutCase",
    "DropoutReport",
    "GroupPhysicsScore",
    "PhysicsSystem",
    "SelectionResult",
    "SelectionWeights",
    "Sensor",
    "SensorGroup",
    "coupled_process_benchmark",
    "evaluate_sensor_dropout",
    "group_physics_score",
    "group_utility",
    "observability_matrix",
    "select_groups",
]
