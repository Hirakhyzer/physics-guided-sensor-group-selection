"""Physics-guided sensor group selection research package."""

from .adaptive import ReconfigurationResult, reconfigure_selection
from .baselines import BaselineSelection, cheapest_first, observability_greedy, reliability_first
from .benchmarks import battery_pack_benchmark, benchmark_suite, coupled_process_benchmark, thermal_building_benchmark
from .conditioning import ObservabilityConditioning, observability_conditioning
from .estimation import EstimationResult, EstimatorConfig, kalman_estimate
from .evaluation import EvaluationCase, EvaluationSummary, evaluate_sensor_set
from .model import PhysicsSystem, Sensor, SensorGroup
from .physics import GroupPhysicsScore, group_physics_score, observability_matrix
from .resilient_estimation import ResilientEstimationResult, innovation_gated_estimate
from .robustness import DropoutCase, DropoutReport, evaluate_sensor_dropout
from .scenarios import DisturbanceKind, SensorDisturbance, SimulationTrace, simulate_measurements
from .selection import SelectionResult, SelectionWeights, group_utility, select_groups
from .statistics import PairedComparison, paired_bootstrap_delta

__all__ = [
    "BaselineSelection",
    "DisturbanceKind",
    "DropoutCase",
    "DropoutReport",
    "EstimationResult",
    "EstimatorConfig",
    "EvaluationCase",
    "EvaluationSummary",
    "GroupPhysicsScore",
    "ObservabilityConditioning",
    "PairedComparison",
    "PhysicsSystem",
    "ReconfigurationResult",
    "ResilientEstimationResult",
    "SelectionResult",
    "SelectionWeights",
    "Sensor",
    "SensorDisturbance",
    "SensorGroup",
    "SimulationTrace",
    "battery_pack_benchmark",
    "benchmark_suite",
    "cheapest_first",
    "coupled_process_benchmark",
    "evaluate_sensor_dropout",
    "evaluate_sensor_set",
    "group_physics_score",
    "group_utility",
    "innovation_gated_estimate",
    "kalman_estimate",
    "observability_conditioning",
    "observability_greedy",
    "observability_matrix",
    "paired_bootstrap_delta",
    "reconfigure_selection",
    "reliability_first",
    "select_groups",
    "simulate_measurements",
    "thermal_building_benchmark",
]
