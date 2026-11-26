import numpy as np

from pgsgs.adaptive import reconfigure_selection
from pgsgs.baselines import cheapest_first, observability_greedy, reliability_first
from pgsgs.benchmarks import benchmark_suite, coupled_process_benchmark
from pgsgs.estimation import kalman_estimate
from pgsgs.evaluation import evaluate_sensor_set
from pgsgs.resilient_estimation import innovation_gated_estimate
from pgsgs.scenarios import DisturbanceKind, SensorDisturbance, simulate_measurements
from pgsgs.selection import select_groups


def test_clean_estimation_has_finite_error():
    system, _ = coupled_process_benchmark()
    names = ("pressure_in", "flow", "level")
    trace = simulate_measurements(system, names, steps=40, seed=3)
    result = kalman_estimate(system, names, trace.measurements, truth=trace.truth)
    assert np.isfinite(result.rmse)
    assert result.rmse >= 0.0
    assert result.estimates.shape == trace.truth.shape


def test_dropout_is_supported_without_imputation():
    system, _ = coupled_process_benchmark()
    names = ("pressure_in", "flow", "level")
    trace = simulate_measurements(
        system,
        names,
        steps=40,
        seed=4,
        disturbances=(SensorDisturbance("flow", DisturbanceKind.DROPOUT, 10, 25),),
    )
    assert np.isnan(trace.measurements[10:25, 1]).all()
    result = kalman_estimate(system, names, trace.measurements, truth=trace.truth)
    assert np.isfinite(result.rmse)


def test_resilient_estimator_rejects_large_bias():
    system, _ = coupled_process_benchmark()
    names = ("pressure_in", "flow", "level")
    trace = simulate_measurements(
        system,
        names,
        steps=60,
        seed=5,
        disturbances=(SensorDisturbance("flow", DisturbanceKind.BIAS, 20, 50, 5.0),),
    )
    result = innovation_gated_estimate(system, names, trace.measurements, truth=trace.truth, gate_sigma=3.0)
    assert result.rejected_measurements > 0
    assert result.rejection_counts["flow"] > 0


def test_adaptive_reconfiguration_excludes_unavailable_sensor():
    system, groups = coupled_process_benchmark()
    result = reconfigure_selection(
        system,
        groups,
        max_cost=3.2,
        max_energy=2.6,
        unavailable_sensors={"flow"},
    )
    assert "flow" not in result.selection.selected_sensors
    assert result.observability_fraction >= 0.0


def test_all_baselines_respect_resource_budgets():
    system, _ = coupled_process_benchmark()
    for baseline in (
        cheapest_first(system, max_cost=2.5, max_energy=2.0),
        reliability_first(system, max_cost=2.5, max_energy=2.0),
        observability_greedy(system, max_cost=2.5, max_energy=2.0),
    ):
        assert baseline.total_cost <= 2.5
        assert baseline.total_energy <= 2.0


def test_physics_selector_reports_marginal_trace():
    system, groups = coupled_process_benchmark()
    result = select_groups(system, groups, max_cost=3.2, max_energy=2.6)
    assert len(result.marginal_trace) == len(result.selected_groups)
    assert all(gain > 0 for _, gain in result.marginal_trace)


def test_evaluation_uses_same_sensor_set_across_scenarios():
    system, _ = coupled_process_benchmark()
    sensors = ("pressure_in", "flow", "level")
    scenarios = {
        "clean": (),
        "dropout": (SensorDisturbance("flow", DisturbanceKind.DROPOUT, 10, 20),),
    }
    summary = evaluate_sensor_set(system, sensors, scenarios, method="test", steps=35, seed=9)
    assert len(summary.cases) == 2
    assert all(case.selected_sensor_count == 3 for case in summary.cases)


def test_benchmark_suite_has_distinct_domains():
    suite = benchmark_suite()
    assert set(suite) == {"coupled-process", "thermal-building", "battery-pack"}
    for system, groups in suite.values():
        assert system.state_dim == 4
        assert len(groups) >= 4


import pytest

from pgsgs.estimation import EstimatorConfig


@pytest.mark.parametrize(("field", "value"), [
    ("process_variance", float("nan")),
    ("initial_variance", float("inf")),
    ("measurement_variance", float("nan")),
    ("measurement_variance", 0.0),
    ("process_variance", True),
])
def test_estimator_rejects_invalid_variance_parameters(field, value):
    with pytest.raises(ValueError):
        EstimatorConfig(**{field: value})


@pytest.mark.parametrize("engine", [kalman_estimate, innovation_gated_estimate])
def test_estimators_reject_invalid_measurement_arrays(engine):
    system, _ = coupled_process_benchmark()
    names = ("pressure_in",)
    with pytest.raises(ValueError, match="nonempty shape"):
        engine(system, names, np.empty((0, 1)))
    with pytest.raises(ValueError, match="infinity"):
        engine(system, names, np.array([[float("inf")]]))
    with pytest.raises(ValueError, match="unique"):
        engine(system, (names[0], names[0]), np.zeros((1, 2)))


@pytest.mark.parametrize("engine", [kalman_estimate, innovation_gated_estimate])
def test_estimators_allow_nan_dropouts_and_reject_nonfinite_truth(engine):
    system, _ = coupled_process_benchmark()
    names = ("pressure_in",)
    trace = engine(system, names, np.array([[float("nan")], [0.1]]))
    assert trace.estimates.shape == (2, system.state_dim)
    with pytest.raises(ValueError, match="truth"):
        engine(system, names, np.zeros((2, 1)),
               truth=np.full((2, system.state_dim), float("nan")))


def test_kalman_rejects_nonfinite_initial_state():
    system, _ = coupled_process_benchmark()
    with pytest.raises(ValueError, match="initial_state"):
        kalman_estimate(system, ("flow",), np.zeros((2, 1)),
                        initial_state=np.full(system.state_dim, float("inf")))


@pytest.mark.parametrize(("field", "value"), [
    ("steps", 0), ("steps", 1.5), ("seed", True),
    ("process_std", float("nan")), ("measurement_std", float("inf")),
])
def test_synthetic_scenario_rejects_invalid_controls(field, value):
    system, _ = coupled_process_benchmark()
    with pytest.raises(ValueError):
        simulate_measurements(system, ("flow",), **{field: value})


@pytest.mark.parametrize("kwargs", [
    dict(sensor="flow", kind=DisturbanceKind.BIAS, start=0, end=3, magnitude=float("nan")),
    dict(sensor="flow", kind=DisturbanceKind.DRIFT, start=1.5, end=3),
])
def test_disturbance_rejects_invalid_magnitude_and_window(kwargs):
    with pytest.raises(ValueError):
        SensorDisturbance(**kwargs)
