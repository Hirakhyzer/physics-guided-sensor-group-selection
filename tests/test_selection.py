import numpy as np

from pgsgs import (
    PhysicsSystem,
    Sensor,
    SensorGroup,
    coupled_process_benchmark,
    evaluate_sensor_dropout,
    group_physics_score,
    select_groups,
)


def test_full_state_group_is_fully_observable_for_identity_dynamics():
    sensors = {
        "x0": Sensor("x0", 0),
        "x1": Sensor("x1", 1),
    }
    system = PhysicsSystem(A=np.eye(2), sensors=sensors, coupling=np.zeros((2, 2)))
    score = group_physics_score(system, SensorGroup("full", ("x0", "x1")))
    assert score.observability_rank == 2
    assert score.observability_fraction == 1.0


def test_selector_respects_cost_and_energy_budgets():
    system, groups = coupled_process_benchmark()
    result = select_groups(system, groups, max_cost=2.1, max_energy=1.6)
    assert result.total_cost <= 2.1
    assert result.total_energy <= 1.6


def test_overlapping_groups_do_not_double_charge_shared_sensor():
    system, _ = coupled_process_benchmark()
    groups = [
        SensorGroup("a", ("flow", "level")),
        SensorGroup("b", ("level", "pressure_out")),
    ]
    result = select_groups(system, groups, max_cost=10.0, max_energy=10.0, max_groups=2)
    expected_cost = sum(system.sensors[name].cost for name in result.selected_sensors)
    assert result.total_cost == expected_cost


def test_dropout_report_covers_each_single_sensor_failure():
    system, _ = coupled_process_benchmark()
    sensors = ("pressure_in", "flow", "level")
    report = evaluate_sensor_dropout(system, sensors, failures=1)
    assert len(report.cases) == len(sensors)
    assert report.worst_case_observability <= report.baseline_observability


def test_duplicate_sensor_group_definition_is_rejected():
    try:
        SensorGroup("bad", ("flow", "flow"))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate sensors should fail")
