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


import pytest


@pytest.mark.parametrize(("field", "value"), [
    ("cost", float("nan")),
    ("energy", float("inf")),
    ("reliability", float("nan")),
    ("reliability", -0.1),
    ("state_index", -1),
    ("state_index", 1.5),
    ("state_index", True),
])
def test_sensor_rejects_invalid_resource_and_index_values(field, value):
    kwargs = dict(name="sensor", state_index=0)
    kwargs[field] = value
    with pytest.raises(ValueError):
        Sensor(**kwargs)


@pytest.mark.parametrize(("A", "coupling"), [
    ([[float("nan")]], [[0.0]]),
    ([[1.0]], [[float("inf")]]),
    (np.zeros((0, 0)), np.zeros((0, 0))),
    ([[1.0, 0.0]], [[0.0, 0.0]]),
])
def test_physics_model_rejects_nonfinite_or_invalid_matrices(A, coupling):
    with pytest.raises(ValueError):
        PhysicsSystem(A=np.asarray(A), sensors={}, coupling=np.asarray(coupling))


def test_physics_system_snapshots_model_matrices():
    source = np.eye(2)
    system = PhysicsSystem(A=source, sensors={"s": Sensor("s", 0)}, coupling=np.zeros((2, 2)))
    source[0, 0] = 50.0
    assert system.A[0, 0] == 1.0
    with pytest.raises(ValueError):
        system.A[0, 0] = 5.0


def test_physics_system_rejects_mismatched_sensor_keys():
    with pytest.raises(ValueError, match="mapping keys"):
        PhysicsSystem(A=np.eye(1), sensors={"wrong": Sensor("correct", 0)}, coupling=np.zeros((1, 1)))


def test_group_rejects_invalid_name_and_members():
    with pytest.raises(ValueError, match="group name"):
        SensorGroup("", ("s",))
    with pytest.raises(ValueError, match="members"):
        SensorGroup("group", ("",))


from pgsgs.adaptive import reconfigure_selection
from pgsgs.baselines import cheapest_first, observability_greedy
from pgsgs.selection import SelectionWeights


@pytest.mark.parametrize(("field", "value"), [
    ("max_cost", float("nan")),
    ("max_energy", float("inf")),
    ("max_cost", -1.0),
    ("max_energy", True),
    ("max_groups", -1),
    ("max_groups", 2.5),
    ("max_groups", True),
])
def test_selector_rejects_invalid_budgets_and_group_limits(field, value):
    system, groups = coupled_process_benchmark()
    args = dict(max_cost=3.2, max_energy=2.6)
    args[field] = value
    with pytest.raises(ValueError):
        select_groups(system, groups, **args)


def test_selection_rejects_duplicate_group_names():
    system, _ = coupled_process_benchmark()
    groups = [SensorGroup("same", ("flow",)), SensorGroup("same", ("level",))]
    with pytest.raises(ValueError, match="names must be unique"):
        select_groups(system, groups, max_cost=3.2, max_energy=2.6)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -0.1, True])
def test_selector_rejects_invalid_weight(weight):
    with pytest.raises(ValueError):
        SelectionWeights(energy=weight)


@pytest.mark.parametrize("method", [cheapest_first, observability_greedy])
def test_baselines_reject_invalid_resource_limits(method):
    system, _ = coupled_process_benchmark()
    with pytest.raises(ValueError, match="max_cost"):
        method(system, max_cost=float("nan"), max_energy=2.0)


@pytest.mark.parametrize(("field", "value"), [
    ("minimum_observability", float("nan")),
    ("minimum_observability", float("inf")),
    ("minimum_observability", 1.2),
    ("reliability_updates", {"flow": float("nan")}),
    ("reliability_updates", {"flow": float("inf")}),
])
def test_reconfiguration_rejects_invalid_thresholds(field, value):
    system, groups = coupled_process_benchmark()
    kwargs = dict(max_cost=3.2, max_energy=2.6)
    kwargs[field] = value
    with pytest.raises(ValueError):
        reconfigure_selection(system, groups, **kwargs)
