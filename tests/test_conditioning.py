import numpy as np
import pytest

from pgsgs.benchmarks import coupled_process_benchmark
from pgsgs.conditioning import observability_conditioning


def test_conditioning_reports_finite_quality_for_benchmark_sensor_set():
    system, _ = coupled_process_benchmark()
    names = tuple(system.sensors)[:2]
    report = observability_conditioning(system, names)
    assert report.rank > 0
    assert report.max_eigenvalue >= report.min_eigenvalue >= 0
    assert np.isfinite(report.regularized_logdet)


def test_empty_sensor_set_is_explicitly_unobservable():
    system, _ = coupled_process_benchmark()
    report = observability_conditioning(system, ())
    assert report.rank == 0
    assert report.min_eigenvalue == 0
    assert report.condition_number == float("inf")


def test_unknown_sensor_is_rejected():
    system, _ = coupled_process_benchmark()
    with pytest.raises(ValueError, match="unknown sensors"):
        observability_conditioning(system, ("missing",))


def test_regularization_must_be_positive():
    system, _ = coupled_process_benchmark()
    with pytest.raises(ValueError, match="regularization"):
        observability_conditioning(system, tuple(system.sensors)[:1], regularization=0)
