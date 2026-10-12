from pgsgs.statistics import paired_bootstrap_delta


def test_lower_rmse_candidate_has_positive_win_rate_and_negative_delta():
    baseline = [0.50, 0.60, 0.55, 0.70]
    candidate = [0.40, 0.50, 0.54, 0.60]
    result = paired_bootstrap_delta(baseline, candidate, bootstrap_samples=500, seed=1)
    assert result.mean_delta < 0
    assert result.candidate_win_rate == 1.0
    assert result.ci95_upper <= 0


def test_pairing_is_required():
    try:
        paired_bootstrap_delta([1.0, 2.0], [1.0], bootstrap_samples=100)
    except ValueError as exc:
        assert "matched" in str(exc)
    else:
        raise AssertionError("unmatched samples should fail")
