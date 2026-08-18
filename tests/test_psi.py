"""Tests for the Population Stability Index (PSI) monitoring computation.

PSI compares a current scoring population against the training baseline to detect
distribution drift. Convention: PSI < 0.1 = stable, 0.1–0.25 = monitor,
> 0.25 = significant shift → alert.
"""
import psi


def test_identical_distributions_have_near_zero_psi():
    baseline = list(range(1000))
    current = list(range(1000))
    value = psi.population_stability_index(baseline, current, buckets=10)
    assert value < 0.01


def test_no_alert_when_stable():
    baseline = list(range(1000))
    report = psi.psi_report(baseline, list(range(1000)), buckets=10)
    assert report["alert"] is False
    assert report["psi"] < psi.PSI_ALERT_THRESHOLD


def test_large_shift_triggers_alert():
    baseline = list(range(1000))            # 0..999
    current = [x + 5000 for x in range(1000)]  # fully shifted out of baseline range
    report = psi.psi_report(baseline, current, buckets=10)
    assert report["psi"] > psi.PSI_ALERT_THRESHOLD
    assert report["alert"] is True


def test_moderate_shift_between_thresholds():
    baseline = list(range(1000))
    # Push ~30% of mass upward — a noticeable but not total shift.
    current = list(range(700)) + [999] * 300
    value = psi.population_stability_index(baseline, current, buckets=10)
    assert value > 0.1


def test_report_structure_and_bucket_count():
    report = psi.psi_report(list(range(1000)), list(range(1000)), buckets=10)
    assert set(report) >= {"psi", "buckets", "alert", "threshold"}
    assert report["threshold"] == psi.PSI_ALERT_THRESHOLD
    assert len(report["buckets"]) >= 1


def test_handles_constant_baseline_without_crashing():
    # Degenerate baseline (no spread) must not raise.
    value = psi.population_stability_index([5.0] * 100, [5.0] * 100, buckets=10)
    assert value == 0.0


def test_empty_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        psi.population_stability_index([], [1, 2, 3])
