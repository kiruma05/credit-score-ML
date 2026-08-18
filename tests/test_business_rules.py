"""Regression + behavior tests for the rule-based decision layer.

Pins the *deterministic* outputs of ``apply_business_rules`` (decision, risk
category, limit, validity) so future refactors cannot silently change lending
behavior. ``interest_rate`` and ``risk_probability`` are currently
``random.uniform`` draws (a known weakness slated for Phase 4 calibration); these
tests therefore only assert their *range*, and document that non-determinism
explicitly.
"""
import random

import pytest

import services


# ---------------------------------------------------------------------------
# Tier mapping — deterministic fields only (decision / risk / limit / validity)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score, expected_decision, expected_risk, income_multiple, validity",
    [
        (820, "APPROVED", "VERY_LOW", 5.0, 90),
        (770, "APPROVED", "LOW", 4.0, 60),
        (700, "MANUAL_REVIEW", "ACCEPTABLE", 2.5, 30),
        (620, "MANUAL_REVIEW", "SUBPRIME", 1.5, 30),
        (400, "REJECTED", "HIGH", 0.0, 0),
    ],
)
def test_score_tiers(score, expected_decision, expected_risk, income_multiple, validity):
    income = 1_000_000.0
    decision, risk, prob, limit, rate, validity_days = services.apply_business_rules(
        score, monthly_income=income, debt_to_income_ratio=0.0, active_loans=0
    )
    assert decision == expected_decision
    assert risk == expected_risk
    assert limit == pytest.approx(income * income_multiple)
    assert validity_days == validity


def test_score_is_clamped_to_300_850():
    # Above range → top tier; below range → bottom tier. Never crashes.
    hi = services.apply_business_rules(9999, 1_000_000.0)
    lo = services.apply_business_rules(-500, 1_000_000.0)
    assert hi[0] == "APPROVED" and hi[1] == "VERY_LOW"
    assert lo[0] == "REJECTED" and lo[1] == "HIGH"


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------
def test_high_dti_hard_rejects_even_with_top_score():
    decision, risk, prob, limit, rate, validity = services.apply_business_rules(
        820, monthly_income=1_000_000.0, debt_to_income_ratio=0.6, active_loans=0
    )
    assert decision == "REJECTED"
    assert risk == "HIGH"
    assert limit == 0.0
    assert rate == 25.0
    assert validity == 0


def test_over_leverage_downgrades_approved_to_manual_review_and_caps_limit():
    income = 1_000_000.0
    decision, risk, prob, limit, rate, validity = services.apply_business_rules(
        820, monthly_income=income, debt_to_income_ratio=0.0, active_loans=5
    )
    assert decision == "MANUAL_REVIEW"
    assert limit <= income  # capped at 1x income


def test_subsistence_income_caps_limit_at_2x():
    income = 400_000.0  # < 500_000 subsistence threshold
    decision, risk, prob, limit, rate, validity = services.apply_business_rules(
        820, monthly_income=income, debt_to_income_ratio=0.0, active_loans=0
    )
    assert limit <= income * 2.0


def test_guard_rails_never_improve_a_decision():
    # A rejected score stays rejected regardless of clean secondary signals.
    decision, *_ = services.apply_business_rules(
        400, monthly_income=1_000_000.0, debt_to_income_ratio=0.0, active_loans=0
    )
    assert decision == "REJECTED"


# ---------------------------------------------------------------------------
# Documented non-determinism (Phase 4 will replace these with calibrated values)
# ---------------------------------------------------------------------------
def test_interest_and_probability_are_within_tier_ranges():
    _, _, prob, _, rate, _ = services.apply_business_rules(820, 1_000_000.0)
    assert 0.01 <= prob <= 0.05
    assert 8.0 <= rate <= 10.0


def test_rate_is_currently_non_deterministic_known_weakness():
    """Same input → different interest rate across calls (weakness #2 / Phase 4).

    This test PINS the current (undesired) behavior so that when Phase 4 makes the
    rate deterministic, this test is intentionally updated — a tripwire that the
    calibration work actually changed the behavior.
    """
    random.seed()  # ensure we are not accidentally seeded elsewhere
    rates = {services.apply_business_rules(770, 1_000_000.0)[4] for _ in range(50)}
    assert len(rates) > 1, "interest rate unexpectedly deterministic — update for Phase 4"
