"""Tests for structured guard-rail override detection.

`evaluate_credit_decision` exposes the full decision plus which guard rail (if
any) overrode the score-tier outcome, so every override can be logged as a
queryable event. `apply_business_rules` must remain a thin wrapper returning the
legacy 6-tuple (backward compatibility with existing callers/tests).
"""
import services


def test_no_override_on_clean_top_tier():
    ev = services.evaluate_credit_decision(820, monthly_income=1_000_000.0,
                                           debt_to_income_ratio=0.0, active_loans=0)
    assert ev["override_reason"] is None
    assert ev["decision"] == "APPROVED"
    assert ev["pre_override_decision"] == "APPROVED"


def test_high_dti_override_is_detected():
    ev = services.evaluate_credit_decision(820, monthly_income=1_000_000.0,
                                           debt_to_income_ratio=0.6, active_loans=0)
    assert ev["override_reason"] == "HIGH_DTI"
    assert ev["pre_override_decision"] == "APPROVED"
    assert ev["decision"] == "REJECTED"
    assert ev["limit"] == 0.0


def test_over_leverage_override_is_detected():
    ev = services.evaluate_credit_decision(820, monthly_income=1_000_000.0,
                                           debt_to_income_ratio=0.0, active_loans=6)
    assert ev["override_reason"] == "OVER_LEVERAGE"
    assert ev["pre_override_decision"] == "APPROVED"
    assert ev["decision"] == "MANUAL_REVIEW"
    assert ev["limit"] <= 1_000_000.0


def test_subsistence_income_override_is_detected():
    ev = services.evaluate_credit_decision(820, monthly_income=400_000.0,
                                           debt_to_income_ratio=0.0, active_loans=0)
    assert ev["override_reason"] == "SUBSISTENCE_INCOME_CAP"
    assert ev["limit"] <= 800_000.0  # 2x income cap


def test_apply_business_rules_still_returns_legacy_6_tuple():
    result = services.apply_business_rules(820, 1_000_000.0)
    assert isinstance(result, tuple)
    assert len(result) == 6
    decision, risk, prob, limit, rate, validity = result
    assert decision == "APPROVED"


def test_build_override_event_only_emits_on_override():
    clean = services.evaluate_credit_decision(820, 1_000_000.0, 0.0, 0)
    assert services.build_override_event(clean, customer_id="C1") is None

    overridden = services.evaluate_credit_decision(820, 1_000_000.0, 0.6, 0)
    event = services.build_override_event(overridden, customer_id="C1", loan_ref="L1")
    assert event is not None
    assert event["customer_id"] == "C1"
    assert event["loan_ref"] == "L1"
    assert event["rule_name"] == "HIGH_DTI"
    assert event["pre_override_decision"] == "APPROVED"
    assert event["post_override_decision"] == "REJECTED"
    assert event["actor"] == "system"
    assert "timestamp" in event
