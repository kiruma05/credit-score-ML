from typing import Tuple
import random


def apply_business_rules(
    score: float,
    monthly_income: float,
    debt_to_income_ratio: float = 0.0,
    active_loans: int = 0,
) -> Tuple[str, str, float, float, float, int]:
    """Tiered credit decision rules plus hard rejection / cap guard rails.

    The model output (``score``) is first clamped to the conventional 300–850
    range, then mapped to a tier (decision, risk, interest, limit, validity).
    Finally guard rails override the model when business-level red flags are
    present — these never make a decision better than the model said, only
    catch obviously risky cases.

    Returns
    -------
    decision : str
    risk_category : str
    risk_probability : float
    final_limit : float  (currency-neutral; caller converts USD↔TZS)
    interest_rate : float
    validity_period_days : int
    """
    # ─── Clamp ────────────────────────────────────────────────────────────────
    score = max(300.0, min(850.0, float(score)))

    # ─── Tiered decision based on score band ─────────────────────────────────
    if 800 <= score <= 850:
        risk_category = "VERY_LOW"
        decision = "APPROVED"
        risk_probability = random.uniform(0.01, 0.05)
        interest_rate = random.uniform(8.0, 10.0)
        limit = monthly_income * 5.0
        validity_period_days = 90
    elif 740 <= score < 800:
        risk_category = "LOW"
        decision = "APPROVED"
        risk_probability = random.uniform(0.05, 0.15)
        interest_rate = random.uniform(10.1, 13.0)
        limit = monthly_income * 4.0
        validity_period_days = 60
    elif 670 <= score < 740:
        risk_category = "ACCEPTABLE"
        decision = "MANUAL_REVIEW"
        risk_probability = random.uniform(0.15, 0.30)
        interest_rate = random.uniform(13.1, 17.0)
        limit = monthly_income * 2.5
        validity_period_days = 30
    elif 580 <= score < 670:
        risk_category = "SUBPRIME"
        decision = "MANUAL_REVIEW"
        risk_probability = random.uniform(0.30, 0.50)
        interest_rate = random.uniform(17.1, 22.0)
        limit = monthly_income * 1.5
        validity_period_days = 30
    else:
        risk_category = "HIGH"
        decision = "REJECTED"
        risk_probability = random.uniform(0.50, 0.90)
        interest_rate = 25.0
        limit = 0.0
        validity_period_days = 0

    # ─── Guard rails ─────────────────────────────────────────────────────────
    # Hard reject on excessive debt-to-income (regulators flag > 0.5).
    if debt_to_income_ratio > 0.5:
        decision = "REJECTED"
        risk_category = "HIGH"
        risk_probability = max(risk_probability, 0.65)
        limit = 0.0
        interest_rate = 25.0
        validity_period_days = 0
    # Already heavily loaned-up: bring to manual review and cap limit at 1x income.
    elif active_loans >= 5:
        if decision == "APPROVED":
            decision = "MANUAL_REVIEW"
        limit = min(limit, monthly_income)
    # Subsistence income — cap limit at 2x to avoid over-extension.
    elif monthly_income < 500_000:
        limit = min(limit, monthly_income * 2.0)

    return (
        decision,
        risk_category,
        round(risk_probability, 4),
        round(limit, 2),
        round(interest_rate, 2),
        validity_period_days,
    )
