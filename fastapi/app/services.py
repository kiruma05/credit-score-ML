from typing import Tuple, Optional
from datetime import datetime, timezone
import random


# ─── Guard-rail reason codes ─────────────────────────────────────────────────
# Stable identifiers logged on every override event so triggers are queryable.
GUARD_RAIL_HIGH_DTI = "HIGH_DTI"
GUARD_RAIL_OVER_LEVERAGE = "OVER_LEVERAGE"
GUARD_RAIL_SUBSISTENCE_INCOME_CAP = "SUBSISTENCE_INCOME_CAP"

# Guard-rail thresholds (externalized so risk/compliance can tune them).
DTI_HARD_REJECT_THRESHOLD = 0.5      # regulators flag DTI > 0.5
OVER_LEVERAGE_LOAN_COUNT = 5         # >= this many active loans → manual review + 1x cap
SUBSISTENCE_INCOME_THRESHOLD = 500_000  # below this → cap limit at 2x income


def _score_tier(score: float, monthly_income: float) -> dict:
    """Map a clamped score to its tier outputs (pre guard rails).

    ``risk_probability`` and ``interest_rate`` are drawn with ``random.uniform``
    within the tier band — a known weakness (non-deterministic) slated for Phase 4
    calibration; preserved here byte-for-byte so behavior is unchanged.
    """
    if 800 <= score <= 850:
        return {
            "risk_category": "VERY_LOW", "decision": "APPROVED",
            "risk_probability": random.uniform(0.01, 0.05),
            "interest_rate": random.uniform(8.0, 10.0),
            "limit": monthly_income * 5.0, "validity_period_days": 90,
        }
    if 740 <= score < 800:
        return {
            "risk_category": "LOW", "decision": "APPROVED",
            "risk_probability": random.uniform(0.05, 0.15),
            "interest_rate": random.uniform(10.1, 13.0),
            "limit": monthly_income * 4.0, "validity_period_days": 60,
        }
    if 670 <= score < 740:
        return {
            "risk_category": "ACCEPTABLE", "decision": "MANUAL_REVIEW",
            "risk_probability": random.uniform(0.15, 0.30),
            "interest_rate": random.uniform(13.1, 17.0),
            "limit": monthly_income * 2.5, "validity_period_days": 30,
        }
    if 580 <= score < 670:
        return {
            "risk_category": "SUBPRIME", "decision": "MANUAL_REVIEW",
            "risk_probability": random.uniform(0.30, 0.50),
            "interest_rate": random.uniform(17.1, 22.0),
            "limit": monthly_income * 1.5, "validity_period_days": 30,
        }
    return {
        "risk_category": "HIGH", "decision": "REJECTED",
        "risk_probability": random.uniform(0.50, 0.90),
        "interest_rate": 25.0, "limit": 0.0, "validity_period_days": 0,
    }


def evaluate_credit_decision(
    score: float,
    monthly_income: float,
    debt_to_income_ratio: float = 0.0,
    active_loans: int = 0,
) -> dict:
    """Full, structured credit decision: score tier + guard rails + override event.

    Returns a dict with the final decision fields plus, when a guard rail
    materially changed the tier outcome, the ``override_reason`` and the
    pre/post decision — so every override can be logged as a queryable event.
    Guard rails never improve a decision, only catch red flags.
    """
    score = max(300.0, min(850.0, float(score)))
    tier = _score_tier(score, monthly_income)

    decision = tier["decision"]
    risk_category = tier["risk_category"]
    risk_probability = tier["risk_probability"]
    interest_rate = tier["interest_rate"]
    limit = tier["limit"]
    validity_period_days = tier["validity_period_days"]

    base_decision = decision
    base_limit = limit
    override_reason: Optional[str] = None

    # ─── Guard rails (mutually exclusive, ordered by severity) ───────────────
    if debt_to_income_ratio > DTI_HARD_REJECT_THRESHOLD:
        decision = "REJECTED"
        risk_category = "HIGH"
        risk_probability = max(risk_probability, 0.65)
        limit = 0.0
        interest_rate = 25.0
        validity_period_days = 0
        override_reason = GUARD_RAIL_HIGH_DTI
    elif active_loans >= OVER_LEVERAGE_LOAN_COUNT:
        if decision == "APPROVED":
            decision = "MANUAL_REVIEW"
        limit = min(limit, monthly_income)
        override_reason = GUARD_RAIL_OVER_LEVERAGE
    elif monthly_income < SUBSISTENCE_INCOME_THRESHOLD:
        limit = min(limit, monthly_income * 2.0)
        override_reason = GUARD_RAIL_SUBSISTENCE_INCOME_CAP

    # Only report an override that MATERIALLY changed the outcome (decision or
    # limit). A guard rail that fired but changed nothing is not logged as noise.
    materially_changed = (decision != base_decision) or (
        round(limit, 2) != round(base_limit, 2)
    )
    if not materially_changed:
        override_reason = None

    return {
        "score": score,
        "decision": decision,
        "risk_category": risk_category,
        "risk_probability": round(risk_probability, 4),
        "limit": round(limit, 2),
        "interest_rate": round(interest_rate, 2),
        "validity_period_days": validity_period_days,
        "pre_override_decision": base_decision,
        "post_override_decision": decision,
        "pre_override_limit": round(base_limit, 2),
        "override_reason": override_reason,
    }


def apply_business_rules(
    score: float,
    monthly_income: float,
    debt_to_income_ratio: float = 0.0,
    active_loans: int = 0,
) -> Tuple[str, str, float, float, float, int]:
    """Tiered credit decision rules plus hard rejection / cap guard rails.

    Backward-compatible thin wrapper over :func:`evaluate_credit_decision` that
    returns the legacy 6-tuple.

    Returns
    -------
    decision : str
    risk_category : str
    risk_probability : float
    final_limit : float  (currency-neutral; caller converts USD↔TZS)
    interest_rate : float
    validity_period_days : int
    """
    ev = evaluate_credit_decision(
        score, monthly_income, debt_to_income_ratio, active_loans
    )
    return (
        ev["decision"],
        ev["risk_category"],
        ev["risk_probability"],
        ev["limit"],
        ev["interest_rate"],
        ev["validity_period_days"],
    )


def build_override_event(
    evaluation: dict,
    *,
    customer_id: str,
    loan_ref: Optional[str] = None,
    actor: str = "system",
) -> Optional[dict]:
    """Build a structured, loggable override event from an evaluation.

    Returns ``None`` when no guard rail materially overrode the score tier, so
    callers can simply skip logging. When an override occurred, returns the
    fields needed for the audit trail: who/what, which rule, and the pre/post
    decision + score.
    """
    if not evaluation.get("override_reason"):
        return None
    return {
        "customer_id": customer_id,
        "loan_ref": loan_ref,
        "rule_name": evaluation["override_reason"],
        "pre_override_decision": evaluation["pre_override_decision"],
        "post_override_decision": evaluation["post_override_decision"],
        "pre_override_score": evaluation["score"],
        "pre_override_limit": evaluation["pre_override_limit"],
        "post_override_limit": evaluation["limit"],
        "actor": actor,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
