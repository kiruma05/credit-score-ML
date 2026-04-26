from typing import Tuple
import random


def apply_business_rules(score: float, monthly_income: float) -> Tuple[str, str, float, float, float, int]:
    """
    Applies realistic, tiered business rules based on a generated credit score.

    Returns:
        - decision (str)
        - risk_category (str)
        - risk_probability (float)
        - final_limit (float, in USD)
        - interest_rate (float)
        - validity_period_days (int)
    """

    # Excellent: 800 - 850
    if 800 <= score <= 850:
        risk_category = "VERY_LOW"
        decision = "APPROVED"
        risk_probability = random.uniform(0.01, 0.05)
        interest_rate = random.uniform(8.0, 10.0)
        limit = monthly_income * 5.0
        validity_period_days = 90
    # Very Good: 740 - 799
    elif 740 <= score < 800:
        risk_category = "LOW"
        decision = "APPROVED"
        risk_probability = random.uniform(0.05, 0.15)
        interest_rate = random.uniform(10.1, 13.0)
        limit = monthly_income * 4.0
        validity_period_days = 60
    # Good: 670 - 739
    elif 670 <= score < 740:
        risk_category = "ACCEPTABLE"
        decision = "MANUAL_REVIEW"
        risk_probability = random.uniform(0.15, 0.30)
        interest_rate = random.uniform(13.1, 17.0)
        limit = monthly_income * 2.5
        validity_period_days = 30
    # Fair: 580 - 669
    elif 580 <= score < 670:
        risk_category = "SUBPRIME"
        decision = "MANUAL_REVIEW"
        risk_probability = random.uniform(0.30, 0.50)
        interest_rate = random.uniform(17.1, 22.0)
        limit = monthly_income * 1.5
        validity_period_days = 30
    # Poor: < 580
    else:
        risk_category = "HIGH"
        decision = "REJECTED"
        risk_probability = random.uniform(0.50, 0.90)
        interest_rate = 25.0
        limit = 0.0
        validity_period_days = 0

    return decision, risk_category, round(risk_probability, 4), round(limit, 2), round(interest_rate,
                                                                                       2), validity_period_days