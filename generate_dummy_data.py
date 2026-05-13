"""Synthetic credit dataset at production scale (TZS).

Generates ~10k rows whose feature ranges match what the live engine emits
from cms_uaa + cms_origination, so the trained model can score real
customers without going off-distribution.

Targets produced:
- payment_history_score   — regression target for CreditScorePredictor
- risk_category_target    — classification target for RiskCategoryPredictor
- is_approved, is_fraud   — kept for downstream models that use them
"""
import os
import numpy as np
import pandas as pd

OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "customer_data.csv")


def _log_uniform(low, high, size, rng):
    """Log-uniform draw so most customers cluster on the low end (realistic)."""
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def generate(num_records: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ─── Demographics ─────────────────────────────────────────────────────────
    age = rng.integers(21, 70, size=num_records)
    married = rng.choice(["YES", "NO"], size=num_records, p=[0.55, 0.45])
    education = rng.choice(
        ["PRIMARY", "SECONDARY", "DIPLOMA", "DEGREE", "POST_GRADUATE"],
        size=num_records, p=[0.20, 0.30, 0.20, 0.25, 0.05],
    )
    dependents = rng.integers(0, 7, size=num_records)
    employment_status = rng.choice(
        ["EMPLOYED", "SELF_EMPLOYED", "UNEMPLOYED"],
        size=num_records, p=[0.55, 0.35, 0.10],
    )
    spouse_employment_status = rng.choice(
        ["EMPLOYED", "UNEMPLOYED", "NOT_APPLICABLE"],
        size=num_records, p=[0.30, 0.20, 0.50],
    )
    residense_status = rng.choice(["OWNED", "RENTED"], size=num_records, p=[0.40, 0.60])
    vehicle_ownership_status = rng.choice(["YES", "NO"], size=num_records, p=[0.30, 0.70])
    vehicle_cat = np.where(
        vehicle_ownership_status == "YES",
        rng.choice(["CAR", "MOTORCYCLE", "TRUCK"], size=num_records, p=[0.5, 0.4, 0.1]),
        "None",
    )

    # ─── Financial — TZS scale ────────────────────────────────────────────────
    # Log-uniform: heavy-tailed, most customers earn 500K-5M, few earn 20M+
    monthly_income = _log_uniform(200_000, 50_000_000, num_records, rng).round(2)
    annual_income = (monthly_income * 12 * rng.normal(1.0, 0.05, num_records)).round(2)
    credit_limit = (_log_uniform(50_000, 500_000_000, num_records, rng)).round(2)

    # Debt skewed toward 0 (most customers have little outstanding debt)
    debt_multiplier = rng.beta(0.5, 2.0, num_records) * 20  # mostly 0..3, rare up to 20
    total_outstanding_debt = (monthly_income * debt_multiplier).round(2)

    active_loans = np.clip(rng.poisson(0.5, num_records), 0, 5)
    number_of_late_payments_36 = np.clip(rng.poisson(0.3, num_records), 0, 10)
    credit_history_length_months = rng.integers(6, 360, size=num_records)
    credit_utilization_ratio = np.clip(rng.beta(2, 5, num_records), 0.0, 1.0).round(4)

    avg_monthly_balance = (monthly_income * rng.uniform(0.5, 1.5, num_records)).round(2)
    savings_account_balance = (monthly_income * rng.uniform(0.0, 12.0, num_records)).round(2)

    requested_amount = _log_uniform(500_000, 50_000_000, num_records, rng).round(2)
    loan_purpose = rng.choice(
        ["Personal", "Business", "Home renovation", "Education", "Medical", "Vehicle"],
        size=num_records, p=[0.30, 0.30, 0.10, 0.10, 0.10, 0.10],
    )
    previous_collateral_value = np.where(
        rng.uniform(size=num_records) < 0.4,  # 40% have any collateral
        _log_uniform(100_000, 200_000_000, num_records, rng).round(2),
        0.0,
    )

    debt_to_income_ratio = (total_outstanding_debt / (annual_income.clip(min=1.0))).round(4)
    debt_to_income_ratio = np.clip(debt_to_income_ratio, 0.0, 5.0)

    # ─── Target: payment_history_score (300..850, derived from real signals) ──
    base = 750.0
    dti_penalty = 250.0 * np.clip(debt_to_income_ratio - 0.3, 0.0, 1.0)  # heavy if DTI > 0.3
    util_penalty = 100.0 * credit_utilization_ratio
    late_penalty = 30.0 * number_of_late_payments_36
    loan_penalty = 15.0 * np.clip(active_loans - 1, 0, 5)
    history_bonus = np.where(credit_history_length_months >= 24, 30.0, 0.0)
    employment_bonus = np.where(employment_status == "EMPLOYED", 25.0,
                        np.where(employment_status == "SELF_EMPLOYED", 10.0, -30.0))
    income_bonus = 40.0 * np.clip(np.log10(monthly_income / 500_000), 0.0, 2.0)
    noise = rng.normal(0.0, 15.0, num_records)

    payment_history_score = (
        base - dti_penalty - util_penalty - late_penalty - loan_penalty
        + history_bonus + employment_bonus + income_bonus + noise
    )
    payment_history_score = np.clip(payment_history_score, 300, 850).round(0)

    # ─── Target: risk_category (LOW / MEDIUM / HIGH from score) ───────────────
    risk_category_target = np.where(
        payment_history_score >= 700, "LOW",
        np.where(payment_history_score >= 500, "MEDIUM", "HIGH"),
    )

    # ─── Targets for downstream models ────────────────────────────────────────
    is_approved = (payment_history_score >= 640).astype(int)
    fraud_condition = (savings_account_balance < monthly_income * 0.1) & (requested_amount > monthly_income * 12)
    is_fraud = np.where(fraud_condition, 1,
                        (rng.uniform(size=num_records) < 0.02).astype(int))
    is_high_risk = (risk_category_target == "HIGH").astype(int)

    df = pd.DataFrame({
        "customer_id": [f"cust_{i}" for i in range(num_records)],
        "nida": [f"nida_{1000 + i}" for i in range(num_records)],
        "age": age,
        "married": married,
        "education": education,
        "dependents": dependents,
        "employment_status": employment_status,
        "spouse_employment_status": spouse_employment_status,
        "monthly_income": monthly_income,
        "residense_status": residense_status,
        "vehicle_ownership_status": vehicle_ownership_status,
        "vehicle_cat": vehicle_cat,
        "credit_history_length_months": credit_history_length_months,
        "payment_history_score": payment_history_score,
        "total_outstanding_debt": total_outstanding_debt,
        "credit_utilization_ratio": credit_utilization_ratio,
        "number_of_late_payments_36": number_of_late_payments_36,
        "active_loans": active_loans,
        "avg_monthly_balance": avg_monthly_balance,
        "savings_account_balance": savings_account_balance,
        "requested_amount": requested_amount,
        "loan_purpose": loan_purpose,
        "previous_collateral_value": previous_collateral_value,
        "debt_to_income_ratio": debt_to_income_ratio,
        "risk_category_target": risk_category_target,
        "is_approved": is_approved,
        "is_fraud": is_fraud,
        "is_high_risk": is_high_risk,
    })

    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print("\nSample distribution snapshot:")
    print(df[["monthly_income", "total_outstanding_debt", "payment_history_score",
              "risk_category_target", "active_loans"]].describe(include="all"))
    print("\nRisk category distribution:")
    print(df["risk_category_target"].value_counts(normalize=True).round(3))
