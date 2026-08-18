"""Leakage guard: no model may train on its own target or a proxy for it.

Two layers of protection:
1. Contract test on the shared ``leakage.feature_columns`` helper.
2. Source-level assertion that BOTH trainers route their feature selection
   through that helper (the trainers import pandas/pyspark and cannot be imported
   here, so we assert against their source text).
"""
import os
import re

import leakage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(REPO_ROOT, "airflow", "jobs")

TARGET_AND_LEAKAGE = {
    "payment_history_score",
    "risk_category_target",
    "is_approved",
    "is_fraud",
    "is_high_risk",
    "customer_id",
    "nida",
}


def test_feature_columns_strips_every_leakage_column():
    all_cols = [
        "age", "monthly_income", "employment_status",  # legitimate features
        *TARGET_AND_LEAKAGE,
    ]
    selected = leakage.feature_columns(all_cols)
    assert set(selected) == {"age", "monthly_income", "employment_status"}
    for bad in TARGET_AND_LEAKAGE:
        assert bad not in selected


def test_feature_columns_preserves_order_and_non_leakage():
    cols = ["b", "payment_history_score", "a", "nida", "c"]
    assert leakage.feature_columns(cols) == ["b", "a", "c"]


def test_leakage_set_covers_all_known_targets_and_ids():
    assert TARGET_AND_LEAKAGE <= set(leakage.LEAKAGE_COLUMNS)


def _read(job_filename):
    with open(os.path.join(JOBS_DIR, job_filename), encoding="utf-8") as fh:
        return fh.read()


def test_both_trainers_use_the_shared_leakage_helper():
    for trainer in ("train_models_pandas.py", "train_models.py"):
        src = _read(trainer)
        assert re.search(r"\bfeature_columns\b", src), (
            f"{trainer} must select features via leakage.feature_columns()"
        )
        assert "leakage" in src, f"{trainer} must import the shared leakage module"
