"""Single source of truth for training-time leakage exclusion.

Both credit trainers (`train_models_pandas.py` and `train_models.py`) MUST build
their feature set through :func:`feature_columns` so that no model can ever see
its own target — or a proxy derived from it — as an input feature.

Why this exists
---------------
`payment_history_score` is the regression target of ``CreditScorePredictor``;
``risk_category_target`` is the classification target of ``RiskCategoryPredictor``.
``is_approved`` / ``is_fraud`` / ``is_high_risk`` are all derived from the score
in ``generate_dummy_data.py`` and are therefore leakage. ``customer_id`` / ``nida``
are identifiers. Letting any of these into the feature matrix lets the model
"cheat" (the pandas trainer's comment notes this caused the model to regurgitate
the default score of 500).

This module has **no third-party dependencies** on purpose, so the leakage-guard
test can import it without pulling in pandas/pyspark/mlflow.
"""
from __future__ import annotations

from typing import Iterable, List

# Columns that must never appear as model input features.
LEAKAGE_COLUMNS = frozenset(
    {
        "customer_id",
        "nida",
        "payment_history_score",  # target: CreditScorePredictor (regression)
        "risk_category_target",   # target: RiskCategoryPredictor (classification)
        "is_approved",            # derived from score → leakage
        "is_fraud",               # derived label → leakage
        "is_high_risk",           # derived from risk_category_target → leakage
    }
)


def feature_columns(all_columns: Iterable[str]) -> List[str]:
    """Return only the safe feature columns, preserving input order.

    Every trainer must derive its feature list from this function so the
    leakage-exclusion policy is defined in exactly one place.
    """
    return [c for c in all_columns if c not in LEAKAGE_COLUMNS]
