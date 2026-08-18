"""Tests for the PSI monitoring IO layer (score loading from the training CSV)."""
import os

import psi_monitoring

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "customer_data.csv")


def test_load_baseline_scores_from_real_dataset():
    scores = psi_monitoring.load_baseline_scores(CSV, column="payment_history_score")
    assert len(scores) > 0
    assert all(isinstance(s, float) for s in scores)
    # Scores are clamped to 300–850 by the generator.
    assert all(300.0 <= s <= 850.0 for s in scores)


def test_load_baseline_scores_ignores_missing_column_gracefully():
    scores = psi_monitoring.load_baseline_scores(CSV, column="does_not_exist")
    assert scores == []


def test_load_current_scores_returns_empty_without_db(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert psi_monitoring.load_current_scores() == []
