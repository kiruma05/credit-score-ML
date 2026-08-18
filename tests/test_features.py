"""Regression tests for the feature pipeline (``features.py``).

Full DB-backed extraction is covered by integration tests elsewhere; here we pin
the deterministic, DB-free behaviors that protect refactors:
- the seeded default feature vector and its casing contract,
- the ``uaa_db is None`` fallback path (returns defaults + fallback quality),
- the pure helper functions.
"""
from datetime import date

import features


# ---------------------------------------------------------------------------
# Default / fallback path — no database available
# ---------------------------------------------------------------------------
def test_no_uaa_db_returns_seeded_defaults_and_fallback_quality():
    feats, quality = features.fetch_features("00000000", uaa_db=None, origination_db=None)
    # Fallback signalling drives strict-mode rejection / demo-mode seeding upstream.
    assert quality["uaa_source"] == "fallback"
    assert quality["score_basis"] == "seeded_defaults"
    assert quality["applications_found"] == 0
    # A representative slice of the 22-feature default vector.
    assert feats["age"] == 35
    assert feats["monthly_income"] == 500.0
    assert feats["employment_status"] == "EMPLOYED"


def test_default_categoricals_are_uppercase():
    # Mixed-case would be silently zeroed by the trainer's OneHotEncoder
    # (handle_unknown='ignore'); defaults must match the learned UPPERCASE cats.
    feats, _ = features.fetch_features("00000000", uaa_db=None, origination_db=None)
    for key in ("married", "education", "employment_status", "residense_status"):
        assert feats[key] == feats[key].upper()


def test_feature_vector_has_expected_keys():
    feats, _ = features.fetch_features("00000000", uaa_db=None, origination_db=None)
    # The default vector has 21 features. NOTE: the module docstring historically
    # claimed "22"; this test pins the real count so the docs stay honest.
    assert len(feats) == 21


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_years_between_computes_age():
    assert features._years_between(date(1990, 1, 1), today=date(2020, 1, 1)) == 30
    # Birthday not yet reached this year.
    assert features._years_between(date(1990, 6, 1), today=date(2020, 5, 31)) == 29


def test_years_between_handles_missing():
    assert features._years_between(None) is None
    assert features._years_between("not-a-date") is None


def test_first_non_null_skips_none_empty_and_zero():
    assert features._first_non_null(None, "", 0, 42) == 42
    assert features._first_non_null(None, "  ", 0.0) is None
    assert features._first_non_null("value") == "value"


def test_metadata_avg_income_history():
    meta = {"monthlyIncomeHistory": [{"amount": 100}, {"amount": 200}, {"amount": 300}]}
    assert features._metadata_avg_income_history(meta) == 200.0
    assert features._metadata_avg_income_history({}) is None
    assert features._metadata_avg_income_history(None) is None
