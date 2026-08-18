"""Governance-artifact tests.

Enforce that every registered model has a complete governance record, that the
current models are honestly marked as synthetic-target / not-production-approved,
and that the data dictionary covers every served feature and contains no
outcome-point (post-application) feature — a documented, stricter leakage guard.
"""
import json
import os

import pytest

from governance import schema

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOV_DIR = os.path.join(REPO_ROOT, "governance")

MODEL_RECORDS = [
    "CreditScorePredictor.json",
    "RiskCategoryPredictor.json",
    "CreditLimitPredictor.json",
    "FraudDetector.json",
]


def _load(name):
    with open(os.path.join(GOV_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("fname", MODEL_RECORDS)
def test_model_governance_record_is_complete(fname):
    problems = schema.validate_record(_load(fname))
    assert problems == [], problems


def test_all_current_models_are_synthetic_and_not_production_approved():
    for fname in MODEL_RECORDS:
        rec = _load(fname)
        assert rec["target_type"] == "synthetic_formula", fname
        assert rec["approval_status"] == "NOT_APPROVED_FOR_PRODUCTION", fname


# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------
def _load_dd():
    return _load("data_dictionary.json")


def test_data_dictionary_covers_every_served_feature():
    import features

    feats, _ = features.fetch_features("0", uaa_db=None, origination_db=None)
    documented = {e["name"] for e in _load_dd()["features"]}
    missing = set(feats) - documented
    assert not missing, f"features missing from data dictionary: {missing}"


def test_data_dictionary_entries_are_valid():
    for entry in _load_dd()["features"]:
        problems = schema.validate_feature_entry(entry)
        assert problems == [], (entry.get("name"), problems)


def test_no_served_feature_is_outcome_point():
    """Every feature used at scoring time must be known at application time.

    An outcome-point feature (only known after the loan performs) would be
    leakage. This pins the invariant so a future feature addition can't sneak
    post-application data into the model.
    """
    import features

    feats, _ = features.fetch_features("0", uaa_db=None, origination_db=None)
    by_name = {e["name"]: e for e in _load_dd()["features"]}
    offenders = [n for n in feats if by_name[n]["timing"] != "sample_point"]
    assert not offenders, f"outcome-point features served to the model: {offenders}"
