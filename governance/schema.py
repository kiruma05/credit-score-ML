"""Validation for governance records and data-dictionary entries.

Pure-Python (stdlib only) so the governance tests run without extra deps.
"""
from __future__ import annotations

from typing import List

# ── Per-model governance record ──────────────────────────────────────────────
REQUIRED_RECORD_FIELDS = [
    "model_id",
    "model_name",          # MLflow registered name
    "owner",
    "algorithm",
    "training_period",
    "training_data",       # data snapshot reference
    "target",              # target column / expression
    "target_type",         # synthetic_formula | empirical
    "features",            # count or reference to the data dictionary
    "known_limitations",
    "validation",          # metrics + method (may be "not_validated")
    "approval_status",
]

VALID_TARGET_TYPES = {"synthetic_formula", "empirical"}
VALID_APPROVAL = {
    "NOT_APPROVED_FOR_PRODUCTION",
    "PENDING_VALIDATION",
    "APPROVED_FOR_PRODUCTION",
    "RETIRED",
}


def validate_record(record: dict) -> List[str]:
    """Return a list of problems (empty means the record is valid)."""
    problems: List[str] = []
    for field in REQUIRED_RECORD_FIELDS:
        value = record.get(field)
        if value in (None, "", [], {}):
            problems.append(f"missing or empty field: {field}")

    ttype = record.get("target_type")
    if ttype is not None and ttype not in VALID_TARGET_TYPES:
        problems.append(f"invalid target_type: {ttype}")

    approval = record.get("approval_status")
    if approval is not None and approval not in VALID_APPROVAL:
        problems.append(f"invalid approval_status: {approval}")

    if not isinstance(record.get("known_limitations", []), list) or not record.get(
        "known_limitations"
    ):
        problems.append("known_limitations must be a non-empty list")

    return problems


# ── Data-dictionary entry ────────────────────────────────────────────────────
REQUIRED_FEATURE_FIELDS = ["name", "source", "dtype", "description", "timing"]
VALID_TIMING = {"sample_point", "outcome_point"}


def validate_feature_entry(entry: dict) -> List[str]:
    """Return a list of problems for one data-dictionary feature entry."""
    problems: List[str] = []
    for field in REQUIRED_FEATURE_FIELDS:
        if not entry.get(field):
            problems.append(f"missing field: {field}")
    timing = entry.get("timing")
    if timing is not None and timing not in VALID_TIMING:
        problems.append(f"invalid timing: {timing}")
    return problems
