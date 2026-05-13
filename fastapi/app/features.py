"""Feature extraction for the credit scoring engine.

Builds the 22 model features by joining across cms_uaa.user_accounts (NIDA →
uuid + demographics + income) and cms_origination (user_party_link → party_person
→ loan_application → employment_profile / applicant_asset / collateral_item /
loan_purpose) and falls back to the rich ``loan_application.metadata`` JSONB
column whenever a relational field is NULL.

Returns ``(features, data_quality)`` so callers can tell whether a score is
grounded in live data, partial live data, or seeded synthetic defaults.
"""
from __future__ import annotations

import sys
from datetime import date
from typing import Any, Optional, Tuple, List

from sqlalchemy import text
from sqlalchemy.orm import Session


ACTIVE_STATUSES_EXCLUDE = ("REJECTED", "CANCELLED", "SETTLED", "WITHDRAWN")
# Apps that should NOT influence scoring — drafts are incomplete, rejected/
# cancelled/withdrawn represent customer or bank rejection respectively.
QUALIFYING_STATUSES_EXCLUDE = ("REJECTED", "CANCELLED", "DRAFT", "WITHDRAWN")


def _years_between(dob: Any, today: Optional[date] = None) -> Optional[int]:
    if not dob:
        return None
    today = today or date.today()
    try:
        d = dob if isinstance(dob, date) else date.fromisoformat(str(dob)[:10])
        return today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    except Exception:
        return None


def _first_non_null(*values):
    """Return the first value that is not None, not empty string, and not zero."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, (int, float)) and v == 0:
            continue
        return v
    return None


def _resolve_uaa(nida: str, uaa_db: Session) -> Optional[dict]:
    """Look up customer in cms_uaa.user_accounts by NIDA."""
    try:
        q = text("""
            SELECT
                uuid::text                                          AS uuid,
                date_of_birth,
                COALESCE(monthly_income, 0)::float                  AS monthly_income,
                COALESCE(annual_income, 0)::float                   AS annual_income,
                COALESCE(credit_limit, 0)::float                    AS credit_limit,
                first_name,
                last_name,
                gender
            FROM user_accounts
            WHERE nin = :nida AND deleted = false
            LIMIT 1
        """)
        row = uaa_db.execute(q, {"nida": nida}).first()
        return dict(row._mapping) if row else None
    except Exception as e:
        print(f"[WARNING] cms_uaa lookup failed: {e}", file=sys.stderr)
        return None


def _resolve_party(user_uuid: str, origination_db: Session) -> Optional[str]:
    """Bridge cms_uaa.user_accounts.uuid → cms_origination.party_person via user_party_link."""
    try:
        q = text("""
            SELECT party_id::text AS party_id
            FROM user_party_link
            WHERE user_id = CAST(:uid AS uuid)
            ORDER BY linked_at DESC NULLS LAST
            LIMIT 1
        """)
        row = origination_db.execute(q, {"uid": user_uuid}).first()
        return row.party_id if row else None
    except Exception as e:
        print(f"[WARNING] user_party_link lookup failed: {e}", file=sys.stderr)
        return None


def _fetch_party_person(party_id: str, origination_db: Session) -> Optional[dict]:
    try:
        q = text("""
            SELECT
                marital_status,
                level_of_education,
                dependents,
                house_status,
                has_other_loan,
                has_previous_loan,
                COALESCE(other_loan_monthly_repayment, 0)::float AS other_loan_monthly_repayment,
                COALESCE(previous_loan_repayment, 0)::float      AS previous_loan_repayment,
                spouse_name,
                place_of_birth
            FROM party_person
            WHERE party_id = CAST(:pid AS uuid)
            LIMIT 1
        """)
        row = origination_db.execute(q, {"pid": party_id}).first()
        return dict(row._mapping) if row else None
    except Exception as e:
        print(f"[WARNING] party_person lookup failed: {e}", file=sys.stderr)
        return None


def _fetch_applications(party_id: str, origination_db: Session) -> List[dict]:
    """All applications for a party, latest first, joined with employment + purpose."""
    try:
        q = text("""
            SELECT
                la.application_id::text                         AS application_id,
                la.requested_amount,
                la.term_months,
                la.status,
                la.submitted_at,
                la.metadata,
                la.loan_purpose                                 AS purpose_code,
                la.education_level                              AS app_education,
                la.number_of_dependents                         AS app_dependents,
                la.marital_status                               AS app_marital,
                ep.employment_type,
                ep.gross_salary_monthly,
                ep.net_salary_monthly,
                ep.duration_years,
                ep.economic_sector,
                lp.purpose_text
            FROM loan_application la
            LEFT JOIN employment_profile ep ON ep.application_id = la.application_id
            LEFT JOIN loan_purpose lp       ON lp.application_id = la.application_id
            WHERE la.borrower_party_id = CAST(:pid AS uuid)
              AND COALESCE(la.deleted, false) = false
            ORDER BY la.submitted_at DESC NULLS LAST
        """)
        rows = origination_db.execute(q, {"pid": party_id}).all()
        return [dict(r._mapping) for r in rows]
    except Exception as e:
        print(f"[WARNING] loan_application join failed: {e}", file=sys.stderr)
        return []


def _fetch_collateral(application_ids: List[str], origination_db: Session) -> dict:
    if not application_ids:
        return {"total_value": 0.0, "has_vehicle": False, "vehicle_class": None}
    try:
        q = text("""
            SELECT
                COALESCE(SUM(estimated_value), 0)::float                AS total_value,
                BOOL_OR(vehicle_make IS NOT NULL OR vehicle_model IS NOT NULL) AS has_vehicle,
                MAX(vehicle_class)                                      AS vehicle_class
            FROM collateral_item
            WHERE application_id = ANY(CAST(:ids AS uuid[]))
        """)
        row = origination_db.execute(q, {"ids": application_ids}).first()
        return dict(row._mapping) if row else {"total_value": 0.0, "has_vehicle": False, "vehicle_class": None}
    except Exception as e:
        print(f"[WARNING] collateral_item aggregate failed: {e}", file=sys.stderr)
        return {"total_value": 0.0, "has_vehicle": False, "vehicle_class": None}


def _fetch_assets(application_ids: List[str], origination_db: Session) -> dict:
    if not application_ids:
        return {"total_value": 0.0, "has_vehicle": False, "vehicle_desc": None}
    try:
        q = text("""
            SELECT
                COALESCE(SUM(estimated_value), 0)::float                AS total_value,
                BOOL_OR(asset_desc ILIKE '%vehicle%' OR asset_desc ILIKE '%car%') AS has_vehicle,
                MAX(asset_desc) FILTER (
                    WHERE asset_desc ILIKE '%vehicle%' OR asset_desc ILIKE '%car%'
                ) AS vehicle_desc
            FROM applicant_asset
            WHERE application_id = ANY(CAST(:ids AS uuid[]))
        """)
        row = origination_db.execute(q, {"ids": application_ids}).first()
        return dict(row._mapping) if row else {"total_value": 0.0, "has_vehicle": False, "vehicle_desc": None}
    except Exception as e:
        print(f"[WARNING] applicant_asset aggregate failed: {e}", file=sys.stderr)
        return {"total_value": 0.0, "has_vehicle": False, "vehicle_desc": None}


def _metadata_avg_income_history(metadata: Any) -> Optional[float]:
    """metadata['monthlyIncomeHistory'] is [{"month": "Month 1", "amount": 123}, ...]."""
    if not isinstance(metadata, dict):
        return None
    hist = metadata.get("monthlyIncomeHistory") or []
    amounts = [m.get("amount") for m in hist if isinstance(m, dict) and m.get("amount") is not None]
    return float(sum(amounts) / len(amounts)) if amounts else None


def _metadata_assets_have_vehicle(metadata: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(metadata, dict):
        return False, None
    for a in (metadata.get("assetsOwned") or []):
        name = (a.get("assetName") or "").lower()
        if "vehicle" in name or "car" in name:
            return True, a.get("assetName")
    return False, None


def _max_metadata_value(apps: List[dict], field: str) -> Optional[float]:
    """Return MAX of metadata[field] across the given apps, ignoring null/<=0."""
    vals = []
    for a in apps:
        m = a.get("metadata") or {}
        v = m.get(field)
        if isinstance(v, (int, float)) and v > 0:
            vals.append(float(v))
    return max(vals) if vals else None


def _max_metadata_history_avg(apps: List[dict]) -> Optional[float]:
    """Return MAX of monthlyIncomeHistory averages across the given apps."""
    vals = []
    for a in apps:
        avg = _metadata_avg_income_history(a.get("metadata"))
        if avg is not None and avg > 0:
            vals.append(avg)
    return max(vals) if vals else None


def _latest_metadata_string(apps: List[dict], field: str) -> Optional[str]:
    """First non-empty string value of metadata[field] across apps (latest first)."""
    for a in apps:
        m = a.get("metadata") or {}
        v = m.get(field)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _metadata_collateral_total(metadata: Any) -> float:
    if not isinstance(metadata, dict):
        return 0.0
    total = 0.0
    for c in (metadata.get("collaterals") or []):
        val = c.get("estimatedValue") or c.get("amount")
        if isinstance(val, (int, float)):
            total += float(val)
    return total


def fetch_features(nida: str, uaa_db: Session, origination_db: Session) -> Tuple[dict, dict]:
    """Build the 22-feature dict + data_quality report.

    Never raises; returns quality flags so the caller decides whether to refuse
    (strict mode) or fall back (demo mode).
    """
    data_quality = {
        "uaa_source": "fallback",
        "party_link": "missing",
        "applications_found": 0,
        "qualifying_apps": 0,
        "latest_app_status": None,
        "employment_found": False,
        "collateral_found": False,
        "score_basis": "seeded_defaults",
    }

    features = {
        "age": 35, "married": "NO", "education": "Graduate", "dependents": 0,
        "employment_status": "Employed", "spouse_employment_status": "Unemployed",
        "monthly_income": 500.0,
        "residense_status": "Rented", "vehicle_ownership_status": "NO",
        "vehicle_cat": "None", "credit_history_length_months": 12,
        "payment_history_score": 500, "total_outstanding_debt": 0.0,
        "credit_utilization_ratio": 0.0, "number_of_late_payments_36": 0,
        "active_loans": 0, "avg_monthly_balance": 0.0, "savings_account_balance": 0.0,
        "requested_amount": 1000.0, "loan_purpose": "Personal",
        "previous_collateral_value": 0.0, "debt_to_income_ratio": 0.0,
    }

    # 1. cms_uaa
    uaa = _resolve_uaa(nida, uaa_db) if uaa_db is not None else None
    if not uaa:
        return features, data_quality
    data_quality["uaa_source"] = "live"
    user_uuid = uaa["uuid"]

    age = _years_between(uaa.get("date_of_birth"))
    if age:
        features["age"] = age
    if uaa.get("credit_limit"):
        features["savings_account_balance"] = float(uaa["credit_limit"])
    uaa_monthly = _first_non_null(
        uaa.get("monthly_income"),
        (uaa.get("annual_income") or 0) / 12 or None,
    )

    # 2. user_party_link → party_id (no link → bail with uaa-only quality)
    party_id = _resolve_party(user_uuid, origination_db) if origination_db is not None else None
    if not party_id:
        if uaa_monthly:
            features["monthly_income"] = float(uaa_monthly)
        data_quality["score_basis"] = "uaa_only"
        return features, data_quality
    data_quality["party_link"] = "live"

    # 3. party_person — demographic fields
    person = _fetch_party_person(party_id, origination_db) or {}
    if person:
        features["married"] = "YES" if (person.get("marital_status") or "").upper() == "MARRIED" else "NO"
        if person.get("level_of_education"):
            features["education"] = person["level_of_education"]
        if person.get("dependents") is not None:
            features["dependents"] = int(person["dependents"])
        if person.get("house_status"):
            features["residense_status"] = person["house_status"]

    # 4. all applications for this party
    apps = _fetch_applications(party_id, origination_db)
    data_quality["applications_found"] = len(apps)

    if not apps:
        if uaa_monthly:
            features["monthly_income"] = float(uaa_monthly)
        data_quality["score_basis"] = "partial_live_data"
        return features, data_quality

    # Qualifying = non-draft, non-rejected, non-cancelled, non-withdrawn.
    # These are the only apps that should influence scoring — DRAFT apps are
    # incomplete and may contain typos (e.g. monthlySalary=10).
    qualifying_apps = [
        a for a in apps
        if (a.get("status") or "").upper() not in QUALIFYING_STATUSES_EXCLUDE
    ]
    data_quality["qualifying_apps"] = len(qualifying_apps)
    data_quality["latest_app_status"] = apps[0].get("status")

    # Detail source = latest qualifying (fallback to latest of any kind for
    # the requested_amount/loan_purpose if no qualifying app exists yet).
    detail_source = qualifying_apps[0] if qualifying_apps else apps[0]

    # Cross-app metadata aggregates (MAX across qualifying so one bad-data
    # app cannot drag features down).
    max_history_avg = _max_metadata_history_avg(qualifying_apps)
    max_monthly_salary = _max_metadata_value(qualifying_apps, "monthlySalary")
    max_total_assets = _max_metadata_value(qualifying_apps, "totalAssetsValue")
    max_years_employment = _max_metadata_value(qualifying_apps, "yearsInEmployment")
    metadata_employment_status = _latest_metadata_string(qualifying_apps, "employmentStatus")

    # latest-app detail fields
    if detail_source.get("requested_amount") is not None:
        features["requested_amount"] = float(detail_source["requested_amount"])
    features["loan_purpose"] = (
        detail_source.get("purpose_text") or detail_source.get("purpose_code") or "Personal"
    )

    # Employment status — table first, metadata fallback
    if detail_source.get("employment_type"):
        features["employment_status"] = detail_source["employment_type"]
    elif metadata_employment_status:
        features["employment_status"] = metadata_employment_status
    data_quality["employment_found"] = bool(
        detail_source.get("employment_type") or metadata_employment_status
    )

    # Credit history length — table first, metadata fallback
    duration_years = _first_non_null(
        detail_source.get("duration_years"),
        max_years_employment,
    )
    if duration_years:
        features["credit_history_length_months"] = int(duration_years) * 12

    # Override demographics with application-level if party_person was sparse
    if not person.get("level_of_education") and detail_source.get("app_education"):
        features["education"] = detail_source["app_education"]
    if person.get("dependents") in (None, 0) and detail_source.get("app_dependents") is not None:
        features["dependents"] = int(detail_source["app_dependents"])
    if (not person.get("marital_status")) and detail_source.get("app_marital"):
        features["married"] = "YES" if (detail_source["app_marital"] or "").upper() == "MARRIED" else "NO"

    # Income priority chain — prefer history average (robust across 6 months)
    # over single monthlySalary value (vulnerable to typos like "10").
    monthly_income = _first_non_null(
        detail_source.get("gross_salary_monthly"),
        max_history_avg,
        max_monthly_salary,
        uaa_monthly,
    )
    if monthly_income:
        features["monthly_income"] = float(monthly_income)

    avg_balance = _first_non_null(
        detail_source.get("net_salary_monthly"),
        max_history_avg,
        detail_source.get("gross_salary_monthly"),
    )
    if avg_balance:
        features["avg_monthly_balance"] = float(avg_balance)

    # Savings / liquid balance — credit_limit from cms_uaa, metadata fallback
    if not features["savings_account_balance"] and max_total_assets:
        features["savings_account_balance"] = float(max_total_assets)

    # Active loans + outstanding debt aggregated across non-rejected apps.
    # Note: DRAFT apps DON'T count as active loans (they aren't loans yet).
    active_apps = [
        a for a in apps
        if (a.get("status") or "").upper() not in ACTIVE_STATUSES_EXCLUDE
        and (a.get("status") or "").upper() != "DRAFT"
    ]
    features["active_loans"] = len(active_apps)
    debt = sum(float(a.get("requested_amount") or 0) for a in active_apps)
    if (person.get("has_other_loan") or "").upper() == "YES":
        features["active_loans"] += 1
        debt += float(person.get("other_loan_monthly_repayment") or 0) * 12
    if (person.get("has_previous_loan") or "").upper() == "YES":
        features["payment_history_score"] = 700
    features["total_outstanding_debt"] = debt

    # Debt-to-income / utilisation (guard against div-by-zero)
    mi = features["monthly_income"] or 0
    if mi > 0:
        features["debt_to_income_ratio"] = round(debt / (mi * 12), 4)
    if features["savings_account_balance"] > 0:
        features["credit_utilization_ratio"] = round(
            min(1.0, debt / features["savings_account_balance"]), 4
        )

    # Collateral aggregate — qualifying apps only (drafts shouldn't grant
    # collateral credit). Falls back to metadata.collaterals if table empty.
    qualifying_ids = [a["application_id"] for a in qualifying_apps] or [a["application_id"] for a in apps]
    coll = _fetch_collateral(qualifying_ids, origination_db)
    if coll.get("total_value", 0) > 0:
        features["previous_collateral_value"] = float(coll["total_value"])
        data_quality["collateral_found"] = True
    else:
        meta_coll = _metadata_collateral_total(detail_source.get("metadata"))
        if meta_coll > 0:
            features["previous_collateral_value"] = meta_coll
            data_quality["collateral_found"] = True

    # Vehicle ownership (collateral, then assets, then metadata)
    if coll.get("has_vehicle"):
        features["vehicle_ownership_status"] = "YES"
        features["vehicle_cat"] = coll.get("vehicle_class") or "Vehicle"
    else:
        assets = _fetch_assets(qualifying_ids, origination_db)
        if assets.get("has_vehicle"):
            features["vehicle_ownership_status"] = "YES"
            features["vehicle_cat"] = assets.get("vehicle_desc") or "Vehicle"
        else:
            meta_veh, meta_veh_name = _metadata_assets_have_vehicle(detail_source.get("metadata"))
            if meta_veh:
                features["vehicle_ownership_status"] = "YES"
                features["vehicle_cat"] = meta_veh_name or "Vehicle"

    # score_basis classification
    if features["monthly_income"] > 500 and qualifying_apps and data_quality["employment_found"]:
        data_quality["score_basis"] = "live_data"
    elif qualifying_apps or apps:
        data_quality["score_basis"] = "partial_live_data"
    else:
        data_quality["score_basis"] = "uaa_only"

    return features, data_quality
