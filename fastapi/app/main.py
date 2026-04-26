import os
import uuid
import random
import sys
from datetime import datetime, date, timedelta
from urllib.parse import quote_plus
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker, joinedload
from sqlalchemy import create_engine, text
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from decimal import Decimal
import pandas as pd
import numpy as np
import requests
import json
import mlflow
from mlflow.tracking import MlflowClient

from app import models, services
from app.database import SessionLocal, engine
from app.fraud.fraud_model import load_model, load_label_encoders
from app.fraud.fraud_router import router as fraud_router

app = FastAPI(
    title="Credit Scoring & Fraud Detection API",
    description="Unified API for credit scoring, fraud detection, and loan lifecycle management.",
    version="5.5.4",
)

app.include_router(fraud_router, prefix="/fraud", tags=["Fraud Detection"])

USD_TO_TZS_RATE = 1
MLFLOW_CLIENT = None

models.Base.metadata.create_all(bind=engine)

try:
    EXTERNAL_DB_URL = (
        f"postgresql+psycopg2://{os.getenv('EXTERNAL_DB_USER')}:{quote_plus(os.getenv('EXTERNAL_DB_PASSWORD', ''))}"
        f"@{os.getenv('EXTERNAL_DB_HOST')}:{os.getenv('EXTERNAL_DB_PORT')}/{os.getenv('EXTERNAL_DB_NAME')}"
    )
    external_engine = create_engine(EXTERNAL_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    ExternalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=external_engine)
    print("[INFO] UAA database connection established (cms_uaa).")
except Exception as e:
    print(f"[WARNING] Failed to connect to UAA DB: {e}", file=sys.stderr)
    ExternalSessionLocal = None

try:
    ORIGINATION_DB_URL = (
        f"postgresql+psycopg2://{os.getenv('ORIGINATION_DB_USER')}:{quote_plus(os.getenv('ORIGINATION_DB_PASSWORD', ''))}"
        f"@{os.getenv('ORIGINATION_DB_HOST')}:{os.getenv('ORIGINATION_DB_PORT')}/{os.getenv('ORIGINATION_DB_NAME')}"
    )
    origination_engine = create_engine(ORIGINATION_DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 10})
    OriginationSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=origination_engine)
    print("[INFO] Origination database connection established (cms_origination).")
except Exception as e:
    print(f"[WARNING] Failed to connect to Origination DB: {e}", file=sys.stderr)
    OriginationSessionLocal = None

@app.on_event("startup")
def startup_event():
    global MLFLOW_CLIENT
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            MLFLOW_CLIENT = MlflowClient(tracking_uri=tracking_uri)
            print(f"[INFO] Connected to MLflow Tracking Server: {tracking_uri}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize MLflow client: {e}", file=sys.stderr)
    else:
        print("[WARNING] MLFLOW_TRACKING_URI not set.", file=sys.stderr)
    print("--- FastAPI Startup: API Ready ---")

    try:
        app.state.fraud_model = load_model()
        app.state.label_encoders = load_label_encoders()
        print(f"[INFO] Fraud detection model loaded successfully: {app.state.fraud_model}")
        print(f"[INFO] Label encoders loaded: {app.state.label_encoders}")
    except Exception as e:
        print(f"[ERROR] Failed to load fraud model or encoders: {str(e)}", file=sys.stderr)
        app.state.fraud_model = None
        app.state.label_encoders = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_external_db():
    if ExternalSessionLocal is None:
        yield None
        return
    db = ExternalSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_origination_db():
    if OriginationSessionLocal is None:
        yield None
        return
    db = OriginationSessionLocal()
    try:
        yield db
    finally:
        db.close()

def find_or_create_customer(nida: str, local_db: Session, ext_db) -> models.Customer:
    customer = local_db.query(models.Customer).filter(models.Customer.nida == nida).first()
    if customer:
        print(f"[INFO] Customer {customer.customer_id} found locally.")
        return customer

    details = {}
    if ext_db is not None:
        print(f"[INFO] Fetching customer with NIDA {nida} from external DB...")
        query = text("""
            SELECT
                uuid            AS customer_id,
                first_name      AS first_name,
                last_name       AS surname,
                date_of_birth   AS date_of_birth
            FROM user_accounts
            WHERE nin = :nida_param
              AND deleted = false
            LIMIT 1
        """)
        try:
            result = ext_db.execute(query, {"nida_param": nida}).first()
            if result:
                details = dict(result._mapping)
                print(f"[INFO] Customer found in external DB: {details.get('customer_id')}")
            else:
                print(f"[WARNING] NIDA {nida} not found in external DB. Creating minimal record.")
        except Exception as e:
            print(f"[WARNING] External DB query failed: {e}. Creating minimal record.", file=sys.stderr)
    else:
        print(f"[WARNING] External DB unavailable. Creating minimal record for NIDA {nida}.")

    new_customer = models.Customer(
        nida=nida,
        customer_id=details.get("customer_id", f"cust-{nida[:12]}"),
        first_name=details.get("first_name", "Unknown"),
        surname=details.get("surname", "Unknown"),
        date_of_birth=details.get("date_of_birth"),
    )
    local_db.add(new_customer)
    local_db.commit()
    local_db.refresh(new_customer)
    print(f"[INFO] Customer {new_customer.customer_id} created locally.")
    return new_customer

def generate_new_inference() -> dict:
    print("[INFO] Generating mock credit inference...")
    score = random.randint(300, 850)
    monthly_income_usd = round(random.uniform(200, 2000), 2)
    decision, risk_category, risk_probability, limit_usd, interest_rate, validity_days = \
        services.apply_business_rules(score, monthly_income_usd)
    return {
        "credit_score": float(score),
        "risk_category": risk_category,
        "risk_probability": risk_probability,
        "decision": decision,
        "recommended_credit_limit": limit_usd,
        "suggested_interest_rate": interest_rate,
        "validity_period_days": validity_days,
    }

def get_production_model_version(model_name: str = "CreditScorePredictor") -> str:
    if MLFLOW_CLIENT is None:
        return "v-client-unavailable"
    try:
        versions = MLFLOW_CLIENT.get_latest_versions(model_name, stages=["Production"])
        return f"v{versions[0].version}" if versions else "v-N/A"
    except Exception:
        return "v-error"

CREDIT_SCORE_URI = os.getenv("MLFLOW_SCORE_MODEL_URI", "http://mlflow-server-score:8001")
CREDIT_RISK_URI = os.getenv("MLFLOW_RISK_MODEL_URI", "http://mlflow-server-risk:8002")
CREDIT_LIMIT_URI = os.getenv("MLFLOW_LIMIT_MODEL_URI", "http://mlflow-server-limit:8003")


def _invoke_mlflow_server(uri: str, features: dict):
    """Call an MLflow model server and return the first prediction, or None on failure."""
    try:
        resp = requests.post(
            f"{uri}/invocations",
            json={"dataframe_records": [features]},
            timeout=10,
        )
        if resp.status_code == 200:
            preds = resp.json().get("predictions", [])
            return preds[0] if preds else None
    except Exception as e:
        print(f"[WARNING] MLflow invocation at {uri} failed: {e}", file=sys.stderr)
    return None


def _get_customer_features(customer: models.Customer, uaa_db, origination_db) -> dict:
    """
    Fetch financial features from cms_uaa (income) and cms_origination (loan history).
    Falls back to defaults for any unavailable source.
    """
    features = {
        "age": 35, "married": "NO", "education": "Graduate", "dependents": 0,
        "employment_status": "Employed", "spouse_employment_status": "Unemployed",
        "monthly_income": round(random.uniform(200, 2000), 2),
        "residense_status": "Rented", "vehicle_ownership_status": "NO",
        "vehicle_cat": "None", "credit_history_length_months": 12,
        "payment_history_score": 500, "total_outstanding_debt": 0,
        "credit_utilization_ratio": 0.0, "number_of_late_payments_36": 0,
        "active_loans": 0, "avg_monthly_balance": 0, "savings_account_balance": 0,
        "requested_amount": 1000, "loan_purpose": "Personal",
        "previous_collateral_value": 0, "debt_to_income_ratio": 0.0,
    }

    # --- cms_uaa: income, age, credit_limit ---
    if uaa_db is not None:
        try:
            q = text("""
                SELECT
                    CASE WHEN date_of_birth IS NOT NULL AND date_of_birth != ''
                         THEN DATE_PART('year', AGE(date_of_birth::date))::int
                         ELSE 35 END                            AS age,
                    COALESCE(monthly_income, annual_income / 12, 500)::float  AS monthly_income,
                    COALESCE(annual_income, 0)::float                         AS annual_income,
                    COALESCE(credit_limit, 0)::float                          AS savings_account_balance
                FROM user_accounts
                WHERE uuid = :cid AND deleted = false
                LIMIT 1
            """)
            result = uaa_db.execute(q, {"cid": customer.customer_id}).first()
            if result:
                features.update({k: v for k, v in dict(result._mapping).items() if v is not None})
        except Exception as e:
            print(f"[WARNING] UAA features query failed: {e}", file=sys.stderr)

    # --- cms_origination: loan history, employment, collateral ---
    if origination_db is not None:
        try:
            q = text("""
                SELECT
                    COALESCE(ep.employment_type, 'Employed')                         AS employment_status,
                    COALESCE(ep.gross_salary_monthly, 0)::float                      AS avg_monthly_balance,
                    COALESCE(ep.duration_years * 12, 12)                             AS credit_history_length_months,
                    COUNT(DISTINCT la.application_id)
                        FILTER (WHERE la.status NOT IN ('REJECTED','CANCELLED'))     AS active_loans,
                    COALESCE(SUM(la.requested_amount)
                        FILTER (WHERE la.status NOT IN ('REJECTED','CANCELLED')), 0) AS total_outstanding_debt,
                    COALESCE(MAX(la.requested_amount), 1000)                         AS requested_amount,
                    COALESCE(MAX(la.loan_purpose), 'Personal')                       AS loan_purpose,
                    COALESCE(MAX(ci.estimated_value), 0)                             AS previous_collateral_value
                FROM loan_application la
                LEFT JOIN employment_profile ep ON ep.application_id = la.application_id
                LEFT JOIN collateral_item ci    ON ci.application_id = la.application_id
                WHERE la.borrower_party_id = :cid
                GROUP BY ep.employment_type, ep.gross_salary_monthly, ep.duration_years
                LIMIT 1
            """)
            result = origination_db.execute(q, {"cid": customer.customer_id}).first()
            if result:
                features.update({k: v for k, v in dict(result._mapping).items() if v is not None})
        except Exception as e:
            print(f"[WARNING] Origination features query failed: {e}", file=sys.stderr)

    monthly = float(features["monthly_income"])
    debt = float(features["total_outstanding_debt"])
    features["debt_to_income_ratio"] = round(debt / monthly, 4) if monthly > 0 else 0.0
    return features


def _build_credit_inference(features: dict) -> dict:
    """
    Call the 3 credit model servers. Falls back to random score for any unreachable server.
    """
    monthly_income = float(features.get("monthly_income", random.uniform(200, 2000)))

    raw_score = _invoke_mlflow_server(CREDIT_SCORE_URI, features)
    raw_risk = _invoke_mlflow_server(CREDIT_RISK_URI, features)
    raw_limit = _invoke_mlflow_server(CREDIT_LIMIT_URI, features)

    credit_score = float(raw_score) if raw_score is not None else float(random.randint(300, 850))
    decision, risk_category, risk_probability, limit_usd, interest_rate, validity_days = \
        services.apply_business_rules(credit_score, monthly_income)

    return {
        "credit_score": credit_score,
        "risk_category": str(raw_risk) if raw_risk is not None else risk_category,
        "risk_probability": risk_probability,
        "decision": decision,
        "recommended_credit_limit": float(raw_limit) if raw_limit is not None else limit_usd,
        "suggested_interest_rate": interest_rate,
        "validity_period_days": validity_days,
    }


@app.get("/")
def root():
    return {"message": "Welcome to Credit Scoring & Fraud Detection API!"}


# ─── Credit Scoring ───────────────────────────────────────────────────────────

@app.post("/predict", response_model=models.PredictionResponse, tags=["Credit Scoring"])
def predict(
    request: models.PredictionRequest,
    local_db: Session = Depends(get_db),
    ext_db: Session = Depends(get_external_db),
    origination_db: Session = Depends(get_origination_db),
):
    """
    Run a credit assessment for a customer identified by NIDA.
    Returns a cached result if still valid, otherwise runs a fresh inference.
    """
    customer = find_or_create_customer(request.nida, local_db, ext_db)
    today = date.today()

    cached = (
        local_db.query(models.CachedInference)
        .filter(
            models.CachedInference.customer_id == customer.customer_id,
            models.CachedInference.end_inference_date >= today,
        )
        .order_by(models.CachedInference.last_inference_date.desc())
        .first()
    )

    if cached:
        print(f"[INFO] Returning cached inference for customer {customer.customer_id}")
        inference = {
            "credit_score": float(cached.credit_score),
            "risk_category": cached.risk_category,
            "risk_probability": float(cached.risk_probability),
            "decision": cached.decision,
            "recommended_credit_limit": float(cached.recommended_credit_limit),
            "suggested_interest_rate": float(cached.suggested_interest_rate),
            "validity_period_days": cached.validity_period_days,
            "model_version": cached.model_version,
        }
    else:
        features = _get_customer_features(customer, ext_db, origination_db)
        inference = _build_credit_inference(features)
        model_version = get_production_model_version("CreditScorePredictor")
        inference["model_version"] = model_version

        new_inference = models.CachedInference(
            customer_id=customer.customer_id,
            credit_score=inference["credit_score"],
            decision=inference["decision"],
            recommended_credit_limit=inference["recommended_credit_limit"],
            suggested_interest_rate=inference["suggested_interest_rate"],
            risk_category=inference["risk_category"],
            risk_probability=inference["risk_probability"],
            validity_period_days=inference["validity_period_days"],
            last_inference_date=today,
            end_inference_date=today + timedelta(days=inference["validity_period_days"]),
            model_version=model_version,
        )
        local_db.add(new_inference)
        local_db.commit()
        print(f"[INFO] New inference created for customer {customer.customer_id}")

    active_loans = (
        local_db.query(models.Loan)
        .filter(
            models.Loan.customer_id == customer.customer_id,
            models.Loan.status == "ACTIVE",
        )
        .all()
    )
    outstanding = sum(float(loan.outstanding_balance) for loan in active_loans)
    credit_limit = float(inference["recommended_credit_limit"]) * USD_TO_TZS_RATE
    spending_limit = max(0.0, credit_limit - outstanding)

    return models.PredictionResponse(
        customer_id=customer.customer_id,
        request_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        credit_score=inference["credit_score"],
        risk_category=inference["risk_category"],
        risk_probability=inference["risk_probability"],
        decision=inference["decision"],
        recommended_credit_limit=credit_limit,
        currency="TZS",
        spending_limit=spending_limit,
        outstanding_balance=outstanding,
        suggested_interest_rate=inference["suggested_interest_rate"],
        validity_period=f"{inference['validity_period_days']} days",
        model_version=inference["model_version"],
    )


@app.get("/status/{customer_id}", response_model=models.StatusResponse, tags=["Credit Scoring"])
def status(customer_id: str, local_db: Session = Depends(get_db)):
    """Return the current credit status and active loans for a customer."""
    customer = local_db.query(models.Customer).filter(
        models.Customer.customer_id == customer_id
    ).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")

    today = date.today()
    inference = (
        local_db.query(models.CachedInference)
        .filter(
            models.CachedInference.customer_id == customer_id,
            models.CachedInference.end_inference_date >= today,
        )
        .order_by(models.CachedInference.last_inference_date.desc())
        .first()
    )

    active_loans = (
        local_db.query(models.Loan)
        .filter(models.Loan.customer_id == customer_id, models.Loan.status == "ACTIVE")
        .all()
    )
    outstanding = sum(float(loan.outstanding_balance) for loan in active_loans)

    credit_info = None
    if inference:
        credit_limit = float(inference.recommended_credit_limit) * USD_TO_TZS_RATE
        credit_info = {
            "credit_score": float(inference.credit_score),
            "decision": inference.decision,
            "risk_category": inference.risk_category,
            "recommended_credit_limit": credit_limit,
            "spending_limit": max(0.0, credit_limit - outstanding),
            "suggested_interest_rate": float(inference.suggested_interest_rate),
            "valid_until": inference.end_inference_date.isoformat(),
            "model_version": inference.model_version,
        }

    return models.StatusResponse(
        message="Customer status retrieved.",
        status="FOUND",
        detail={
            "customer_id": customer.customer_id,
            "name": f"{customer.first_name} {customer.surname}",
            "outstanding_balance": outstanding,
            "active_loan_count": len(active_loans),
            "active_loans": [
                {
                    "loan_ref": loan.loan_ref,
                    "disbursed_amount": float(loan.disbursed_amount),
                    "outstanding_balance": float(loan.outstanding_balance),
                    "disbursal_date": loan.disbursal_date.isoformat() if loan.disbursal_date else None,
                }
                for loan in active_loans
            ],
            "credit_assessment": credit_info,
        },
    )


# ─── Loan Management ──────────────────────────────────────────────────────────

@app.post("/disburse", response_model=models.StatusResponse, tags=["Loan Management"])
def disburse(request: models.DisburseRequest, local_db: Session = Depends(get_db)):
    """Disburse a loan to an approved customer against their valid credit assessment."""
    customer = local_db.query(models.Customer).filter(
        models.Customer.customer_id == request.customer_id
    ).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {request.customer_id} not found.")

    if local_db.query(models.Loan).filter(models.Loan.loan_ref == request.loan_ref).first():
        raise HTTPException(status_code=409, detail=f"Loan reference {request.loan_ref} already exists.")

    today = date.today()
    inference = (
        local_db.query(models.CachedInference)
        .filter(
            models.CachedInference.customer_id == request.customer_id,
            models.CachedInference.end_inference_date >= today,
            models.CachedInference.decision == "APPROVED",
        )
        .order_by(models.CachedInference.last_inference_date.desc())
        .first()
    )
    if not inference:
        raise HTTPException(
            status_code=403,
            detail="No valid approved credit assessment found. Run /predict first.",
        )

    active_loans = (
        local_db.query(models.Loan)
        .filter(models.Loan.customer_id == request.customer_id, models.Loan.status == "ACTIVE")
        .all()
    )
    outstanding = sum(float(loan.outstanding_balance) for loan in active_loans)
    available = float(inference.recommended_credit_limit) * USD_TO_TZS_RATE - outstanding

    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Disbursement amount must be positive.")
    if request.amount > available:
        raise HTTPException(
            status_code=403,
            detail=f"Requested {request.amount} exceeds available limit {round(available, 2)}.",
        )

    loan = models.Loan(
        loan_ref=request.loan_ref,
        customer_id=request.customer_id,
        inference_id=inference.id,
        disbursed_amount=Decimal(str(request.amount)),
        outstanding_balance=Decimal(str(request.amount)),
        disbursal_date=datetime.utcnow(),
        status="ACTIVE",
    )
    local_db.add(loan)
    local_db.commit()
    local_db.refresh(loan)

    return models.StatusResponse(
        message="Loan disbursed successfully.",
        status="ACTIVE",
        detail={
            "loan_ref": loan.loan_ref,
            "customer_id": loan.customer_id,
            "disbursed_amount": float(loan.disbursed_amount),
            "outstanding_balance": float(loan.outstanding_balance),
            "disbursal_date": loan.disbursal_date.isoformat(),
        },
    )


@app.post("/repay", response_model=models.StatusResponse, tags=["Loan Management"])
def repay(request: models.RepayRequest, local_db: Session = Depends(get_db)):
    """Record a repayment against an active loan."""
    loan = (
        local_db.query(models.Loan)
        .filter(
            models.Loan.loan_ref == request.loan_ref,
            models.Loan.customer_id == request.customer_id,
        )
        .first()
    )
    if not loan:
        raise HTTPException(status_code=404, detail=f"Loan {request.loan_ref} not found for this customer.")
    if loan.status == "SETTLED":
        raise HTTPException(status_code=400, detail=f"Loan {request.loan_ref} is already fully settled.")
    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Repayment amount must be positive.")

    local_db.add(models.Repayment(
        loan_ref=request.loan_ref,
        amount=Decimal(str(request.amount)),
        payment_date=datetime.utcnow(),
    ))

    new_balance = max(0.0, float(loan.outstanding_balance) - request.amount)
    loan.outstanding_balance = Decimal(str(new_balance))
    if new_balance == 0.0:
        loan.status = "SETTLED"

    local_db.commit()
    local_db.refresh(loan)

    return models.StatusResponse(
        message="Repayment recorded successfully.",
        status=loan.status,
        detail={
            "loan_ref": loan.loan_ref,
            "repayment_amount": request.amount,
            "outstanding_balance": float(loan.outstanding_balance),
            "loan_status": loan.status,
        },
    )