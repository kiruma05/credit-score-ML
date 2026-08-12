import os
import subprocess
import uuid
import random
import sys
from datetime import datetime, date, timedelta
from urllib.parse import quote_plus
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
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
from app.auth import require_api_auth
from app.auth_router import router as auth_router
from app.explainer import load_model_for_shap, explain_prediction, summarize_drivers
from app.features import fetch_features
from app.middleware import RequestIDMiddleware
from app.utils.response import (
    Envelope,
    error_response,
    paginated,
    success_response,
)

app = FastAPI(
    title="Credit Scoring & Fraud Detection API",
    description="Unified API for credit scoring, fraud detection, and loan lifecycle management.",
    version="5.5.4",
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)
app.add_middleware(RequestIDMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Wrap HTTPException so every error response shares the envelope shape.

    Endpoints can raise either a plain detail string or a structured
    ``{"code": "...", "message": "...", "details": {...}}`` payload — we unpack
    the structured form when available.
    """
    code = "HTTP_ERROR"
    message = "Request failed"
    details = None
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code", code)
        message = exc.detail.get("message", message)
        details = exc.detail.get("details")
    elif isinstance(exc.detail, str):
        message = exc.detail
    return error_response(code, message, request, details=details, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_response(
        code="VALIDATION_ERROR",
        message="Request validation failed",
        request=request,
        details=exc.errors(),
        status_code=422,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] Unhandled exception: {exc}", file=sys.stderr)
    return error_response(
        code="INTERNAL_ERROR",
        message="An unexpected error occurred",
        request=request,
        status_code=500,
    )


app.include_router(auth_router, prefix="/auth")
app.include_router(
    fraud_router,
    prefix="/fraud",
    tags=["Fraud Detection"],
    dependencies=[Depends(require_api_auth)],
)

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
    """Resolve a customer by NIDA.

    In strict mode (``SCORING_STRICT_MODE=true``) the NIDA MUST exist in
    cms_uaa — we refuse to create a local stub because doing so would (a)
    poison local state with a fake customer_id and (b) prevent later resolution
    if cms_uaa is populated for that NIDA afterwards.

    In demo mode we fall back to a deterministic stub so the seeded synthetic
    scoring path can still run.
    """
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
                print(f"[WARNING] NIDA {nida} not found in external DB.")
        except Exception as e:
            print(f"[WARNING] External DB query failed: {e}.", file=sys.stderr)
    else:
        print(f"[WARNING] External DB unavailable for NIDA {nida}.")

    if not details:
        if SCORING_STRICT_MODE:
            print(
                f"[REJECT] NIDA={nida} not in cms_uaa — refusing to create local stub (strict mode).",
                flush=True,
            )
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "CUSTOMER_DATA_MISSING",
                    "message": "Customer record not found in source systems",
                    "details": {
                        "nida": nida,
                        "checked_sources": ["cms_uaa"],
                        "hint": "Verify the NIDA exists in user_accounts (cms_uaa), or set SCORING_STRICT_MODE=false for demo mode.",
                    },
                },
            )
        print(f"[INFO] Demo mode — creating local stub for NIDA {nida}.")

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


SCORING_STRICT_MODE = os.getenv("SCORING_STRICT_MODE", "true").lower() == "true"


def _seed_synthetic_features(features: dict, nida: str) -> None:
    """Overwrite default features with deterministic, NIDA-seeded values.

    Same NIDA always yields the same features (and therefore the same score),
    different NIDAs yield different feature vectors. Used only when external
    customer data is unavailable AND strict mode is disabled.
    """
    rng = random.Random(hash(nida))
    monthly_income = round(rng.uniform(200, 2000), 2)
    features.update({
        "monthly_income": monthly_income,
        "payment_history_score": rng.randint(300, 750),
        "credit_history_length_months": rng.randint(6, 60),
        "active_loans": rng.randint(0, 3),
        "number_of_late_payments_36": rng.randint(0, 5),
        "total_outstanding_debt": round(rng.uniform(0, monthly_income * 6), 2),
        "credit_utilization_ratio": round(rng.uniform(0, 1), 4),
        "avg_monthly_balance": round(rng.uniform(0, monthly_income * 1.2), 2),
        "savings_account_balance": round(rng.uniform(0, monthly_income * 3), 2),
        "requested_amount": rng.randint(500, 5000),
        "age": rng.randint(21, 65),
    })


def _get_customer_features(customer: models.Customer, uaa_db, origination_db):
    """Resolve the 22 model features for a customer.

    Thin wrapper over ``features.fetch_features``. Adds strict-mode rejection
    and the demo-mode seeded fallback that the new module deliberately does
    not handle (so it remains testable in isolation).
    """
    feats, data_quality = fetch_features(customer.nida, uaa_db, origination_db)

    if data_quality["uaa_source"] == "fallback":
        if SCORING_STRICT_MODE:
            print(
                f"[REJECT] NIDA={customer.nida} customer_id={customer.customer_id} "
                f"not found in cms_uaa — refusing to score (strict mode).",
                flush=True,
            )
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "CUSTOMER_DATA_MISSING",
                    "message": "Customer record not found in source systems",
                    "details": {
                        "nida": customer.nida,
                        "customer_id": customer.customer_id,
                        "checked_sources": ["cms_uaa", "cms_origination"],
                        "hint": "Verify the NIDA exists in user_accounts, or set SCORING_STRICT_MODE=false for demo mode.",
                    },
                },
            )
        print(
            f"[FALLBACK] NIDA={customer.nida} customer_id={customer.customer_id} "
            f"→ seeded synthetic features (demo mode).",
            flush=True,
        )
        _seed_synthetic_features(feats, customer.nida)
    elif data_quality["score_basis"] != "live_data":
        print(
            f"[PARTIAL] NIDA={customer.nida} customer_id={customer.customer_id} "
            f"score_basis={data_quality['score_basis']} apps={data_quality['applications_found']} "
            f"employment={data_quality['employment_found']}.",
            flush=True,
        )

    return feats, data_quality


def _build_credit_inference(features: dict) -> dict:
    """Call the 3 credit model servers + apply business-rule guard rails."""
    monthly_income = float(features.get("monthly_income", random.uniform(200, 2000)))
    dti = float(features.get("debt_to_income_ratio", 0.0))
    active_loans = int(features.get("active_loans", 0))

    raw_score = _invoke_mlflow_server(CREDIT_SCORE_URI, features)
    raw_risk = _invoke_mlflow_server(CREDIT_RISK_URI, features)
    raw_limit = _invoke_mlflow_server(CREDIT_LIMIT_URI, features)

    credit_score = float(raw_score) if raw_score is not None else float(random.randint(300, 850))
    decision, risk_category, risk_probability, limit_usd, interest_rate, validity_days = \
        services.apply_business_rules(credit_score, monthly_income, dti, active_loans)

    return {
        "credit_score": credit_score,
        "risk_category": str(raw_risk) if raw_risk is not None else risk_category,
        "risk_probability": risk_probability,
        "decision": decision,
        "recommended_credit_limit": float(raw_limit) if raw_limit is not None else limit_usd,
        "suggested_interest_rate": interest_rate,
        "validity_period_days": validity_days,
    }


@app.get("/", response_model=Envelope[dict])
def root(http_request: Request):
    return success_response(
        data={"service": "Credit Scoring & Fraud Detection API", "version": app.version},
        message="Welcome to Credit Scoring & Fraud Detection API!",
        request=http_request,
    )


@app.get("/health", response_model=Envelope[dict], tags=["System"])
def health(http_request: Request):
    """Liveness + dependency probe.

    Returns the connectivity status of every upstream the scoring path relies
    on (postgres, cms_uaa, cms_origination, MLflow). Use this to verify your
    .env values before calling /predict.
    """
    checks = {
        "postgres": "unknown",
        "cms_uaa": "unknown",
        "cms_origination": "unknown",
        "mlflow_tracking": "unknown",
        "strict_mode": str(SCORING_STRICT_MODE).lower(),
    }

    # Local PostgreSQL (where customer/loan state lives)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {type(e).__name__}"

    # cms_uaa
    if ExternalSessionLocal is None:
        checks["cms_uaa"] = "not_configured"
    else:
        try:
            db = ExternalSessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            checks["cms_uaa"] = "ok"
        except Exception as e:
            checks["cms_uaa"] = f"error: {type(e).__name__}"

    # cms_origination
    if OriginationSessionLocal is None:
        checks["cms_origination"] = "not_configured"
    else:
        try:
            db = OriginationSessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            checks["cms_origination"] = "ok"
        except Exception as e:
            checks["cms_origination"] = f"error: {type(e).__name__}"

    # MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        checks["mlflow_tracking"] = "not_configured"
    else:
        try:
            r = requests.get(f"{tracking_uri}/health", timeout=3)
            checks["mlflow_tracking"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception as e:
            checks["mlflow_tracking"] = f"error: {type(e).__name__}"

    everything_ok = all(
        v == "ok" or v == "not_configured" or v.startswith("true") or v.startswith("false")
        for v in checks.values()
    )
    return success_response(
        data=checks,
        message="All checks passed" if everything_ok else "One or more dependencies are unhealthy",
        request=http_request,
        status_code=200 if everything_ok else 503,
    )


# ─── Admin ────────────────────────────────────────────────────────────────────

_TRAIN_SCRIPT_CANDIDATES = (
    "/app/airflow_jobs/train_models_pandas.py",  # mounted via docker-compose
    "/opt/airflow/jobs/train_models_pandas.py",  # alternate mount
)


@app.post("/admin/retrain", response_model=Envelope[dict], tags=["Admin"])
def admin_retrain(
    http_request: Request,
    authorization: Optional[str] = Header(None),
):
    """Trigger an asynchronous retrain of all 3 credit models.

    The training script runs as a background subprocess and writes to MLflow
    on completion. Returns immediately with the PID so callers can poll
    MLflow / inspect logs separately.
    """
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token:
        raise HTTPException(
            status_code=503,
            detail={"code": "ADMIN_TOKEN_NOT_CONFIGURED",
                    "message": "Admin token not configured on server."},
        )
    if not authorization or authorization != f"Bearer {admin_token}":
        raise HTTPException(
            status_code=401,
            detail={"code": "ADMIN_AUTH_FAILED",
                    "message": "Invalid or missing admin token."},
        )

    script = next((p for p in _TRAIN_SCRIPT_CANDIDATES if os.path.exists(p)), None)
    if script is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "TRAIN_SCRIPT_NOT_MOUNTED",
                    "message": "Training script not found in container.",
                    "details": {"checked": list(_TRAIN_SCRIPT_CANDIDATES)}},
        )

    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    log_path = f"/tmp/retrain-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.log"
    proc = subprocess.Popen(
        ["python", script],
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"[INFO] Retraining started: pid={proc.pid} script={script} log={log_path}", flush=True)
    return success_response(
        data={
            "job_id": proc.pid,
            "status": "started",
            "script": script,
            "log_path": log_path,
            "hint": "Watch MLflow UI for new model versions; promote to Production manually when ready.",
        },
        message="Retraining triggered",
        request=http_request,
        status_code=202,
    )


# ─── Credit Scoring ───────────────────────────────────────────────────────────

@app.post("/predict", response_model=Envelope[models.PredictionResponse], tags=["Credit Scoring"])
def predict(
    request: models.PredictionRequest,
    http_request: Request,
    local_db: Session = Depends(get_db),
    ext_db: Session = Depends(get_external_db),
    origination_db: Session = Depends(get_origination_db),
    _auth: models.ApiClient = Depends(require_api_auth),
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

    data_quality = None

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
        # Anything in the cache is live data (we never persist fallback scores).
        data_quality = {
            "uaa_source": "live",
            "origination_source": "live",
            "score_basis": "live_data_cached",
        }
    else:
        features, data_quality = _get_customer_features(customer, ext_db, origination_db)
        inference = _build_credit_inference(features)
        model_version = get_production_model_version("CreditScorePredictor")
        inference["model_version"] = model_version

        # Only cache scores grounded in live data — fallback scores must re-run
        # on every call so the moment real cms_uaa data lands, scoring refreshes.
        if data_quality["score_basis"] == "live_data":
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
            print(f"[INFO] New inference cached for customer {customer.customer_id}")
        else:
            print(
                f"[INFO] Inference for {customer.customer_id} not cached "
                f"(score_basis={data_quality['score_basis']})."
            )

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

    prediction = models.PredictionResponse(
        customer_id=customer.customer_id,
        request_id=http_request.state.request_id,
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
        data_quality=data_quality,
    )
    return success_response(
        data=prediction,
        message="Credit assessment completed",
        request=http_request,
    )


@app.post("/explain", response_model=Envelope[dict], tags=["Credit Scoring"])
def explain(
    request: models.PredictionRequest,
    http_request: Request,
    local_db: Session = Depends(get_db),
    ext_db: Session = Depends(get_external_db),
    origination_db: Session = Depends(get_origination_db),
    _auth: models.ApiClient = Depends(require_api_auth),
):
    """
    Run a credit assessment and return a full SHAP explanation of the score.
    Shows which features drove the credit score up or down for this customer.
    """
    customer = find_or_create_customer(request.nida, local_db, ext_db)
    features, data_quality = _get_customer_features(customer, ext_db, origination_db)
    inference = _build_credit_inference(features)
    model_version = get_production_model_version("CreditScorePredictor")

    # Load model from MLflow registry for SHAP (bypasses serving endpoint)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    explanation_result = {
        "method": "unavailable",
        "base_score": None,
        "top_helps": [],
        "top_hurts": [],
    }
    explanation_error = None

    shap_model = load_model_for_shap("CreditScorePredictor", tracking_uri)
    if shap_model is not None:
        # 7 features each side → 14 SHAP-ranked factors per decision
        explanation_result = explain_prediction(
            shap_model, features, top_n_helps=7, top_n_hurts=7
        )
    else:
        explanation_error = "Model registry unavailable — SHAP explanation skipped."

    explanation_result["summary"] = summarize_drivers(
        explanation_result.get("top_helps", []),
        explanation_result.get("top_hurts", []),
    )

    payload = {
        "customer_id": customer.customer_id,
        "request_id": http_request.state.request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "credit_score": inference["credit_score"],
        "risk_category": inference["risk_category"],
        "risk_probability": inference["risk_probability"],
        "decision": inference["decision"],
        "recommended_credit_limit": inference["recommended_credit_limit"],
        "suggested_interest_rate": inference["suggested_interest_rate"],
        "validity_period": f"{inference['validity_period_days']} days",
        "model_version": model_version,
        "data_quality": data_quality,
        "explanation": explanation_result if not explanation_error else None,
        "explanation_error": explanation_error,
    }
    return success_response(
        data=payload,
        message="Credit assessment with SHAP explanation completed",
        request=http_request,
    )


@app.get("/status/{customer_id}", response_model=Envelope[models.StatusResponse], tags=["Credit Scoring"])
def status(
    customer_id: str,
    http_request: Request,
    local_db: Session = Depends(get_db),
    _auth: models.ApiClient = Depends(require_api_auth),
):
    """Return the current credit status and active loans for a customer."""
    customer = local_db.query(models.Customer).filter(
        models.Customer.customer_id == customer_id
    ).first()
    if not customer:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "CUSTOMER_NOT_FOUND",
                "message": f"Customer {customer_id} not found.",
                "details": {"customer_id": customer_id},
            },
        )

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

    return success_response(
        data=models.StatusResponse(
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
        ),
        message="Customer status retrieved",
        request=http_request,
    )


# ─── Loan Management ──────────────────────────────────────────────────────────

@app.post("/disburse", response_model=Envelope[models.StatusResponse], tags=["Loan Management"])
def disburse(
    request: models.DisburseRequest,
    http_request: Request,
    local_db: Session = Depends(get_db),
    _auth: models.ApiClient = Depends(require_api_auth),
):
    """Disburse a loan to an approved customer against their valid credit assessment."""
    customer = local_db.query(models.Customer).filter(
        models.Customer.customer_id == request.customer_id
    ).first()
    if not customer:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "CUSTOMER_NOT_FOUND",
                "message": f"Customer {request.customer_id} not found.",
                "details": {"customer_id": request.customer_id},
            },
        )

    if local_db.query(models.Loan).filter(models.Loan.loan_ref == request.loan_ref).first():
        raise HTTPException(
            status_code=409,
            detail={
                "code": "LOAN_REF_CONFLICT",
                "message": f"Loan reference {request.loan_ref} already exists.",
                "details": {"loan_ref": request.loan_ref},
            },
        )

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
            detail={
                "code": "NO_VALID_ASSESSMENT",
                "message": "No valid approved credit assessment found. Run /predict first.",
            },
        )

    active_loans = (
        local_db.query(models.Loan)
        .filter(models.Loan.customer_id == request.customer_id, models.Loan.status == "ACTIVE")
        .all()
    )
    outstanding = sum(float(loan.outstanding_balance) for loan in active_loans)
    available = float(inference.recommended_credit_limit) * USD_TO_TZS_RATE - outstanding

    if request.amount <= 0:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_AMOUNT", "message": "Disbursement amount must be positive."},
        )
    if request.amount > available:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "LIMIT_EXCEEDED",
                "message": f"Requested {request.amount} exceeds available limit {round(available, 2)}.",
                "details": {"requested": request.amount, "available": round(available, 2)},
            },
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

    return success_response(
        data=models.StatusResponse(
            message="Loan disbursed successfully.",
            status="ACTIVE",
            detail={
                "loan_ref": loan.loan_ref,
                "customer_id": loan.customer_id,
                "disbursed_amount": float(loan.disbursed_amount),
                "outstanding_balance": float(loan.outstanding_balance),
                "disbursal_date": loan.disbursal_date.isoformat(),
            },
        ),
        message="Loan disbursed successfully",
        request=http_request,
        status_code=201,
    )


@app.post("/repay", response_model=Envelope[models.StatusResponse], tags=["Loan Management"])
def repay(
    request: models.RepayRequest,
    http_request: Request,
    local_db: Session = Depends(get_db),
    _auth: models.ApiClient = Depends(require_api_auth),
):
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
        raise HTTPException(
            status_code=404,
            detail={
                "code": "LOAN_NOT_FOUND",
                "message": f"Loan {request.loan_ref} not found for this customer.",
                "details": {"loan_ref": request.loan_ref, "customer_id": request.customer_id},
            },
        )
    if loan.status == "SETTLED":
        raise HTTPException(
            status_code=400,
            detail={
                "code": "LOAN_ALREADY_SETTLED",
                "message": f"Loan {request.loan_ref} is already fully settled.",
                "details": {"loan_ref": request.loan_ref},
            },
        )
    if request.amount <= 0:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_AMOUNT", "message": "Repayment amount must be positive."},
        )

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

    return success_response(
        data=models.StatusResponse(
            message="Repayment recorded successfully.",
            status=loan.status,
            detail={
                "loan_ref": loan.loan_ref,
                "repayment_amount": request.amount,
                "outstanding_balance": float(loan.outstanding_balance),
                "loan_status": loan.status,
            },
        ),
        message="Repayment recorded successfully",
        request=http_request,
    )