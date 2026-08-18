---
okf_version: "0.1"
type: concept
title: API Reference
---

# API Reference

All endpoints are defined in `fastapi/app/main.py` unless noted, and every
response is wrapped in a standard `Envelope` (`fastapi/app/utils/response.py`)
carrying `success`, `message`, `data`, and a request id. Business endpoints
require API key/secret headers — see [Authentication](authentication.md).

## System

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | none | Root/service banner. |
| `GET` | `/health` | none | Liveness + dependency probe. |

## Credit Scoring (`tags=["Credit Scoring"]`)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/predict` | ✅ | Run a credit assessment for a customer by **NIDA**. Returns a cached result if still valid, otherwise runs a fresh inference. → [Credit Scoring Engine](credit-scoring.md) |
| `POST` | `/explain` | ✅ | Run an assessment and return a **SHAP** explanation of which features drove the score up/down (`fastapi/app/explainer.py`). |
| `GET` | `/status/{customer_id}` | ✅ | Current credit status and active loans for a customer. |

### `POST /predict`

- **Request:** `PredictionRequest { nida: str }`
- **Response:** `PredictionResponse` — `credit_score`, `risk_category`,
  `risk_probability`, `decision`, `recommended_credit_limit` (TZS),
  `spending_limit`, `outstanding_balance`, `suggested_interest_rate`,
  `validity_period`, `model_version`, and a `data_quality` block describing
  whether the score is based on `live` or `fallback` data.

## Loan Management (`tags=["Loan Management"]`)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/disburse` | ✅ | Disburse a loan to an approved customer against a **valid** credit assessment. Request: `DisburseRequest { customer_id, loan_ref, amount }`. |
| `POST` | `/repay` | ✅ | Record a repayment against an active loan. Request: `RepayRequest { customer_id, loan_ref, amount }`. |

## Admin (`tags=["Admin"]`)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/admin/retrain` | admin token | Trigger an asynchronous retrain of all three credit models. |

## Fraud (`fastapi/app/fraud/fraud_router.py`)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/fraud/predict` | ✅ | Score a transaction for fraud. Returns `FraudPrediction`. → [Fraud Detection](fraud-detection.md) |

## Auth Management (`fastapi/app/auth_router.py`, `tags=["Auth Management"]`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/clients` | Create an API client; returns the key + one-time secret (`CreateClientResponse`). |
| `GET` | `/clients` | List API clients (`ClientSummary`). |
| `DELETE` | `/clients/{api_key}` | Revoke/delete an API client. |

## Cross-references

- Request/response shapes: [Data Model](data-model.md)
- Header auth and dev toggles: [Authentication](authentication.md)
