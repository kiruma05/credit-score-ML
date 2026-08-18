---
okf_version: "0.1"
type: index
title: Credit Scoring & Fraud Detection Platform
---

# Credit Scoring & Fraud Detection Platform

A containerized platform that turns a customer's national ID (**NIDA**) into a
**credit decision** — score, risk band, recommended limit, and interest rate —
and separately flags **fraudulent transactions**. It combines a FastAPI service,
three credit ML models plus one fraud model served through MLflow, an
Airflow/Spark training stack, and PostgreSQL + MinIO for storage.

- **Service name:** `Credit Scoring & Fraud Detection API` (FastAPI, version `5.5.4`)
- **Primary entry point:** `fastapi/app/main.py`
- **Exposed at:** `https://localhost:18008` (self-signed TLS → container port 8000)
- **Interactive API docs:** `https://localhost:18008/docs` (Swagger) and `/redoc`

## What the platform does

1. A client calls `POST /predict` with a NIDA.
2. The API resolves that customer's features from two external banking databases
   (`cms_uaa`, `cms_origination`).
3. Three MLflow-served models produce a **score**, a **risk** label, and a
   **limit**; a business-rules engine applies guard rails and tiering.
4. The result is cached for a validity window and returned as a credit decision.
5. Loans can then be disbursed and repaid against that decision, and
   transactions can be screened for fraud via `POST /fraud/predict`.

## Documentation map

- [Architecture & Services](architecture.md) — every container, port, and how requests flow.
- [API Reference](api-reference.md) — all HTTP endpoints, grouped by domain.
- [Credit Scoring Engine](credit-scoring.md) — features → 3 models → business rules → decision.
- [Scoring Engine — Algorithm, Features & Assumptions](scoring-engine-algorithm.md) — the hybrid ML+rules design, why these algorithms, and the assumptions.
- [Model Risk & Weaknesses](model-risk-and-weaknesses.md) — critical review of the engine's weaknesses and a prioritized remediation checklist.
- [Fraud Detection](fraud-detection.md) — the fraud model, loading strategy, and endpoint.
- [Data Model](data-model.md) — PostgreSQL tables, API schemas, and the external DBs.
- [Authentication](authentication.md) — API key/secret model and dev/prod toggles.
- [ML Pipeline (MLflow · Airflow · Spark)](ml-pipeline.md) — training, retraining, serving, and PSI monitoring. (alias: [Training & Registry](training-and-registry.md))
- [Deployment & Configuration](deployment.md) — docker-compose, environment, and CI/CD. (alias: [Runtime Topology](runtime.md))
- **Governance:** per-model records + feature data dictionary live in `governance/` (see `governance/README.md`).

> This wiki was authored by reading the source directly. Files are cited by path
> so you can jump straight to the code.
