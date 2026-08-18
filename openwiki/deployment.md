---
okf_version: "0.1"
type: concept
title: Deployment & Configuration
---

# Deployment & Configuration

The whole stack is defined in `docker-compose.yml` and configured via a `.env`
file (template: `.env.example`). The API image is built from `./fastapi`,
Airflow from `./airflow`, MLflow servers from `./mlflow`.

## Running locally

1. Copy `.env.example` → `.env` and fill in the values (see below).
2. TLS: the API serves HTTPS with a self-signed cert; place `cert.pem`/`key.pem`
   in `./certs` (mounted to `/app/certs`).
3. `docker compose up --build`.
4. Reach the services on their host ports (see [Architecture](architecture.md)):
   - API: `https://localhost:18008` (`/docs` for Swagger)
   - Airflow UI: `http://localhost:18087`
   - MLflow: `http://localhost:15000`
   - MinIO console: `http://localhost:19001`

The FastAPI container command runs
`uvicorn app.main:app --host 0.0.0.0 --port 8000` with `--ssl-keyfile` /
`--ssl-certfile`. Postgres databases are initialized by `scripts/init-dbs.sh`;
MinIO buckets by the one-shot `mc-create-bucket` service.

## Environment variables (`.env`)

Grouped by concern (keys from `.env.example`):

- **Postgres:** `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- **Airflow:** `AIRFLOW_DB_USER/PASSWORD/NAME`, `AIRFLOW_USER`, `AIRFLOW_PASSWORD`
- **MLflow backend:** `MLFLOW_DB_USER/PASSWORD/NAME`
- **Object store (MinIO):** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- **External `cms_uaa` DB:** `EXTERNAL_DB_HOST/PORT/NAME/USER/PASSWORD`
- **External `cms_origination` DB:** `ORIGINATION_DB_HOST/PORT/NAME/USER/PASSWORD`
- **Scoring behavior:** `SCORING_STRICT_MODE`
- **Auth:** `DISABLE_AUTH`, `ADMIN_TOKEN`, `ENFORCE_KEY_EXPIRY`, `REQUIRE_HTTPS`, `ALLOWED_ORIGINS`
- **Model serving:** `MLFLOW_TRACKING_URI`, `MLFLOW_SCORE_MODEL_URI`,
  `MLFLOW_RISK_MODEL_URI`, `MLFLOW_LIMIT_MODEL_URI`, `MLFLOW_FRAUD_MODEL_URI`,
  `FRAUD_MODEL_NAME`, `FRAUD_MODEL_VERSION`, `ENCODER_PATH`

> The app-internal `DATABASE_URL` is composed in `docker-compose.yml` from the
> MLflow DB credentials and points at the `postgres` service.

## CI/CD & security tooling

- **`Jenkinsfile-dev`** — the dev pipeline. The server pulls from the GitHub
  remote for deployment.
- **`trivy-docker-image-scan.sh`** — Trivy image vulnerability scan.
- **`opa-docker-security.rego`** — OPA policy for Docker security checks.
- **`.github/workflows/`** — GitHub Actions workflows.
- Helpers: `wait-for-it.sh` (service readiness), `start-mlflow.sh`.

## Operational notes

- **Strict vs demo scoring:** with `SCORING_STRICT_MODE=true`, unknown NIDAs are
  rejected (`404`); with it off, synthetic features are seeded for demos. See
  [Credit Scoring Engine](credit-scoring.md).
- **Auth in dev:** `DISABLE_AUTH=true` bypasses the API key/secret check — never
  use in production. See [Authentication](authentication.md).

## Cross-references

- Full service/port table: [Architecture & Services](architecture.md)
- Data stores: [Data Model](data-model.md)
