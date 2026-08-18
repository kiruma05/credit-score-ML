---
type: runtime architecture
title: Container Runtime and Dependencies
description: Docker Compose topology for the credit scoring platform, including service startup order, persistence, model serving, and configuration boundaries.
tags: [architecture, docker-compose, operations]
---

# Container Runtime and Dependencies

`docker-compose.yml` is the authoritative local composition root. It runs the API alongside a PostgreSQL-backed ML platform: Airflow schedules jobs, Spark executes distributed jobs, MLflow tracks/registers models, MinIO stores MLflow artifacts, and four MLflow serving processes expose production models to the API.

```mermaid
flowchart TD
  API["fastapi-app"] --> PG["PostgreSQL"]
  API --> UAA["cms_uaa external database"]
  API --> ORIG["cms_origination external database"]
  API --> SCORE["CreditScorePredictor server"]
  API --> RISK["RiskCategoryPredictor server"]
  API --> LIMIT["CreditLimitPredictor server"]
  API --> FRAUD["FraudDetector server"]
  SCORE --> MLF["MLflow tracking server"]
  RISK --> MLF
  LIMIT --> MLF
  FRAUD --> MLF
  AIR["Airflow scheduler and webserver"] --> MLF
  AIR --> SPARK["Spark master and workers"]
  MLF --> PG
  MLF --> MINIO["MinIO artifact store"]
```

This shows the Compose-declared dependencies and the application call paths. The two external databases are configured connections, not Compose services.

## Service map

| Component | Owner / entrypoint | Host port | State and role |
|---|---|---:|---|
| API | `fastapi/app/main.py`, `uvicorn app.main:app` | `18008` → `8000` | Public API; Compose enables TLS with `/app/certs/key.pem` and `cert.pem`. Owns API tables in the MLflow database named by `DATABASE_URL`. |
| MLflow tracking | `start-mlflow.sh` | `15000` → `5000` | Tracking and model registry backend in PostgreSQL; artifact root `s3://mlflow/`. Startup requires PostgreSQL health and successful bucket initialization. |
| Credit model servers | `mlflow models serve -m models:/…/Production` | `18001`–`18003` | Production-stage serving for `CreditScorePredictor`, `RiskCategoryPredictor`, and `CreditLimitPredictor`. |
| Fraud model server | same mechanism | `18004` → `8004` | Serves `FraudDetector/Production`; API may instead load a configured registry version or a local pickle. |
| PostgreSQL | `postgres:14.0`, `scripts/init-dbs.sh` | `15432` → `5432` | Persistent `pgdata`; initialization creates distinct Airflow and MLflow databases/users. The API's `DATABASE_URL` targets the MLflow DB in Compose. |
| MinIO + initializer | `minio`, `mc-create-bucket` | `19000`, `19001` | Persistent `minio_data`; initializer creates bucket `mlflow` before tracking starts. |
| Airflow | `airflow/Dockerfile`, DAGs in `airflow/dags` | webserver `18087` → `8080` | Scheduler migrates Airflow DB, creates configured admin user, then schedules. DAG/jobs/data/log volumes are mounted. |
| Spark | `spark/Dockerfile` | master UI `19080`, master `17077` | One master and two workers. Training/jobs and data are mounted into Spark paths. |

## Startup, mounts, and configuration

The API is gated on PostgreSQL health and MLflow/model-server service startup, but `depends_on: service_started` is not a readiness probe for the model endpoints. The application additionally has timeouts and fallbacks; see [credit and loan API](../api/credit-and-loans.md) and [fraud API](../api/fraud.md).

`x-airflow-common` mounts `airflow/dags`, `airflow/jobs`, `data`, and `logs`. `x-spark-common` mounts `spark/training` as Spark jobs and `data` as input data. The API mounts `fastapi/app`, `data`, and `airflow/jobs` so `/admin/retrain` can execute `train_models_pandas.py` in-process. That last mount is a deliberate cross-system coupling documented in [training and registry](../ml/training-and-registry.md).

Use `.env.example` as the configuration inventory; it contains placeholders and groups internal PostgreSQL/Airflow/MLflow/MinIO settings, external database locations, API hardening flags, and MLflow URIs. Do not put actual values in source or documentation. `DATABASE_URL` is mandatory: `app.database` fails during import if it is absent. The external UAA and origination sessions are attempted at module import; failed construction leaves their dependencies yielding `None`, which alters strict/demo scoring behavior.

## Exact Compose wiring

All declared long-running services join `credit-net`: `fastapi-app`, `postgres`, `minio`, `mlflow-server`, the four model servers, `spark-master`, `spark-worker`, `spark-worker2`, `webserver`, and `scheduler`; `mc-create-bucket` also joins it as a one-shot initializer. Host/container mappings are: API `18008:8000`; tracking `15000:5000` and `16060:6060`; score `18001:8001`; risk `18002:8002`; limit `18003:8003`; fraud `18004:8004`; PostgreSQL `15432:5432`; MinIO API `19000:9000` and console `19001:9001`; Airflow webserver `18087:8080`; Spark UI `19080:8080` and master `17077:7077`. Spark workers have no host ports.

The primary ordered chain is PostgreSQL health → MinIO health → `mc-create-bucket` successful completion → `mlflow-server` started → the model-server containers started → `fastapi-app` started. `scheduler` and Airflow common services wait for PostgreSQL health; webserver waits for scheduler. Spark workers wait for master. The API additionally waits for PostgreSQL health plus tracking and each model service *started*, not healthy. `mc-create-bucket` first runs `mc alias set minio http://minio:9000 …`, then `mc mb minio/mlflow --ignore-existing`, which establishes the tracking artifact bucket.

Container mount destinations are as follows: Spark has `./spark/training:/opt/bitnami/spark/jobs` and `./data:/opt/bitnami/spark/data`. Airflow has `./data:/opt/bitnami/spark/data`, `./airflow/dags:/opt/airflow/dags`, `./airflow/jobs:/opt/bitnami/spark/jobs`, `./logs:/opt/airflow/logs`, `./airflow/jobs:/opt/airflow/jobs`, and `./data:/opt/airflow/data`. FastAPI mounts application code at `/app/app`, data at both `/app/data` and `/opt/airflow/data`, training jobs at `/app/airflow_jobs`, and read-only certificates at `/app/certs`.

Compose derives `DATABASE_URL` for FastAPI as `postgresql+psycopg2://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@postgres:5432/${MLFLOW_DB_NAME}`; the external UAA and origination host, port, database, user, and password variables are passed through separately. It supplies `MLFLOW_TRACKING_URI=http://mlflow-server:5000` plus `MLFLOW_SCORE_MODEL_URI=http://mlflow-server-score:8001`, `MLFLOW_RISK_MODEL_URI=http://mlflow-server-risk:8002`, `MLFLOW_LIMIT_MODEL_URI=http://mlflow-server-limit:8003`, and `MLFLOW_FRAUD_MODEL_URI=http://mlflow-server-fraud:8004`. Tracking itself uses a PostgreSQL backend URI built from `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`, with `s3://mlflow/` as artifact root and MinIO endpoint credentials. In contrast, serving processes resolve `models:/CreditScorePredictor/Production`, `models:/RiskCategoryPredictor/Production`, `models:/CreditLimitPredictor/Production`, and `models:/FraudDetector/Production`; those are model-registry stage URIs, not tracking or artifact addresses.

## Operations sequence

1. Copy `.env.example` to a private `.env` and supply internal plus external settings.
2. Start the Compose stack. PostgreSQL initialization runs only when its persistent volume is first initialized; changing database-user variables later does not rerun it automatically.
3. Run an Airflow credit or fraud training DAG, or invoke the corresponding job, to create registry versions.
4. Promote the required versions to MLflow's `Production` stage before expecting the three credit model servers to serve them. The code registers versions but does not promote them.
5. Probe `GET /health`, then create an API client before calling protected endpoints; see [authentication and contracts](../api/auth-and-contracts.md).

## Focused validation

- Configuration/rendering: `docker compose config` (requires the private environment file).
- Runtime dependency view: `curl -k https://localhost:18008/health`; the response is a normal API envelope and reports local PostgreSQL, both external databases, MLflow tracking, and strict-mode status.
- ML artifact path: manually trigger Airflow DAG `mlflow_infrastructure_smoke_test`; it creates, registers, downloads, and content-checks a dummy artifact.

The repository contains no tracked automated test suites. Runtime/DAG checks are the narrowest available validation evidence.
