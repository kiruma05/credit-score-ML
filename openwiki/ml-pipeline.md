---
okf_version: "0.1"
type: concept
title: ML Pipeline (MLflow · Airflow · Spark)
---

# ML Pipeline (MLflow · Airflow · Spark)

Models are trained by Airflow-orchestrated jobs (some on Spark, some in pandas),
tracked/registered in MLflow, stored in MinIO, and served by dedicated MLflow
model containers that the API calls at inference time.

## Serving layer

MLflow runs as a tracking/registry server (`mlflow-server`, `:15000`) plus four
per-model serving containers (all built from `./mlflow`):

| Model | Registry name | Serving container | Port |
|---|---|---|---|
| Credit score | `CreditScorePredictor` | `mlflow-server-score` | `:18001` |
| Risk | (risk model) | `mlflow-server-risk` | `:18002` |
| Credit limit | (limit model) | `mlflow-server-limit` | `:18003` |
| Fraud | `FraudDetector` | `mlflow-server-fraud` | `:18004` |

The API reaches these via `MLFLOW_SCORE_MODEL_URI`, `MLFLOW_RISK_MODEL_URI`,
`MLFLOW_LIMIT_MODEL_URI`, and `MLFLOW_FRAUD_MODEL_URI`. Artifacts are stored in
MinIO through `MLFLOW_S3_ENDPOINT_URL=http://minio:9000`.

## Training jobs (`airflow/jobs/`)

| Job | Role |
|---|---|
| `train_models.py` | Train the credit models on Spark. |
| `train_models_pandas.py` | Train the credit models with pandas/scikit-learn. |
| `train_fraud_model.py` | Train the fraud model. |
| `smoke_test.py` | MLflow connectivity/infrastructure smoke test. |
| `simple_spark_job.py` | Minimal Spark job (pipeline sanity check). |

## DAGs (`airflow/dags/`)

| DAG id | File | Schedule | What it runs |
|---|---|---|---|
| `fraud_detection_dag` | `fraud_detection_dag.py` | manual (`None`) | start → `train_fraud_model` (Bash) → end |
| `pandas_model_training` | `pandas_training_dag.py` | manual (`None`) | start → `train_pandas_sklearn_models` (Bash) → end |
| `model_retraining_with_spark_operator` | `retraining_dag.py` | manual (`None`) | `SparkSubmitOperator` → `spark_train_submit_task` |
| `mlflow_infrastructure_smoke_test` | `smoke_test_dag.py` | manual (`None`) | `SparkSubmitOperator` → MLflow connectivity test |
| `sparking_flow` | `spark_submit_dag.py` | `@daily` | `SparkSubmitOperator` → `python_job` |

Most training DAGs are **manual-trigger** (`schedule_interval=None`); only
`sparking_flow` is scheduled daily. Spark DAGs submit to `spark-master`
(`:17077`) with two workers.

## Triggering retraining from the API

`POST /admin/retrain` (guarded by `ADMIN_TOKEN`) asynchronously retrains all
three credit models — see [API Reference](api-reference.md). Fraud retraining is
handled by `fastapi/app/fraud/retrain_fraud_model.py` and the
`fraud_detection_dag`.

## Cross-references

- How served models are consumed at request time: [Credit Scoring Engine](credit-scoring.md), [Fraud Detection](fraud-detection.md)
- Container topology and ports: [Architecture & Services](architecture.md)
