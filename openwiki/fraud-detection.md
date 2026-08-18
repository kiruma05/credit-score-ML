---
okf_version: "0.1"
type: concept
title: Fraud Detection
---

# Fraud Detection

Fraud detection is a self-contained module under `fastapi/app/fraud/` with its
own router, model loader, and retraining helper. It is independent of the credit
scoring models.

## Endpoint

- `POST /fraud/predict` (`fastapi/app/fraud/fraud_router.py`) — scores a
  transaction and returns a `FraudPrediction`. Requires API auth.

## Model loading strategy (`fastapi/app/fraud/fraud_model.py`)

`load_model()` tries three sources in order, so the service degrades gracefully:

1. **Served endpoint** — if `MLFLOW_FRAUD_MODEL_URI` is set, it health-checks
   `{uri}/ping` and, if live, returns `{"type": "served", "uri": ...}`
   (served by container `mlflow-server-fraud` on `:18004`).
2. **MLflow registry** — otherwise loads `models:/{MODEL_NAME}/{MODEL_VERSION}`
   from `MLFLOW_TRACKING_URI` (defaults: `FraudDetector` / version `1`).
3. **Local file** — finally falls back to `/opt/airflow/data/model.pkl`.

If all three fail it raises `RuntimeError`. Relevant env vars: `MODEL_NAME`,
`MODEL_VERSION`, `MLFLOW_TRACKING_URI`, `MLFLOW_FRAUD_MODEL_URI`, `ENCODER_PATH`.

`load_label_encoders()` loads the categorical encoders saved at training time
from `ENCODER_PATH` (default `/opt/airflow/data/label_encoders.pkl`), returning
`{}` if the file is missing rather than failing.

## Retraining

- `fastapi/app/fraud/retrain_fraud_model.py` — retrain helper callable from the API.
- `airflow/jobs/train_fraud_model.py` — the training job.
- `airflow/dags/fraud_detection_dag.py` — the `fraud_detection_dag` (manual
  trigger, `schedule_interval=None`) that runs the fraud training job.

## Supporting data & scripts

- `data/label_encoders.pkl`, `data/model.pkl` — persisted encoders/model used by
  the local fallback path.
- `generate_dummy_fraud_data.py` — generates synthetic fraud training data.

## Cross-references

- Model serving infra: [ML Pipeline](ml-pipeline.md)
- Endpoint list: [API Reference](api-reference.md)
