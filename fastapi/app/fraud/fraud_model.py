import os
import pickle
import mlflow.sklearn
import requests
import sys

# Environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "FraudDetector")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
ENCODER_PATH = os.getenv("ENCODER_PATH", "/opt/airflow/data/label_encoders.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
FRAUD_MODEL_URI = os.getenv("MLFLOW_FRAUD_MODEL_URI", "")

def load_model():
    """
    Load the fraud model.
    Prefer served model endpoint (if configured), fall back to MLflow registry, then local file.
    """
    print(f"[INFO] Attempting to load fraud model. MODEL_NAME={MODEL_NAME}, MODEL_VERSION={MODEL_VERSION}")
    print(f"[INFO] MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}, FRAUD_MODEL_URI={FRAUD_MODEL_URI}")

    # Try connecting to served model endpoint (if FRAUD_MODEL_URI is set)
    if FRAUD_MODEL_URI:
        try:
            print(f"[INFO] Checking served fraud model at {FRAUD_MODEL_URI}/ping")
            health = requests.get(f"{FRAUD_MODEL_URI}/ping", timeout=5)
            print(f"[INFO] Served model endpoint response: status={health.status_code}, text={health.text}")
            if health.status_code in (200, 405):
                print(f"[INFO] Fraud model served endpoint is live at {FRAUD_MODEL_URI}")
                return {"type": "served", "uri": FRAUD_MODEL_URI}
            else:
                print(f"[WARNING] Served model endpoint returned status {health.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] Failed to reach served model endpoint: {str(e)}", file=sys.stderr)
    else:
        print("[INFO] Skipping served model endpoint (FRAUD_MODEL_URI not set)")

    # Fall back to MLflow registry
    try:
        print(f"[INFO] Falling back to MLflow registry at {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        print(f"[INFO] Loading fraud model from MLflow registry: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("[INFO] Fraud model loaded successfully from registry")
        return {"type": "registry", "model": model}
    except Exception as e:
        print(f"[ERROR] Failed to load model from MLflow registry: {str(e)}", file=sys.stderr)

    # Fall back to local model file
    try:
        local_model_path = "/opt/airflow/data/model.pkl"
        print(f"[INFO] Falling back to local model file at {local_model_path}")
        if os.path.exists(local_model_path):
            with open(local_model_path, "rb") as f:
                model = pickle.load(f)
            print(f"[INFO] Local model loaded successfully from {local_model_path}")
            return {"type": "local", "model": model}
        else:
            print(f"[ERROR] Local model file not found at {local_model_path}", file=sys.stderr)
            raise RuntimeError(f"No model available: served, registry, and local failed")
    except Exception as e:
        print(f"[ERROR] Failed to load local model: {str(e)}", file=sys.stderr)
        raise RuntimeError(f"Could not load fraud model: {str(e)}")

def load_label_encoders():
    """Load label encoders used during model training."""
    print(f"[INFO] Checking for label encoders at {ENCODER_PATH}")
    try:
        if os.path.exists(ENCODER_PATH):
            print(f"[INFO] Found label encoders file at {ENCODER_PATH}")
            with open(ENCODER_PATH, "rb") as f:
                encoders = pickle.load(f)
            print(f"[INFO] Label encoders loaded successfully from {ENCODER_PATH}")
            return encoders
        else:
            print(f"[WARNING] Label encoder file not found at {ENCODER_PATH}", file=sys.stderr)
            return {}
    except Exception as e:
        print(f"[ERROR] Failed to load label encoders: {str(e)}", file=sys.stderr)
        return {}