import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from urllib import request, error

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score


def check_mlflow_connection(uri, timeout=20):
    """Checks if the MLflow tracking server is reachable before proceeding."""
    print(f"\n🔍 Checking MLflow connection to {uri} (timeout: {timeout}s)")
    try:
        health_uri = f"{uri}/health"
        response = request.urlopen(health_uri, timeout=timeout)
        if response.status == 200:
            print("✅ MLflow connection successful!")
            return
    except Exception:
        # Fallback: try listing experiments
        try:
            mlflow.set_tracking_uri(uri)
            client = MlflowClient()
            client.list_experiments()
            print("✅ MLflow connection verified via fallback.")
            return
        except Exception as e:
            print(f"❌ CRITICAL: Failed to connect to MLflow server at {uri}. Error: {e}", file=sys.stderr)
            sys.exit(1)


def log_and_register_model(client, model, model_name, artifact_path, run_id):
    """Logs and registers a model with MLflow."""
    print(f"\n📦 Logging & Registering Model: {model_name}")
    mlflow.sklearn.log_model(model, artifact_path)
    print(f"📁 Artifact logged at: {artifact_path}")

    artifact_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        client.get_registered_model(name=model_name)
        print(f"ℹ️ Model '{model_name}' already registered.")
    except mlflow.exceptions.MlflowException:
        print(f"🆕 Registering new model: {model_name}")
        client.create_registered_model(name=model_name)

    new_version = client.create_model_version(
        name=model_name,
        source=artifact_uri,
        run_id=run_id
    )
    print(f"✅ Model version {new_version.version} created for '{model_name}'")


def train():
    """Main training pipeline."""
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        print("❌ CRITICAL: MLFLOW_TRACKING_URI environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    check_mlflow_connection(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("\n📊 Loading and preparing data...")
    data_path = "/opt/airflow/data/customer_data.csv"
    df = pd.read_csv(data_path).fillna(0)

    # Drop identifiers + targets + leakage helpers. payment_history_score is
    # the target of CreditScorePredictor (so it must NEVER be an input feature
    # — was causing the model to just regurgitate the default value of 500).
    # is_approved / is_fraud / is_high_risk are derived from the target too.
    leakage_cols = [c for c in (
        'customer_id', 'nida',
        'risk_category_target',
        'payment_history_score',
        'is_approved', 'is_fraud', 'is_high_risk',
    ) if c in df.columns]
    X = df.drop(columns=leakage_cols)
    y_score = df['payment_history_score']
    y_risk = df['risk_category_target']
    y_limit = df['monthly_income'] * 3

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    X[categorical_features] = X[categorical_features].astype(str)

    X_train, X_test, y_score_train, y_score_test, y_risk_train, y_risk_test, y_limit_train, y_limit_test = train_test_split(
        X, y_score, y_risk, y_limit, test_size=0.2, random_state=42, stratify=y_risk
    )

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Train Credit Score Model
    with mlflow.start_run(run_name="CreditScore_GBT_Regressor_Sklearn") as run:
        score_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        score_pipeline.fit(X_train, y_score_train)
        preds = score_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_score_test, preds))
        mlflow.log_metric("rmse", rmse)
        log_and_register_model(client, score_pipeline, "CreditScorePredictor", "credit_score_model", run.info.run_id)

    # Train Risk Category Model
    with mlflow.start_run(run_name="RiskCategory_RandomForest_Sklearn") as run:
        risk_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        risk_pipeline.fit(X_train, y_risk_train)
        preds = risk_pipeline.predict(X_test)
        f1 = f1_score(y_risk_test, preds, average='weighted')
        mlflow.log_metric("f1_score", f1)
        log_and_register_model(client, risk_pipeline, "RiskCategoryPredictor", "risk_category_model", run.info.run_id)

    # Train Credit Limit Model
    with mlflow.start_run(run_name="CreditLimit_GBT_Regressor_Sklearn") as run:
        limit_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        limit_pipeline.fit(X_train, y_limit_train)
        preds = limit_pipeline.predict(X_test)
        rmse_limit = np.sqrt(mean_squared_error(y_limit_test, preds))
        mlflow.log_metric("rmse", rmse_limit)
        log_and_register_model(client, limit_pipeline, "CreditLimitPredictor", "credit_limit_model", run.info.run_id)

    print("\n🎉 All models trained, evaluated, and registered successfully!")


if __name__ == '__main__':
    train()
