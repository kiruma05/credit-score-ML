from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import os

router = APIRouter()

MODEL_NAME = os.getenv("FRAUD_MODEL_NAME", "FraudDetector")
MODEL_VERSION = os.getenv("FRAUD_MODEL_VERSION", "1")

class Transaction(BaseModel):
    customer_age: str
    name_email_similarity: str
    bank_branch_count_8w: str
    bank_months_count: str
    credit_risk_score: str
    current_address_months_count: str
    customer_id: str
    nida_1000: str | None = None  # Added for potential categorical feature
    transaction_amount: str | None = None
    transaction_type: str | None = None
    merchant_category: str | None = None
    location: str | None = None
    time_of_transaction: str | None = None
    device_type: str | None = None
    is_new_account: str | None = None

def get_fraud_model():
    """Dependency to inject fraud model and encoders."""
    from ..main import app
    model_info = app.state.fraud_model
    encoders = app.state.label_encoders
    if model_info is None:
        raise HTTPException(status_code=503, detail="Fraud model not loaded.")
    if encoders is None:
        raise HTTPException(status_code=503, detail="Label encoders not loaded.")
    return model_info, encoders

@router.post("/predict")
def predict_fraud(transaction: Transaction, model_encoders: tuple = Depends(get_fraud_model)):
    """Predict fraud likelihood from transaction features."""
    model_info, encoders = model_encoders

    # Convert input to DataFrame
    data = pd.DataFrame([transaction.dict(exclude_unset=True)])

    # Apply label encoding for categorical features
    categorical_features = list(encoders.keys())  # Get from encoders
    for col in categorical_features:
        if col in data.columns and data[col].notna().all():
            try:
                data[col] = encoders[col].transform(data[col])
            except ValueError as e:
                print(f"[WARNING] Invalid value for {col}: {data[col].iloc[0]}. Using default: {encoders[col].classes_[0]}")
                data[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
        else:
            print(f"[WARNING] Missing categorical feature {col}. Using default: {encoders[col].classes_[0]}")
            data[col] = encoders[col].transform([encoders[col].classes_[0]])[0]

    # Convert numeric features to float
    numeric_features = [
        'customer_age', 'name_email_similarity', 'bank_branch_count_8w',
        'bank_months_count', 'credit_risk_score', 'current_address_months_count',
        'transaction_amount'
    ]
    for col in numeric_features:
        if col in data.columns and data[col].notna().all():
            try:
                data[col] = data[col].astype(float)
            except ValueError as e:
                print(f"[ERROR] Failed to convert {col} to float: {data[col].iloc[0]}")
                raise HTTPException(status_code=500, detail=f"Invalid numeric value for {col}: {str(e)}")
        elif col not in data.columns:
            print(f"[WARNING] Missing numeric feature {col}. Using default: 0.0")
            data[col] = 0.0

    # Ensure all model features are present
    model_features = getattr(model_info["model"], "feature_names_in_", [
        'customer_age', 'name_email_similarity', 'bank_branch_count_8w',
        'bank_months_count', 'credit_risk_score', 'current_address_months_count',
        'customer_id', 'nida_1000'  # Added nida_1000
    ])
    for feature in model_features:
        if feature not in data.columns:
            print(f"[WARNING] Missing feature {feature}. Using default: 0.0")
            data[feature] = 0.0

    try:
        if model_info["type"] == "served":
            response = requests.post(
                f"{model_info['uri']}/invocations",
                json={"inputs": data[model_features].to_dict(orient="records")},
                timeout=10,
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Served model request failed: {response.text}",
                )
            result = response.json()
            prediction = int(result.get("predictions", [0])[0])
            probability = float(result.get("probabilities", [0.0])[0]) if "probabilities" in result else None
        else:
            model = model_info["model"]
            prediction = int(model.predict(data[model_features])[0])
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(data[model_features])[0][1])
            else:
                probability = None

        return {
            "fraud_prediction": prediction,
            "fraud_probability": round(probability, 4) if probability is not None else None,
            "model_version": f"{MODEL_NAME}_v{MODEL_VERSION}",
            "served_from": model_info["type"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud prediction failed: {str(e)}")