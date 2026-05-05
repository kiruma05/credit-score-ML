import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_model_for_shap(model_name: str, tracking_uri: str):
    """Load a sklearn model from MLflow registry for SHAP explanation."""
    try:
        import mlflow.sklearn
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded %s from MLflow registry for SHAP.", model_name)
        return model
    except Exception as e:
        logger.warning("Could not load %s from MLflow registry: %s", model_name, e)
        return None


def explain_prediction(model, features: dict, top_n: int = 5) -> dict:
    """
    Compute SHAP values for a single prediction.
    Falls back to feature_importances_ if SHAP fails.
    Returns top_n features sorted by absolute impact.
    """
    feature_names = list(features.keys())
    df = pd.DataFrame([features])

    # Convert all values to float where possible
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except (ValueError, TypeError):
            df[col] = 0.0

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        # For regressors shap_values is a 2D array; for classifiers it may be a list
        if isinstance(shap_values, list):
            values = shap_values[1][0]  # class 1 for classifiers
        else:
            values = shap_values[0]

        base_score = float(explainer.expected_value) if not isinstance(
            explainer.expected_value, (list, np.ndarray)
        ) else float(explainer.expected_value[1])

        drivers = [
            {
                "feature": name,
                "impact": round(float(val), 4),
                "value": features.get(name),
                "direction": "helps_score" if val > 0 else "hurts_score",
            }
            for name, val in zip(feature_names, values)
        ]
        drivers.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return {"top_drivers": drivers[:top_n], "base_score": round(base_score, 2), "method": "shap"}

    except Exception as e:
        logger.warning("SHAP failed, falling back to feature_importances_: %s", e)
        return _fallback_importance(model, features, top_n)


def _fallback_importance(model, features: dict, top_n: int) -> dict:
    """Use model.feature_importances_ when SHAP is unavailable."""
    try:
        importances = model.feature_importances_
        feature_names = list(features.keys())[:len(importances)]
        drivers = [
            {
                "feature": name,
                "impact": round(float(imp), 4),
                "value": features.get(name),
                "direction": "helps_score",
            }
            for name, imp in zip(feature_names, importances)
        ]
        drivers.sort(key=lambda x: x["impact"], reverse=True)
        return {"top_drivers": drivers[:top_n], "base_score": None, "method": "feature_importance"}
    except Exception as e:
        logger.error("Fallback importance also failed: %s", e)
        return {"top_drivers": [], "base_score": None, "method": "unavailable"}


def summarize_drivers(top_drivers: list) -> str:
    """Generate a one-line human-readable summary of the top drivers."""
    if not top_drivers:
        return "Explanation unavailable."
    hurts = [d["feature"] for d in top_drivers if d["direction"] == "hurts_score"]
    helps = [d["feature"] for d in top_drivers if d["direction"] == "helps_score"]
    parts = []
    if hurts:
        parts.append(f"Score reduced by: {', '.join(hurts[:2])}")
    if helps:
        parts.append(f"Score supported by: {', '.join(helps[:2])}")
    return ". ".join(parts) + "." if parts else "No dominant drivers found."
