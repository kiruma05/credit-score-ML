import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_model_for_shap(model_name: str, tracking_uri: str):
    try:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        logger.info("Loaded %s from MLflow registry for SHAP.", model_name)
        return model
    except Exception as e:
        logger.warning("Could not load %s from MLflow registry: %s", model_name, e)
        return None


def _extract_pipeline_parts(model):
    """Extract preprocessor and final estimator from a sklearn Pipeline."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        steps = model.named_steps
        step_names = list(steps.keys())
        final_estimator = steps[step_names[-1]]
        if len(step_names) > 1:
            preprocessor = model[:-1]  # all steps except the last
        else:
            preprocessor = None
        return preprocessor, final_estimator
    return None, model


def explain_prediction(model, features: dict, top_n: int = 5) -> dict:
    """
    Compute SHAP values for a single prediction.
    Returns at least `top_n` positive (helps) AND `top_n` negative (hurts) drivers.
    Handles both raw sklearn estimators and Pipeline objects.
    Falls back to feature_importances_ if SHAP fails.
    """
    feature_names = list(features.keys())
    df = pd.DataFrame([features])

    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except (ValueError, TypeError):
            df[col] = 0.0

    preprocessor, estimator = _extract_pipeline_parts(model)

    try:
        import shap

        if preprocessor is not None:
            X = preprocessor.transform(df)
            try:
                transformed_names = preprocessor.get_feature_names_out()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            X = df.values
            transformed_names = feature_names

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        base_score = float(explainer.expected_value) if not isinstance(
            explainer.expected_value, (list, np.ndarray)
        ) else float(explainer.expected_value[1])

        # Build all drivers (skip near-zero noise)
        all_drivers = [
            {
                "feature": name.split("__")[-1],  # strip pipeline prefix
                "impact": round(float(val), 4),
                "value": features.get(name.split("__")[-1]),
                "direction": "helps_score" if val > 0 else "hurts_score",
            }
            for name, val in zip(transformed_names, values)
            if abs(val) > 0.001
        ]

        # helps: highest positive impact first (descending)
        helps = sorted(
            [d for d in all_drivers if d["direction"] == "helps_score"],
            key=lambda x: x["impact"],
            reverse=True,
        )[:top_n]

        # hurts: highest magnitude negative impact first
        # (most damaging at top → least damaging at bottom)
        hurts = sorted(
            [d for d in all_drivers if d["direction"] == "hurts_score"],
            key=lambda x: abs(x["impact"]),
            reverse=True,
        )[:top_n]

        # Combined top drivers (sorted by absolute impact)
        combined = sorted(
            helps + hurts, key=lambda x: abs(x["impact"]), reverse=True
        )

        return {
            "top_drivers": combined,
            "top_helps": helps,
            "top_hurts": hurts,
            "base_score": round(base_score, 2),
            "method": "shap",
        }

    except Exception as e:
        logger.warning("SHAP failed, falling back to feature_importances_: %s", e)
        return _fallback_importance(estimator, feature_names, top_n)


def _fallback_importance(estimator, feature_names: list, top_n: int) -> dict:
    """
    Fallback when SHAP fails — uses sklearn's global feature_importances_.
    Note: feature_importances_ are always positive (no direction).
    """
    try:
        importances = estimator.feature_importances_
        drivers = [
            {
                "feature": name,
                "impact": round(float(imp), 4),
                "value": None,
                "direction": "helps_score",  # global importance has no direction
            }
            for name, imp in zip(feature_names, importances)
            if imp > 0
        ]
        drivers.sort(key=lambda x: x["impact"], reverse=True)
        return {
            "top_drivers": drivers[: top_n * 2],
            "top_helps": drivers[:top_n],
            "top_hurts": [],
            "base_score": None,
            "method": "feature_importance",
        }
    except Exception as e:
        logger.error("Fallback importance also failed: %s", e)
        return {
            "top_drivers": [],
            "top_helps": [],
            "top_hurts": [],
            "base_score": None,
            "method": "unavailable",
        }


def summarize_drivers(top_drivers: list) -> str:
    if not top_drivers:
        return "Explanation unavailable."
    hurts = [d["feature"] for d in top_drivers if d["direction"] == "hurts_score"]
    helps = [d["feature"] for d in top_drivers if d["direction"] == "helps_score"]
    parts = []
    if hurts:
        parts.append(f"Score reduced by: {', '.join(hurts[:7])}")
    if helps:
        parts.append(f"Score supported by: {', '.join(helps[:7])}")
    return ". ".join(parts) + "." if parts else "No dominant drivers found."
