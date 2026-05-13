"""SHAP-based explainability for the credit scoring model.

Uses the model-agnostic ``shap.Explainer(model.predict, background)`` API so
it works on any sklearn pipeline regardless of preprocessor / estimator type
— no more dtype, sparse-matrix or ColumnTransformer pitfalls. Background
samples are drawn once from the training CSV and cached on the module.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BACKGROUND: Optional[pd.DataFrame] = None
_BACKGROUND_PATHS = (
    "/app/data/customer_data.csv",          # mounted via docker-compose
    "/opt/airflow/data/customer_data.csv",  # alternate mount
)


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
    """Return ``(preprocessor, estimator)`` for a sklearn Pipeline, else ``(None, model)``."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        steps = model.named_steps
        step_names = list(steps.keys())
        final_estimator = steps[step_names[-1]]
        preprocessor = model[:-1] if len(step_names) > 1 else None
        return preprocessor, final_estimator
    return None, model


def _load_background(n: int = 50, columns: Optional[list] = None) -> Optional[pd.DataFrame]:
    """Load (and cache) a small background sample for SHAP expected-value baseline."""
    global _BACKGROUND
    if _BACKGROUND is not None:
        if columns is not None:
            # Reindex to match the requested column order on every call
            return _BACKGROUND.reindex(columns=columns, fill_value=0)
        return _BACKGROUND

    for path in _BACKGROUND_PATHS:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            drop_cols = [c for c in (
                "customer_id", "nida",
                "risk_category_target", "is_approved", "is_fraud", "is_high_risk",
                "payment_history_score",  # this is the model target, not an input
            ) if c in df.columns]
            df = df.drop(columns=drop_cols)
            _BACKGROUND = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
            logger.info("SHAP background loaded from %s (%d rows)", path, len(_BACKGROUND))
            if columns is not None:
                return _BACKGROUND.reindex(columns=columns, fill_value=0)
            return _BACKGROUND
        except Exception as e:
            logger.warning("Failed to load SHAP background from %s: %s", path, e)

    logger.warning("SHAP background CSV not found in any expected location.")
    return None


def explain_prediction(model, features: dict, top_n: int = 5) -> dict:
    """Compute SHAP values for a single prediction.

    Uses the model-agnostic ``shap.Explainer(model.predict, background)`` so
    we can hand the whole Pipeline to SHAP and avoid pre/post-processing
    nightmares. Falls back to ``feature_importances_`` if SHAP fails or the
    background CSV is unavailable.
    """
    feature_names = list(features.keys())
    df = pd.DataFrame([features])

    # Match the dtype expectations of the trained pipeline.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("None").astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    preprocessor, estimator = _extract_pipeline_parts(model)

    try:
        import shap

        bg = _load_background(columns=df.columns.tolist())
        if bg is None or len(bg) == 0:
            raise RuntimeError("SHAP background dataset unavailable")

        # Align dtypes between request row and background.
        for col in df.columns:
            if col in bg.columns and df[col].dtype != bg[col].dtype:
                try:
                    df[col] = df[col].astype(bg[col].dtype)
                except (ValueError, TypeError):
                    pass

        explainer = shap.Explainer(model.predict, bg)
        sv = explainer(df)
        # sv.values shape: (1, n_features); sv.base_values shape: (1,)
        values = np.asarray(sv.values[0], dtype=np.float64)
        base = float(np.asarray(sv.base_values).flatten()[0])

        contributions = [
            (name, float(val))
            for name, val in zip(df.columns, values)
            if abs(val) > 1e-6
        ]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        drivers = [
            {
                "feature": name,
                "impact": round(val, 4),
                "direction": "helps_score" if val > 0 else "hurts_score",
                "value": features.get(name),
            }
            for name, val in contributions[:top_n]
        ]
        return {"top_drivers": drivers, "base_score": round(base, 2), "method": "shap"}

    except Exception as e:
        logger.warning("SHAP failed, falling back to feature_importances_: %s", e)
        return _fallback_importance(estimator, feature_names, features, top_n)


def _fallback_importance(estimator, feature_names: list, features: dict, top_n: int) -> dict:
    try:
        importances = estimator.feature_importances_
        # If the importance vector is longer than the raw feature list (due to
        # one-hot encoding inside the pipeline), align by truncation.
        n = min(len(feature_names), len(importances))
        drivers = [
            {
                "feature": feature_names[i],
                "impact": round(float(importances[i]), 4),
                "value": features.get(feature_names[i]),
                "direction": "helps_score",
            }
            for i in range(n)
        ]
        drivers.sort(key=lambda x: x["impact"], reverse=True)
        return {"top_drivers": drivers[:top_n], "base_score": None, "method": "feature_importance"}
    except Exception as e:
        logger.error("Fallback importance also failed: %s", e)
        return {"top_drivers": [], "base_score": None, "method": "unavailable"}


def summarize_drivers(top_drivers: list) -> str:
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
