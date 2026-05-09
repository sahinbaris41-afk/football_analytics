from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from football_analytics.features import build_training_dataframe
from football_analytics.train import MODEL_PATH

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    shap = None


def _load_bundle(model_path=MODEL_PATH) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError("Model bulunamadi. Once modeli egit.")
    return joblib.load(model_path)


def compute_model_explanations(
    model_path=MODEL_PATH,
    sample_size: int = 200,
) -> Dict[str, Any]:
    bundle = _load_bundle(model_path=model_path)
    feature_columns = bundle["feature_columns"]
    pipeline = bundle["model"]
    model_name = bundle.get("model_name", "unknown_model")

    df = build_training_dataframe()
    if df.empty:
        raise ValueError("Aciklama icin tarihsel egitim verisi bulunamadi.")

    df = df.sort_values("kickoff_time").reset_index(drop=True)
    sample_df = df.tail(min(sample_size, len(df))).copy()
    X = sample_df[feature_columns].copy()

    preprocessor = pipeline[:-1]
    estimator = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X)

    method = "coefficients"
    global_importance = None
    sample_explanation = None

    if shap is not None:
        try:
            if model_name == "random_forest":
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_transformed)
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
                if values.ndim == 3:
                    values = values.mean(axis=2)
                mean_abs = np.abs(values).mean(axis=0)
                method = "shap_tree"
                sample_matrix = values
            else:
                explainer = shap.LinearExplainer(estimator, X_transformed)
                shap_values = explainer.shap_values(X_transformed)
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
                if values.ndim == 3:
                    values = values.mean(axis=2)
                mean_abs = np.abs(values).mean(axis=0)
                method = "shap_linear"
                sample_matrix = values

            global_importance = (
                pd.DataFrame(
                    {
                        "feature": feature_columns,
                        "importance": mean_abs,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

            top_features = global_importance.head(10)["feature"].tolist()
            sample_explanation = sample_df[["kickoff_time", "league_name", "home_team", "away_team"]].copy()
            for feature in top_features:
                col_idx = feature_columns.index(feature)
                sample_explanation[f"shap_{feature}"] = sample_matrix[:, col_idx]
        except Exception:
            method = "fallback"

    if global_importance is None:
        if hasattr(estimator, "feature_importances_"):
            scores = estimator.feature_importances_
            method = "feature_importances"
        elif hasattr(estimator, "coef_"):
            scores = np.abs(estimator.coef_).mean(axis=0)
            method = "coefficients"
        else:
            scores = np.zeros(len(feature_columns))
            method = "unavailable"

        global_importance = (
            pd.DataFrame({"feature": feature_columns, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return {
        "model_name": model_name,
        "method": method,
        "global_importance": global_importance,
        "sample_explanation": sample_explanation,
    }
