from typing import Any, Dict, List

import pandas as pd

from football_analytics.train import train_and_save_model


def train_and_compare_models(model_types: List[str] | None = None) -> Dict[str, Any]:
    model_types = model_types or ["hist_gradient_boosting", "random_forest", "logreg"]
    results = []

    for model_type in model_types:
        metrics = train_and_save_model(model_type=model_type)
        row = {
            "model_type": model_type,
            "rows_total": metrics.get("rows_total"),
            "rows_train": metrics.get("rows_train"),
            "rows_test": metrics.get("rows_test"),
            "accuracy": metrics.get("accuracy"),
            "log_loss": metrics.get("log_loss"),
            "brier_score": metrics.get("brier_score"),
            "cv_log_loss_mean": metrics.get("cv_log_loss_mean"),
            "cv_log_loss_std": metrics.get("cv_log_loss_std"),
        }
        results.append(row)

    comparison_df = pd.DataFrame(results).sort_values("log_loss", ascending=True).reset_index(drop=True)
    return {"comparison": comparison_df}
