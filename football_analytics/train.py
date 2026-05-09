import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_analytics.config import MODELS_DIR
from football_analytics.db import create_tables
from football_analytics.features import MODEL_FEATURES, available_model_features, build_training_dataframe

MODEL_PATH = MODELS_DIR / "football_1x2_model.joblib"
REPORT_PATH = MODELS_DIR / "football_1x2_report.json"


def multiclass_brier_score(y_true, probas, num_classes: int = 3) -> float:
    y_true = np.asarray(y_true)
    y_onehot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((y_onehot - probas) ** 2, axis=1)))


def build_model_pipeline(model_type: str = "hist_gradient_boosting") -> Pipeline:
    if model_type == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_depth=6,
                        max_iter=350,
                        min_samples_leaf=8,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_type == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save_model(model_type: str = "hist_gradient_boosting") -> Dict[str, Any]:
    create_tables()
    df = build_training_dataframe()

    if df.empty or len(df) < 40:
        raise ValueError("Model egitimi icin yeterli tamamlanmis mac verisi yok.")

    df = df.sort_values("kickoff_time").reset_index(drop=True)

    feature_columns = available_model_features(df)
    if not feature_columns:
        raise ValueError("Model egitimi icin kullanilabilir feature kolonu bulunamadi.")

    X = df[feature_columns].copy()
    y = df["target"].astype(int).copy()

    split_idx = max(30, int(len(df) * 0.8))
    if split_idx >= len(df):
        split_idx = len(df) - 1

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipeline = build_model_pipeline(model_type=model_type)

    cv_losses = []
    n_splits = min(4, max(2, len(X_train) // 25))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_model = build_model_pipeline(model_type=model_type)
        fold_model.fit(X_tr, y_tr)
        fold_proba = fold_model.predict_proba(X_val)
        cv_losses.append(log_loss(y_val, fold_proba, labels=[0, 1, 2]))

    pipeline.fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)
    test_pred = np.argmax(test_proba, axis=1)

    metrics = {
        "model_type": model_type,
        "training_scope": "trusted_iddaa_leagues",
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "league_count": int(df["league_name"].nunique()) if "league_name" in df.columns else None,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "log_loss": float(log_loss(y_test, test_proba, labels=[0, 1, 2])),
        "brier_score": float(multiclass_brier_score(y_test.values, test_proba, num_classes=3)),
        "cv_log_loss_mean": float(np.mean(cv_losses)) if cv_losses else None,
        "cv_log_loss_std": float(np.std(cv_losses)) if cv_losses else None,
        "feature_columns": feature_columns,
    }

    bundle = {
        "model_name": model_type,
        "model_type": model_type,
        "feature_columns": feature_columns,
        "model": pipeline,
        "metrics": metrics,
    }

    joblib.dump(bundle, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save_model(), indent=2, ensure_ascii=False))
