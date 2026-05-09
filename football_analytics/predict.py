from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from football_analytics.db import get_session, save_prediction
from football_analytics.features import build_live_feature_dataframe, build_upcoming_feature_dataframe
from football_analytics.train import MODEL_PATH


def _label_from_index(idx: int) -> str:
    return {0: "H", 1: "D", 2: "A"}[idx]


def _normalize_prob_triplet(home: float, draw: float, away: float) -> tuple[float, float, float]:
    values = np.array([home, draw, away], dtype=float)
    values = np.clip(values, 1e-6, None)
    total = float(values.sum())
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    values = values / total
    return float(values[0]), float(values[1]), float(values[2])


def _calibrate_prediction_row(row: pd.Series) -> pd.Series:
    prob_home, prob_draw, prob_away = _normalize_prob_triplet(
        row.get("prob_home", 1 / 3),
        row.get("prob_draw", 1 / 3),
        row.get("prob_away", 1 / 3),
    )

    imp_home = row.get("imp_home")
    imp_draw = row.get("imp_draw")
    imp_away = row.get("imp_away")

    has_market = all(pd.notna(value) for value in [imp_home, imp_draw, imp_away])
    if has_market:
        imp_home, imp_draw, imp_away = _normalize_prob_triplet(imp_home, imp_draw, imp_away)
        # Bookmaker mantigina daha yakin olmak icin model olasiliklarini piyasa ile harmanla.
        prob_home = (0.65 * prob_home) + (0.35 * imp_home)
        prob_draw = (0.65 * prob_draw) + (0.35 * imp_draw)
        prob_away = (0.65 * prob_away) + (0.35 * imp_away)
    else:
        # Piyasa yoksa modeli tek basina fazla agresif kullanma; uniforma dogru hafif fren uygula.
        prob_home = (0.85 * prob_home) + (0.15 / 3)
        prob_draw = (0.85 * prob_draw) + (0.15 / 3)
        prob_away = (0.85 * prob_away) + (0.15 / 3)

    prob_home, prob_draw, prob_away = _normalize_prob_triplet(prob_home, prob_draw, prob_away)
    row["prob_home"] = prob_home
    row["prob_draw"] = prob_draw
    row["prob_away"] = prob_away
    row["predicted_label"] = _label_from_index(int(np.argmax([prob_home, prob_draw, prob_away])))
    row["confidence"] = max(prob_home, prob_draw, prob_away)

    if has_market:
        row["edge_home"] = np.clip(prob_home - imp_home, -0.20, 0.20)
        row["edge_draw"] = np.clip(prob_draw - imp_draw, -0.20, 0.20)
        row["edge_away"] = np.clip(prob_away - imp_away, -0.20, 0.20)
    else:
        row["edge_home"] = np.nan
        row["edge_draw"] = np.nan
        row["edge_away"] = np.nan

    return row


def _predict_matches(df: pd.DataFrame, model_path: Optional[Path] = None) -> pd.DataFrame:
    model_path = model_path or MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError("Model bulunamadi. Once modeli egit.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    model_name = bundle.get("model_name", "unknown_model")

    if df.empty:
        return df

    X = df[feature_columns].copy()
    probs = model.predict_proba(X)
    pred_idx = np.argmax(probs, axis=1)

    df["prob_home"] = probs[:, 0]
    df["prob_draw"] = probs[:, 1]
    df["prob_away"] = probs[:, 2]
    df["predicted_label"] = [_label_from_index(i) for i in pred_idx]
    df["confidence"] = probs.max(axis=1)
    df = df.apply(_calibrate_prediction_row, axis=1)

    session = get_session()
    try:
        for _, row in df.iterrows():
            save_prediction(
                session=session,
                match_id=int(row["match_id"]),
                model_name=model_name,
                prob_home=float(row["prob_home"]),
                prob_draw=float(row["prob_draw"]),
                prob_away=float(row["prob_away"]),
                predicted_label=str(row["predicted_label"]),
                confidence=float(row["confidence"]),
                edge_home=float(row["edge_home"]) if pd.notna(row["edge_home"]) else None,
                edge_draw=float(row["edge_draw"]) if pd.notna(row["edge_draw"]) else None,
                edge_away=float(row["edge_away"]) if pd.notna(row["edge_away"]) else None,
            )
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        df.attrs["persistence_warning"] = str(exc)
    finally:
        session.close()

    ordered_cols = [
        "kickoff_time",
        "league_name",
        "home_team",
        "away_team",
        "league_known",
        "has_valid_1x2_odds",
        "has_supported_market_odds",
        "betting_ready",
        "betting_ready_reason",
        "odds_consistency_ok",
        "odds_consistency_gap",
        "odds_consistency_market",
        "odds_consistency_reference_source",
        "league_quality_index",
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_points_last5",
        "away_points_last5",
        "form_diff",
        "home_goal_diff_last5",
        "away_goal_diff_last5",
        "goal_diff_form_diff",
        "home_goals_for_last5",
        "away_goals_for_last5",
        "home_goals_against_last5",
        "away_goals_against_last5",
        "home_rest_days",
        "away_rest_days",
        "rest_diff",
        "market_home_odds",
        "market_draw_odds",
        "market_away_odds",
        "market_over_25_odds",
        "market_under_25_odds",
        "market_btts_yes_odds",
        "market_btts_no_odds",
        "market_double_chance_1x_odds",
        "market_double_chance_12_odds",
        "market_double_chance_x2_odds",
        "market_handicap_home_10_odds",
        "market_handicap_draw_10_odds",
        "market_handicap_away_10_odds",
        "imp_home",
        "imp_draw",
        "imp_away",
        "market_overround",
        "prob_home",
        "prob_draw",
        "prob_away",
        "predicted_label",
        "confidence",
        "edge_home",
        "edge_draw",
        "edge_away",
        "source",
        "status",
        "match_id",
    ]
    ordered_cols = [column for column in ordered_cols if column in df.columns]
    return df[ordered_cols].sort_values("kickoff_time").reset_index(drop=True)


def predict_upcoming_matches(model_path: Optional[Path] = None) -> pd.DataFrame:
    return _predict_matches(build_upcoming_feature_dataframe(), model_path=model_path)


def predict_live_matches(model_path: Optional[Path] = None) -> pd.DataFrame:
    return _predict_matches(build_live_feature_dataframe(), model_path=model_path)


if __name__ == "__main__":
    print(predict_upcoming_matches().head(20).to_string(index=False))
