from typing import Any, Dict, List, Tuple

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from football_analytics.db import create_tables
from football_analytics.features import available_model_features, build_training_dataframe, valid_simple_odds
from football_analytics.recommend import _double_chance_pick, _goal_market_pick, _handicap_10_pick
from football_analytics.train import build_model_pipeline, multiclass_brier_score

OUTCOME_SPECS: List[Tuple[str, str, str, str]] = [
    ("H", "prob_home", "imp_home", "market_home_odds"),
    ("D", "prob_draw", "imp_draw", "market_draw_odds"),
    ("A", "prob_away", "imp_away", "market_away_odds"),
]


def _kelly_fraction(probability: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - probability
    raw = ((b * probability) - q) / b
    return max(0.0, raw)


def _pick_bet(row: pd.Series, edge_threshold: float) -> Dict[str, Any]:
    best_bet: Dict[str, Any] = {"edge": None}

    for label, prob_col, imp_col, odds_col in OUTCOME_SPECS:
        probability = row.get(prob_col)
        implied_probability = row.get(imp_col)
        odds = row.get(odds_col)

        if pd.isna(probability) or pd.isna(implied_probability) or pd.isna(odds):
            continue

        edge = float(probability) - float(implied_probability)
        if edge < edge_threshold:
            continue

        if best_bet["edge"] is None or edge > best_bet["edge"]:
            best_bet = {
                "selected_label": label,
                "market_family": "1X2",
                "recommended_market": "Mac Sonucu",
                "probability": float(probability),
                "implied_probability": float(implied_probability),
                "odds": float(odds),
                "edge": edge,
            }

    goal_pick = _goal_market_pick(row, edge_threshold=edge_threshold)
    if goal_pick:
        if best_bet["edge"] is None or float(goal_pick["recommended_edge"]) > float(best_bet["edge"]):
            selected_label = goal_pick["recommended_label"]
            actual_family = "TOTALS" if selected_label in {"O25", "U25"} else "BTTS"
            best_bet = {
                "selected_label": selected_label,
                "market_family": actual_family,
                "recommended_market": goal_pick["recommended_market"],
                "probability": float(goal_pick["recommended_probability"]),
                "implied_probability": float(goal_pick["recommended_probability"] - goal_pick["recommended_edge"]),
                "odds": float(goal_pick["recommended_odds"]),
                "edge": float(goal_pick["recommended_edge"]),
            }

    double_chance_pick = _double_chance_pick(row, edge_threshold=edge_threshold)
    if double_chance_pick:
        if best_bet["edge"] is None or float(double_chance_pick["recommended_edge"]) > float(best_bet["edge"]):
            best_bet = {
                "selected_label": double_chance_pick["recommended_label"],
                "market_family": "DOUBLE_CHANCE",
                "recommended_market": double_chance_pick["recommended_market"],
                "probability": float(double_chance_pick["recommended_probability"]),
                "implied_probability": float(
                    double_chance_pick["recommended_probability"] - double_chance_pick["recommended_edge"]
                ),
                "odds": float(double_chance_pick["recommended_odds"]),
                "edge": float(double_chance_pick["recommended_edge"]),
            }

    handicap_pick = _handicap_10_pick(row, edge_threshold=edge_threshold)
    if handicap_pick:
        if best_bet["edge"] is None or float(handicap_pick["recommended_edge"]) > float(best_bet["edge"]):
            best_bet = {
                "selected_label": handicap_pick["recommended_label"],
                "market_family": "HANDICAP10",
                "recommended_market": handicap_pick["recommended_market"],
                "probability": float(handicap_pick["recommended_probability"]),
                "implied_probability": float(handicap_pick["recommended_probability"] - handicap_pick["recommended_edge"]),
                "odds": float(handicap_pick["recommended_odds"]),
                "edge": float(handicap_pick["recommended_edge"]),
            }

    return best_bet


def _summarize_period_performance(
    bets_df: pd.DataFrame,
    frequency: str,
    label: str,
) -> pd.DataFrame:
    if bets_df.empty:
        return pd.DataFrame()

    frame = bets_df.copy()
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"])
    frame["period_start"] = frame["kickoff_time"].dt.to_period(frequency).dt.start_time

    summary = (
        frame.groupby("period_start", as_index=False)
        .agg(
            bets=("selected_label", "count"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
            wins=("won", "sum"),
            ending_bankroll=("bankroll", "last"),
        )
        .sort_values("period_start")
    )
    summary["roi"] = np.where(summary["stake"] > 0, summary["pnl"] / summary["stake"], np.nan)
    summary["hit_rate"] = np.where(summary["bets"] > 0, summary["wins"] / summary["bets"], np.nan)
    summary["period_type"] = label
    return summary


def _build_backtest_curves_and_summaries(bets_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if bets_df.empty:
        empty = pd.DataFrame()
        return {
            "bankroll_curve": empty,
            "drawdown_curve": empty,
            "weekly_performance": empty,
            "monthly_performance": empty,
        }

    bankroll_curve = bets_df.copy()
    bankroll_curve["kickoff_time"] = pd.to_datetime(bankroll_curve["kickoff_time"])
    bankroll_curve = bankroll_curve.sort_values("kickoff_time").reset_index(drop=True)
    bankroll_curve["peak_bankroll"] = bankroll_curve["bankroll"].cummax()
    bankroll_curve["drawdown"] = np.where(
        bankroll_curve["peak_bankroll"] > 0,
        (bankroll_curve["peak_bankroll"] - bankroll_curve["bankroll"])
        / bankroll_curve["peak_bankroll"],
        0.0,
    )

    drawdown_curve = bankroll_curve[["kickoff_time", "drawdown", "bankroll", "peak_bankroll"]].copy()
    weekly_performance = _summarize_period_performance(bankroll_curve, "W", "weekly")
    monthly_performance = _summarize_period_performance(bankroll_curve, "M", "monthly")

    return {
        "bankroll_curve": bankroll_curve[["kickoff_time", "bankroll", "drawdown"]].copy(),
        "drawdown_curve": drawdown_curve,
        "weekly_performance": weekly_performance,
        "monthly_performance": monthly_performance,
    }


def run_backtest(
    model_type: str = "hist_gradient_boosting",
    n_splits: int = 5,
    min_train_rows: int = 120,
    edge_threshold: float = 0.03,
    initial_bankroll: float = 1000.0,
    stake_strategy: str = "flat",
    flat_stake: float = 25.0,
    kelly_fraction_multiplier: float = 0.25,
    max_kelly_stake_pct: float = 0.05,
) -> Dict[str, Any]:
    create_tables()
    df = build_training_dataframe()
    if df.empty or len(df) < max(80, min_train_rows + 20):
        raise ValueError("Backtest icin yeterli tarihsel mac verisi yok.")

    df = df.sort_values("kickoff_time").reset_index(drop=True)
    feature_columns = available_model_features(df)
    if not feature_columns:
        raise ValueError("Backtest icin kullanilabilir feature kolonu bulunamadi.")

    X = df[feature_columns].copy()
    y = df["target"].astype(int).copy()

    effective_splits = min(n_splits, max(2, len(df) // 80))
    tscv = TimeSeriesSplit(n_splits=effective_splits)

    out_of_sample_parts: List[pd.DataFrame] = []

    for split_no, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if len(train_idx) < min_train_rows:
            continue

        model = build_model_pipeline(model_type=model_type)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])

        part = df.iloc[test_idx].copy()
        part["split_no"] = split_no
        part["prob_home"] = probs[:, 0]
        part["prob_draw"] = probs[:, 1]
        part["prob_away"] = probs[:, 2]
        part["predicted_label"] = np.array(["H", "D", "A"])[np.argmax(probs, axis=1)]
        part["confidence"] = probs.max(axis=1)
        out_of_sample_parts.append(part)

    if not out_of_sample_parts:
        raise ValueError("Backtest fold'lari olusturulamadi. Daha fazla tarihsel veri gerekli.")

    predictions_df = pd.concat(out_of_sample_parts, ignore_index=True)

    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    max_drawdown = 0.0
    bet_rows: List[Dict[str, Any]] = []

    for _, row in predictions_df.sort_values("kickoff_time").iterrows():
        bet = _pick_bet(row=row, edge_threshold=edge_threshold)
        if bet.get("edge") is None:
            continue

        if stake_strategy == "kelly":
            raw_kelly = _kelly_fraction(bet["probability"], bet["odds"])
            stake = bankroll * min(
                max_kelly_stake_pct,
                raw_kelly * kelly_fraction_multiplier,
            )
        else:
            stake = flat_stake

        stake = max(0.0, min(stake, bankroll))
        if stake <= 0:
            continue

        if bet.get("market_family") == "TOTALS":
            actual_label = row["target_total25_label"]
        elif bet.get("market_family") == "BTTS":
            actual_label = row["target_btts_label"]
        elif bet.get("market_family") == "DOUBLE_CHANCE":
            actual_label = row["target_label"]
        elif bet.get("market_family") == "HANDICAP10":
            actual_label = row.get("target_handicap_10_label")
        else:
            actual_label = row["target_label"]

        if pd.isna(actual_label) or str(actual_label).strip() == "":
            continue

        won = bet["selected_label"] == actual_label
        pnl = stake * (bet["odds"] - 1.0) if won else -stake
        bankroll += pnl
        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown = 0.0 if peak_bankroll <= 0 else (peak_bankroll - bankroll) / peak_bankroll
        max_drawdown = max(max_drawdown, drawdown)

        bet_rows.append(
            {
                "kickoff_time": row["kickoff_time"],
                "league_name": row["league_name"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "selected_label": bet["selected_label"],
                "recommended_market": bet.get("recommended_market"),
                "market_family": bet.get("market_family"),
                "actual_label": actual_label,
                "odds": round(bet["odds"], 4),
                "probability": round(bet["probability"], 4),
                "implied_probability": round(bet["implied_probability"], 4),
                "edge": round(bet["edge"], 4),
                "stake": round(stake, 2),
                "won": won,
                "pnl": round(pnl, 2),
                "bankroll": round(bankroll, 2),
            }
        )

    bets_df = pd.DataFrame(bet_rows)
    total_staked = float(bets_df["stake"].sum()) if not bets_df.empty else 0.0
    total_pnl = float(bets_df["pnl"].sum()) if not bets_df.empty else 0.0
    roi = (total_pnl / total_staked) if total_staked > 0 else None
    hit_rate = float(bets_df["won"].mean()) if not bets_df.empty else None

    summary = {
        "rows_total": int(len(df)),
        "rows_out_of_sample": int(len(predictions_df)),
        "bets_placed": int(len(bets_df)),
        "initial_bankroll": float(initial_bankroll),
        "ending_bankroll": float(round(bankroll, 2)),
        "total_staked": float(round(total_staked, 2)),
        "total_pnl": float(round(total_pnl, 2)),
        "roi": float(roi) if roi is not None else None,
        "hit_rate": float(hit_rate) if hit_rate is not None else None,
        "max_drawdown": float(round(max_drawdown, 4)),
        "accuracy_oos": float(
            accuracy_score(
                predictions_df["target"].astype(int),
                predictions_df["predicted_label"].map({"H": 0, "D": 1, "A": 2}),
            )
        ),
        "log_loss_oos": float(
            log_loss(
                predictions_df["target"].astype(int),
                predictions_df[["prob_home", "prob_draw", "prob_away"]].values,
                labels=[0, 1, 2],
            )
        ),
        "brier_oos": float(
            multiclass_brier_score(
                predictions_df["target"].astype(int).values,
                predictions_df[["prob_home", "prob_draw", "prob_away"]].values,
                num_classes=3,
            )
        ),
        "edge_threshold": float(edge_threshold),
        "stake_strategy": stake_strategy,
        "model_type": model_type,
    }

    analytics = _build_backtest_curves_and_summaries(bets_df)

    return {
        "summary": summary,
        "predictions": predictions_df,
        "bets": bets_df,
        **analytics,
    }


def compare_backtest_strategies(
    model_type: str = "hist_gradient_boosting",
    edge_threshold: float = 0.03,
    initial_bankroll: float = 1000.0,
) -> Dict[str, Any]:
    flat_result = run_backtest(
        model_type=model_type,
        edge_threshold=edge_threshold,
        initial_bankroll=initial_bankroll,
        stake_strategy="flat",
    )
    kelly_result = run_backtest(
        model_type=model_type,
        edge_threshold=edge_threshold,
        initial_bankroll=initial_bankroll,
        stake_strategy="kelly",
    )

    comparison_rows = [
        flat_result["summary"],
        kelly_result["summary"],
    ]
    comparison_df = pd.DataFrame(comparison_rows)

    bankroll_curves = []
    drawdown_curves = []
    weekly_parts = []
    monthly_parts = []
    for label, result in [("flat", flat_result), ("kelly", kelly_result)]:
        bankroll_curve = result.get("bankroll_curve")
        drawdown_curve = result.get("drawdown_curve")
        weekly_performance = result.get("weekly_performance")
        monthly_performance = result.get("monthly_performance")

        if isinstance(bankroll_curve, pd.DataFrame) and not bankroll_curve.empty:
            curve = bankroll_curve.copy()
            curve["strategy"] = label
            bankroll_curves.append(curve[["kickoff_time", "bankroll", "strategy"]])

        if isinstance(drawdown_curve, pd.DataFrame) and not drawdown_curve.empty:
            curve = drawdown_curve.copy()
            curve["strategy"] = label
            drawdown_curves.append(curve[["kickoff_time", "drawdown", "strategy"]])

        if isinstance(weekly_performance, pd.DataFrame) and not weekly_performance.empty:
            frame = weekly_performance.copy()
            frame["strategy"] = label
            weekly_parts.append(frame)

        if isinstance(monthly_performance, pd.DataFrame) and not monthly_performance.empty:
            frame = monthly_performance.copy()
            frame["strategy"] = label
            monthly_parts.append(frame)

    bankroll_curve_df = (
        pd.concat(bankroll_curves, ignore_index=True) if bankroll_curves else pd.DataFrame()
    )
    drawdown_curve_df = (
        pd.concat(drawdown_curves, ignore_index=True) if drawdown_curves else pd.DataFrame()
    )
    weekly_df = pd.concat(weekly_parts, ignore_index=True) if weekly_parts else pd.DataFrame()
    monthly_df = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()

    return {
        "comparison": comparison_df,
        "bankroll_curve": bankroll_curve_df,
        "drawdown_curve": drawdown_curve_df,
        "weekly_performance": weekly_df,
        "monthly_performance": monthly_df,
        "flat": flat_result,
        "kelly": kelly_result,
    }


def compare_model_backtests(
    edge_threshold: float = 0.03,
    initial_bankroll: float = 1000.0,
    flat_stake: float = 25.0,
) -> Dict[str, Any]:
    variants = [
        ("Eski temel model", "logreg"),
        ("Yeni ana model", "hist_gradient_boosting"),
    ]
    rows: List[Dict[str, Any]] = []

    for label, model_type in variants:
        result = run_backtest(
            model_type=model_type,
            edge_threshold=edge_threshold,
            initial_bankroll=initial_bankroll,
            flat_stake=flat_stake,
            stake_strategy="flat",
        )
        summary = result["summary"].copy()
        summary["label"] = label
        rows.append(summary)

    comparison_df = pd.DataFrame(rows)[
        [
            "label",
            "model_type",
            "rows_total",
            "rows_out_of_sample",
            "bets_placed",
            "accuracy_oos",
            "log_loss_oos",
            "brier_oos",
            "roi",
            "hit_rate",
            "max_drawdown",
            "ending_bankroll",
        ]
    ]

    return {"comparison": comparison_df}
