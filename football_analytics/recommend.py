from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pandas as pd

from football_analytics.features import valid_1x2_odds, valid_simple_odds


OUTCOME_SPECS = [
    ("H", "Ev Sahibi", "prob_home", "edge_home", "market_home_odds"),
    ("D", "Beraberlik", "prob_draw", "edge_draw", "market_draw_odds"),
    ("A", "Deplasman", "prob_away", "edge_away", "market_away_odds"),
]

TOTAL_MARKET_SPECS = [
    ("O25", "2.5 Ust", "market_over_25_odds"),
    ("U25", "2.5 Alt", "market_under_25_odds"),
]

BTTS_MARKET_SPECS = [
    ("BYES", "KG Var", "market_btts_yes_odds"),
    ("BNO", "KG Yok", "market_btts_no_odds"),
]

DOUBLE_CHANCE_MARKET_SPECS = [
    ("1X", "Cifte Sans 1X", "market_double_chance_1x_odds"),
    ("12", "Cifte Sans 12", "market_double_chance_12_odds"),
    ("X2", "Cifte Sans X2", "market_double_chance_x2_odds"),
]

HANDICAP_10_SPECS = [
    ("H10H", "Handikap 1:0 / 1", "market_handicap_home_10_odds"),
    ("H10D", "Handikap 1:0 / 0", "market_handicap_draw_10_odds"),
    ("H10A", "Handikap 1:0 / 2", "market_handicap_away_10_odds"),
]

COUPON_PROFILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Temkinli": {
        "max_same_league": 1,
        "max_pick_odds": 2.85,
        "min_confidence": 0.49,
        "min_league_quality": 0.80,
        "max_low_quality_picks": 1,
        "avg_odds_cap": 1.85,
        "max_total_odds_cap": 24.0,
        "high_odds_cutoff": 2.45,
        "max_high_odds_picks": 2,
        "max_coupon_size": 4,
        "allow_low_tier_keywords": 1,
        "sort_priority": ["model_confidence", "recommended_edge", "recommended_odds"],
        "ascending": [False, False, True],
    },
    "Dengeli": {
        "max_same_league": 1,
        "max_pick_odds": 3.80,
        "min_confidence": 0.45,
        "min_league_quality": 0.76,
        "max_low_quality_picks": 1,
        "avg_odds_cap": 2.30,
        "max_total_odds_cap": 45.0,
        "high_odds_cutoff": 2.80,
        "max_high_odds_picks": 2,
        "max_coupon_size": 6,
        "allow_low_tier_keywords": 1,
        "sort_priority": ["recommended_edge", "model_confidence", "recommended_odds"],
        "ascending": [False, False, True],
    },
    "Yuksek Oran": {
        "max_same_league": 2,
        "max_pick_odds": 5.25,
        "min_confidence": 0.40,
        "min_league_quality": 0.70,
        "max_low_quality_picks": 2,
        "avg_odds_cap": 2.80,
        "max_total_odds_cap": 160.0,
        "high_odds_cutoff": 3.40,
        "max_high_odds_picks": 4,
        "max_coupon_size": 8,
        "allow_low_tier_keywords": 2,
        "sort_priority": ["expected_roi", "recommended_edge", "recommended_odds"],
        "ascending": [False, False, False],
    },
}


def _empty_coupon_summary(coupon_size: int) -> Dict[str, Any]:
    return {
        "bet_count": 0,
        "coupon_size": int(coupon_size),
        "requested_coupon_size": int(coupon_size),
        "total_odds": None,
        "combined_probability": None,
        "expected_roi": None,
        "bankroll": None,
        "flat_total_stake": None,
        "kelly_total_stake": None,
        "flat_expected_profit": None,
        "kelly_expected_profit": None,
        "coupon_profile": None,
        "max_total_odds_cap": None,
        "distinct_leagues": 0,
        "high_odds_pick_count": 0,
        "low_quality_pick_count": 0,
        "low_tier_pick_count": 0,
        "avg_pick_odds": None,
        "selection_summary_lines": ["Filtreleri gecen uygun kupon kombinasyonu bulunamadi."],
    }


def _coupon_profile_config(coupon_profile: str | None) -> Dict[str, Any]:
    return COUPON_PROFILE_CONFIGS.get(str(coupon_profile or "Temkinli"), COUPON_PROFILE_CONFIGS["Temkinli"])


def _max_total_odds_for_profile(coupon_size: int, avg_odds_cap: float) -> float:
    return float(avg_odds_cap) ** max(1, int(coupon_size))


LOW_TIER_LEAGUE_KEYWORDS = (
    "amatör",
    "amator",
    "u19",
    "u20",
    "u21",
    "u23",
    "genclik",
    "gençlik",
    "hazirlik",
    "hazırlık",
    "(k)",
    "kadin",
    "kadın",
    "women",
    "reserves",
    "ii",
    "iii",
)


def _is_low_tier_league_name(league_name: Any) -> bool:
    league_text = str(league_name or "").strip().lower()
    if not league_text:
        return True
    return any(keyword in league_text for keyword in LOW_TIER_LEAGUE_KEYWORDS)


def _row_passes_coupon_profile(
    row: pd.Series,
    profile_config: Dict[str, Any],
    low_quality_pick_count: int,
    high_odds_pick_count: int,
    low_tier_keyword_count: int,
) -> bool:
    confidence = row.get("model_confidence")
    if pd.notna(confidence) and float(confidence) < float(profile_config["min_confidence"]):
        return False

    league_quality = row.get("league_quality_index")
    if pd.notna(league_quality) and float(league_quality) < float(profile_config["min_league_quality"]):
        if low_quality_pick_count >= int(profile_config["max_low_quality_picks"]):
            return False

    odds_value = row.get("recommended_odds")
    if pd.notna(odds_value) and float(odds_value) > float(profile_config["max_pick_odds"]):
        return False
    if pd.notna(odds_value) and float(odds_value) > float(profile_config["high_odds_cutoff"]):
        if high_odds_pick_count >= int(profile_config["max_high_odds_picks"]):
            return False

    if _is_low_tier_league_name(row.get("league_name")):
        if low_tier_keyword_count >= int(profile_config["allow_low_tier_keywords"]):
            return False

    return True


def _kelly_fraction(probability: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - probability
    raw = ((b * probability) - q) / b
    return max(0.0, raw)


def _normalize_binary_probs(prob_yes: float, prob_no: float) -> tuple[float, float]:
    values = np.array([prob_yes, prob_no], dtype=float)
    values = np.clip(values, 1e-6, None)
    total = float(values.sum())
    if total <= 0:
        return 0.5, 0.5
    values = values / total
    return float(values[0]), float(values[1])


def _binary_market_probability_from_odds(odds_for: Any, odds_against: Any) -> tuple[float | None, float | None]:
    if not valid_simple_odds(odds_for) or not valid_simple_odds(odds_against):
        return None, None
    inv_for = 1.0 / float(odds_for)
    inv_against = 1.0 / float(odds_against)
    total = inv_for + inv_against
    if total <= 0:
        return None, None
    return inv_for / total, inv_against / total


def _normalize_triplet_probabilities(a: float, b: float, c: float) -> tuple[float, float, float]:
    values = np.array([a, b, c], dtype=float)
    values = np.clip(values, 1e-6, None)
    total = float(values.sum())
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    values = values / total
    return float(values[0]), float(values[1]), float(values[2])


def _triplet_implied_probabilities(odds_a: Any, odds_b: Any, odds_c: Any) -> tuple[float | None, float | None, float | None]:
    if not all(valid_simple_odds(value) for value in (odds_a, odds_b, odds_c)):
        return None, None, None
    inv_a = 1.0 / float(odds_a)
    inv_b = 1.0 / float(odds_b)
    inv_c = 1.0 / float(odds_c)
    return _normalize_triplet_probabilities(inv_a, inv_b, inv_c)


def _poisson_total_over_25_probability(row: pd.Series) -> tuple[float, float]:
    mu_home = (
        float(row.get("home_goals_for_last5", 1.0)) + float(row.get("away_goals_against_last5", 1.0))
    ) / 2.0
    mu_away = (
        float(row.get("away_goals_for_last5", 1.0)) + float(row.get("home_goals_against_last5", 1.0))
    ) / 2.0
    total_mean = max(0.2, min(6.0, mu_home + mu_away))
    under = math.exp(-total_mean) * (1.0 + total_mean + (total_mean**2 / 2.0))
    over = max(0.0, min(1.0, 1.0 - under))
    return over, 1.0 - over


def _btts_probability(row: pd.Series) -> tuple[float, float]:
    mu_home = (
        float(row.get("home_goals_for_last5", 1.0)) + float(row.get("away_goals_against_last5", 1.0))
    ) / 2.0
    mu_away = (
        float(row.get("away_goals_for_last5", 1.0)) + float(row.get("home_goals_against_last5", 1.0))
    ) / 2.0
    mu_home = max(0.15, min(4.0, mu_home))
    mu_away = max(0.15, min(4.0, mu_away))
    yes = (1.0 - math.exp(-mu_home)) * (1.0 - math.exp(-mu_away))
    yes = max(0.0, min(1.0, yes))
    return yes, 1.0 - yes


def _expected_goal_means(row: pd.Series) -> tuple[float, float]:
    mu_home = (
        float(row.get("home_goals_for_last5", 1.0)) + float(row.get("away_goals_against_last5", 1.0))
    ) / 2.0
    mu_away = (
        float(row.get("away_goals_for_last5", 1.0)) + float(row.get("home_goals_against_last5", 1.0))
    ) / 2.0
    return max(0.15, min(4.5, mu_home)), max(0.15, min(4.5, mu_away))


def _poisson_probs(mu: float, max_goals: int = 7) -> np.ndarray:
    return np.array(
        [math.exp(-mu) * (mu**goals) / math.factorial(goals) for goals in range(max_goals + 1)],
        dtype=float,
    )


def _handicap_10_probabilities(row: pd.Series) -> tuple[float, float, float]:
    mu_home, mu_away = _expected_goal_means(row)
    home_probs = _poisson_probs(mu_home)
    away_probs = _poisson_probs(mu_away)

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for home_goals, home_prob in enumerate(home_probs):
        for away_goals, away_prob in enumerate(away_probs):
            joint = float(home_prob * away_prob)
            adjusted_home = home_goals + 1
            if adjusted_home > away_goals:
                home_win += joint
            elif adjusted_home == away_goals:
                draw += joint
            else:
                away_win += joint
    return _normalize_triplet_probabilities(home_win, draw, away_win)


def _goal_market_pick(row: pd.Series, edge_threshold: float) -> Dict[str, Any]:
    best_pick: Dict[str, Any] = {}

    over_prob, under_prob = _poisson_total_over_25_probability(row)
    market_over = row.get("market_over_25_odds")
    market_under = row.get("market_under_25_odds")
    imp_over, imp_under = _binary_market_probability_from_odds(market_over, market_under)

    if imp_over is not None and imp_under is not None:
        over_prob = (0.7 * over_prob) + (0.3 * imp_over)
        under_prob = (0.7 * under_prob) + (0.3 * imp_under)
        over_prob, under_prob = _normalize_binary_probs(over_prob, under_prob)

    for label, market_name, odds_col, probability in [
        ("O25", "2.5 Ust", "market_over_25_odds", over_prob),
        ("U25", "2.5 Alt", "market_under_25_odds", under_prob),
    ]:
        odds = row.get(odds_col)
        if not valid_simple_odds(odds):
            continue
        implied_for, implied_against = _binary_market_probability_from_odds(
            odds,
            market_under if label == "O25" else market_over,
        )
        edge = None if implied_for is None else probability - implied_for
        if edge is None or edge < edge_threshold:
            continue
        expected_roi = (float(probability) * float(odds)) - 1.0
        if not best_pick or float(edge) > best_pick["recommended_edge"]:
            best_pick = {
                "recommended_label": label,
                "recommended_market": market_name,
                "recommended_probability": float(probability),
                "recommended_edge": float(edge),
                "recommended_odds": float(odds),
                "expected_roi": expected_roi,
                "fair_odds": (1.0 / float(probability)) if float(probability) > 0 else np.nan,
            }

    btts_yes_prob, btts_no_prob = _btts_probability(row)
    btts_yes = row.get("market_btts_yes_odds")
    btts_no = row.get("market_btts_no_odds")
    imp_yes, imp_no = _binary_market_probability_from_odds(btts_yes, btts_no)
    if imp_yes is not None and imp_no is not None:
        btts_yes_prob = (0.7 * btts_yes_prob) + (0.3 * imp_yes)
        btts_no_prob = (0.7 * btts_no_prob) + (0.3 * imp_no)
        btts_yes_prob, btts_no_prob = _normalize_binary_probs(btts_yes_prob, btts_no_prob)

    for label, market_name, odds_col, probability in [
        ("BYES", "KG Var", "market_btts_yes_odds", btts_yes_prob),
        ("BNO", "KG Yok", "market_btts_no_odds", btts_no_prob),
    ]:
        odds = row.get(odds_col)
        if not valid_simple_odds(odds):
            continue
        implied_for, implied_against = _binary_market_probability_from_odds(
            odds,
            btts_no if label == "BYES" else btts_yes,
        )
        edge = None if implied_for is None else probability - implied_for
        if edge is None or edge < edge_threshold:
            continue
        expected_roi = (float(probability) * float(odds)) - 1.0
        if not best_pick or float(edge) > best_pick["recommended_edge"]:
            best_pick = {
                "recommended_label": label,
                "recommended_market": market_name,
                "recommended_probability": float(probability),
                "recommended_edge": float(edge),
                "recommended_odds": float(odds),
                "expected_roi": expected_roi,
                "fair_odds": (1.0 / float(probability)) if float(probability) > 0 else np.nan,
            }

    return best_pick


def _double_chance_pick(row: pd.Series, edge_threshold: float) -> Dict[str, Any]:
    best_pick: Dict[str, Any] = {}
    prob_home = float(row.get("prob_home", 0.0) or 0.0)
    prob_draw = float(row.get("prob_draw", 0.0) or 0.0)
    prob_away = float(row.get("prob_away", 0.0) or 0.0)

    model_probs = {
        "1X": prob_home + prob_draw,
        "12": prob_home + prob_away,
        "X2": prob_draw + prob_away,
    }

    for label, market_name, odds_col in DOUBLE_CHANCE_MARKET_SPECS:
        odds = row.get(odds_col)
        if not valid_simple_odds(odds):
            continue
        implied_probability = min(0.999, max(0.001, 1.0 / float(odds)))
        probability = min(0.999, max(0.001, model_probs[label]))
        edge = probability - implied_probability
        if edge < edge_threshold:
            continue
        expected_roi = (probability * float(odds)) - 1.0
        if not best_pick or float(edge) > best_pick["recommended_edge"]:
            best_pick = {
                "recommended_label": label,
                "recommended_market": market_name,
                "recommended_probability": float(probability),
                "recommended_edge": float(edge),
                "recommended_odds": float(odds),
                "expected_roi": expected_roi,
                "fair_odds": (1.0 / float(probability)) if float(probability) > 0 else np.nan,
            }

    return best_pick


def _handicap_10_pick(row: pd.Series, edge_threshold: float) -> Dict[str, Any]:
    best_pick: Dict[str, Any] = {}
    probs = _handicap_10_probabilities(row)
    implied = _triplet_implied_probabilities(
        row.get("market_handicap_home_10_odds"),
        row.get("market_handicap_draw_10_odds"),
        row.get("market_handicap_away_10_odds"),
    )
    if implied[0] is None:
        return best_pick

    for (label, market_name, odds_col), probability, implied_probability in zip(
        HANDICAP_10_SPECS,
        probs,
        implied,
    ):
        odds = row.get(odds_col)
        if not valid_simple_odds(odds):
            continue
        edge = float(probability) - float(implied_probability)
        if edge < edge_threshold:
            continue
        expected_roi = (float(probability) * float(odds)) - 1.0
        if not best_pick or float(edge) > best_pick["recommended_edge"]:
            best_pick = {
                "recommended_label": label,
                "recommended_market": market_name,
                "recommended_probability": float(probability),
                "recommended_edge": float(edge),
                "recommended_odds": float(odds),
                "expected_roi": expected_roi,
                "fair_odds": (1.0 / float(probability)) if float(probability) > 0 else np.nan,
            }

    return best_pick


def _pick_recommendation(row: pd.Series, edge_threshold: float) -> Dict[str, Any]:
    best_pick: Dict[str, Any] = {}

    for label, market_name, prob_col, edge_col, odds_col in OUTCOME_SPECS:
        probability = row.get(prob_col)
        edge = row.get(edge_col)
        odds = row.get(odds_col)

        if pd.isna(probability) or pd.isna(edge) or pd.isna(odds):
            continue
        if float(edge) < edge_threshold:
            continue

        expected_roi = (float(probability) * float(odds)) - 1.0
        if not best_pick or float(edge) > best_pick["recommended_edge"]:
            best_pick = {
                "recommended_label": label,
                "recommended_market": market_name,
                "recommended_probability": float(probability),
                "recommended_edge": float(edge),
                "recommended_odds": float(odds),
                "expected_roi": expected_roi,
                "fair_odds": (1.0 / float(probability)) if float(probability) > 0 else np.nan,
            }

    goal_pick = _goal_market_pick(row, edge_threshold=edge_threshold)
    if goal_pick and (
        not best_pick or float(goal_pick["recommended_edge"]) > float(best_pick["recommended_edge"])
    ):
        best_pick = goal_pick

    double_chance_pick = _double_chance_pick(row, edge_threshold=edge_threshold)
    if double_chance_pick and (
        not best_pick
        or float(double_chance_pick["recommended_edge"]) > float(best_pick["recommended_edge"])
    ):
        best_pick = double_chance_pick

    handicap_pick = _handicap_10_pick(row, edge_threshold=edge_threshold)
    if handicap_pick and (
        not best_pick or float(handicap_pick["recommended_edge"]) > float(best_pick["recommended_edge"])
    ):
        best_pick = handicap_pick

    return best_pick


def _match_rejection_reason(
    row: pd.Series,
    edge_threshold: float,
    source_filter: str | None = None,
    min_confidence_threshold: float = 0.45,
) -> Dict[str, Any]:
    source_name = str(row.get("source", "")).strip().lower()
    source_filter_value = str(source_filter).strip().lower() if source_filter else None

    if source_filter_value and source_name != source_filter_value:
        return {
            "is_recommended": False,
            "reason": f"Kaynak filtresi disinda ({source_name or 'bilinmiyor'})",
            "best_edge": None,
            "best_label": None,
        }

    if "betting_ready" in row and not bool(row.get("betting_ready")):
        return {
            "is_recommended": False,
            "reason": str(row.get("betting_ready_reason") or "Bahis icin hazir degil"),
            "best_edge": None,
            "best_label": None,
        }

    if "odds_consistency_ok" in row and not bool(row.get("odds_consistency_ok")):
        market_name = str(row.get("odds_consistency_market") or "market")
        gap_value = row.get("odds_consistency_gap")
        ref_source = str(row.get("odds_consistency_reference_source") or "diger kaynak")
        gap_text = f"{float(gap_value):.2%}" if pd.notna(gap_value) else "yuksek"
        return {
            "is_recommended": False,
            "reason": f"Oran sapmasi yuksek ({market_name}, {ref_source}, {gap_text})",
            "best_edge": None,
            "best_label": None,
        }

    league_name = str(row.get("league_name", "") or "").strip()
    if not league_name or league_name.lower() == "unknown league":
        return {
            "is_recommended": False,
            "reason": "Lig bilgisi yok",
            "best_edge": None,
            "best_label": None,
        }

    home_odds = row.get("market_home_odds")
    draw_odds = row.get("market_draw_odds")
    away_odds = row.get("market_away_odds")
    over_odds = row.get("market_over_25_odds")
    under_odds = row.get("market_under_25_odds")
    btts_yes_odds = row.get("market_btts_yes_odds")
    btts_no_odds = row.get("market_btts_no_odds")
    dc_1x = row.get("market_double_chance_1x_odds")
    dc_12 = row.get("market_double_chance_12_odds")
    dc_x2 = row.get("market_double_chance_x2_odds")
    handicap_h = row.get("market_handicap_home_10_odds")
    handicap_d = row.get("market_handicap_draw_10_odds")
    handicap_a = row.get("market_handicap_away_10_odds")

    has_valid_1x2 = valid_1x2_odds(home_odds, draw_odds, away_odds)
    has_valid_totals = valid_simple_odds(over_odds) and valid_simple_odds(under_odds)
    has_valid_btts = valid_simple_odds(btts_yes_odds) and valid_simple_odds(btts_no_odds)
    has_valid_dc = all(valid_simple_odds(value) for value in (dc_1x, dc_12, dc_x2))
    has_valid_handicap = all(valid_simple_odds(value) for value in (handicap_h, handicap_d, handicap_a))

    if not has_valid_1x2 and not has_valid_totals and not has_valid_btts and not has_valid_dc and not has_valid_handicap:
        reason = "Desteklenen market oranlari yok"
        if source_name == "nesine":
            reason = "Kaynak marketi yok"
        return {
            "is_recommended": False,
            "reason": reason,
            "best_edge": None,
            "best_label": None,
        }

    confidence = row.get("confidence")
    if pd.notna(confidence) and float(confidence) < min_confidence_threshold:
        return {
            "is_recommended": False,
            "reason": f"Confidence dusuk ({float(confidence):.2%})",
            "best_edge": None,
            "best_label": None,
        }

    best_pick = _pick_recommendation(row, edge_threshold=edge_threshold)
    if not best_pick:
        return {
            "is_recommended": False,
            "reason": "Edge hesaplanamiyor",
            "best_edge": None,
            "best_label": None,
        }

    best_label = str(best_pick["recommended_label"])
    best_edge = float(best_pick["recommended_edge"])
    best_odds = best_pick.get("recommended_odds")
    if best_edge < edge_threshold:
        return {
            "is_recommended": False,
            "reason": f"Edge yetersiz ({best_edge:.2%})",
            "best_edge": best_edge,
            "best_label": best_label,
        }
    if best_edge > 0.20:
        return {
            "is_recommended": False,
            "reason": f"Edge asiri yuksek ({best_edge:.2%})",
            "best_edge": best_edge,
            "best_label": best_label,
        }
    if best_odds is not None and pd.notna(best_odds) and float(best_odds) > 4.75:
        return {
            "is_recommended": False,
            "reason": f"Oran cok yuksek ({float(best_odds):.2f})",
            "best_edge": best_edge,
            "best_label": best_label,
        }

    return {
        "is_recommended": True,
        "reason": "Kupona uygun",
        "best_edge": best_edge,
        "best_label": best_label,
    }


def build_recommendation_audit(
    predictions: pd.DataFrame,
    edge_threshold: float = 0.03,
    source_filter: str | None = None,
    min_confidence_threshold: float = 0.45,
) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    rows = []
    for _, row in predictions.iterrows():
        decision = _match_rejection_reason(
            row=row,
            edge_threshold=edge_threshold,
            source_filter=source_filter,
            min_confidence_threshold=min_confidence_threshold,
        )
        rows.append(
            {
                "kickoff_time": row.get("kickoff_time"),
                "league_name": row.get("league_name"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "source": row.get("source"),
                "status": row.get("status"),
                "reason": decision["reason"],
                "is_recommended": decision["is_recommended"],
                "best_edge": round(decision["best_edge"], 4)
                if decision["best_edge"] is not None
                else np.nan,
                "best_label": decision["best_label"],
                "market_home_odds": row.get("market_home_odds"),
                "market_draw_odds": row.get("market_draw_odds"),
                "market_away_odds": row.get("market_away_odds"),
                "market_over_25_odds": row.get("market_over_25_odds"),
                "market_under_25_odds": row.get("market_under_25_odds"),
                "market_btts_yes_odds": row.get("market_btts_yes_odds"),
                "market_btts_no_odds": row.get("market_btts_no_odds"),
                "market_double_chance_1x_odds": row.get("market_double_chance_1x_odds"),
                "market_double_chance_12_odds": row.get("market_double_chance_12_odds"),
                "market_double_chance_x2_odds": row.get("market_double_chance_x2_odds"),
                "market_handicap_home_10_odds": row.get("market_handicap_home_10_odds"),
                "market_handicap_draw_10_odds": row.get("market_handicap_draw_10_odds"),
                "market_handicap_away_10_odds": row.get("market_handicap_away_10_odds"),
                "confidence": row.get("confidence"),
                "match_id": row.get("match_id"),
            }
        )

    audit_df = pd.DataFrame(rows)
    if audit_df.empty:
        return audit_df

    return audit_df.sort_values(
        ["is_recommended", "kickoff_time", "league_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_betting_recommendations(
    predictions: pd.DataFrame,
    bankroll: float = 1000.0,
    flat_stake: float = 25.0,
    edge_threshold: float = 0.03,
    kelly_fraction_multiplier: float = 0.25,
    max_kelly_stake_pct: float = 0.05,
    coupon_size: int = 4,
    source_filter: str | None = None,
    min_confidence_threshold: float = 0.45,
    coupon_profile: str = "Temkinli",
) -> Dict[str, Any]:
    if predictions is None or predictions.empty:
        empty = pd.DataFrame()
        return {
            "recommended_bets": empty,
            "coupon_picks": empty,
            "coupon_summary": {},
            "match_audit": empty,
        }

    df = predictions.copy()
    audit_df = build_recommendation_audit(
        predictions=df,
        edge_threshold=edge_threshold,
        source_filter=source_filter,
        min_confidence_threshold=min_confidence_threshold,
    )

    if not audit_df.empty:
        eligible_match_ids = audit_df[audit_df["is_recommended"]]["match_id"].dropna().tolist()
        if "match_id" in df.columns:
            df = df[df["match_id"].isin(eligible_match_ids)].copy()

    if df.empty:
        empty = pd.DataFrame()
        return {
            "recommended_bets": empty,
            "coupon_picks": empty,
            "coupon_summary": _empty_coupon_summary(coupon_size),
            "match_audit": audit_df,
        }

    recommendation_rows = []

    for _, row in df.iterrows():
        pick = _pick_recommendation(row, edge_threshold=edge_threshold)
        if not pick:
            continue

        kelly_fraction = _kelly_fraction(
            probability=pick["recommended_probability"],
            odds=pick["recommended_odds"],
        )
        kelly_stake = bankroll * min(
            max_kelly_stake_pct,
            kelly_fraction * kelly_fraction_multiplier,
        )
        risk_band = "Dusuk"
        if pick["recommended_edge"] >= 0.08 and row.get("confidence", 0.0) >= 0.55:
            risk_band = "Gorece guclu"
        elif pick["recommended_edge"] >= 0.05:
            risk_band = "Orta"

        recommendation_rows.append(
            {
                "kickoff_time": row.get("kickoff_time"),
                "league_name": row.get("league_name"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "recommended_label": pick["recommended_label"],
                "recommended_market": pick["recommended_market"],
                "recommended_odds": round(pick["recommended_odds"], 4),
                "recommended_probability": round(pick["recommended_probability"], 4),
                "recommended_edge": round(pick["recommended_edge"], 4),
                "expected_roi": round(pick["expected_roi"], 4),
                "fair_odds": round(pick["fair_odds"], 4)
                if pd.notna(pick["fair_odds"])
                else np.nan,
                "model_confidence": round(float(row.get("confidence", np.nan)), 4),
                "flat_stake": round(float(flat_stake), 2),
                "kelly_stake": round(max(0.0, kelly_stake), 2),
                "flat_expected_profit": round(float(flat_stake) * pick["expected_roi"], 2),
                "kelly_expected_profit": round(max(0.0, kelly_stake) * pick["expected_roi"], 2),
                "risk_band": risk_band,
                "source": row.get("source"),
                "odds_source": row.get("odds_source"),
                "status": row.get("status"),
                "match_id": row.get("match_id"),
                "league_quality_index": row.get("league_quality_index"),
            }
        )

    recommended_bets = pd.DataFrame(recommendation_rows)
    if recommended_bets.empty:
        return {
            "recommended_bets": recommended_bets,
            "coupon_picks": recommended_bets,
            "coupon_summary": _empty_coupon_summary(coupon_size),
            "match_audit": audit_df,
        }

    recommended_bets = recommended_bets.sort_values(
        ["recommended_edge", "model_confidence", "recommended_odds"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    recommended_bets["selection_rank"] = np.arange(1, len(recommended_bets) + 1)
    coupon_picks, coupon_summary = build_coupon_from_recommendations(
        recommended_bets=recommended_bets,
        coupon_size=coupon_size,
        bankroll=bankroll,
        sort_by="recommended_edge",
        coupon_profile=coupon_profile,
    )

    return {
        "recommended_bets": recommended_bets,
        "coupon_picks": coupon_picks,
        "coupon_summary": coupon_summary,
        "match_audit": audit_df,
    }


def build_coupon_from_recommendations(
    recommended_bets: pd.DataFrame,
    coupon_size: int = 4,
    bankroll: float | None = None,
    sort_by: str = "recommended_edge",
    ascending: bool = False,
    max_same_league: int = 1,
    coupon_profile: str = "Temkinli",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if recommended_bets is None or recommended_bets.empty:
        return pd.DataFrame(), _empty_coupon_summary(coupon_size)

    profile_config = _coupon_profile_config(coupon_profile)
    requested_coupon_size = max(1, int(coupon_size))
    coupon_size = min(requested_coupon_size, int(profile_config.get("max_coupon_size", requested_coupon_size)))
    if sort_by not in recommended_bets.columns:
        sort_by = "recommended_edge"

    sort_columns = [column for column in profile_config["sort_priority"] if column in recommended_bets.columns]
    if "segment_priority" in recommended_bets.columns and "segment_priority" not in sort_columns:
        sort_columns = ["segment_priority"] + sort_columns
    if "segment_score" in recommended_bets.columns and "segment_score" not in sort_columns:
        insert_at = 1 if sort_columns else 0
        sort_columns = sort_columns[:insert_at] + ["segment_score"] + sort_columns[insert_at:]
    if not sort_columns:
        sort_columns = [sort_by]
    ascending_flags = []
    for column in sort_columns:
        if column in {"segment_priority", "segment_score"}:
            ascending_flags.append(False)
        else:
            ascending_flags.append(None)
    profile_flags = profile_config["ascending"]
    profile_index = 0
    for idx, flag in enumerate(ascending_flags):
        if flag is None:
            if profile_index < len(profile_flags):
                ascending_flags[idx] = profile_flags[profile_index]
            else:
                ascending_flags[idx] = ascending
            profile_index += 1
    if len(ascending_flags) < len(sort_columns):
        ascending_flags += [ascending] * (len(sort_columns) - len(ascending_flags))

    ranked = (
        recommended_bets.sort_values(
            sort_columns,
            ascending=ascending_flags,
        )
        .copy()
        .reset_index(drop=True)
    )

    picked_rows = []
    league_counts: Dict[str, int] = {}
    used_teams = set()
    low_quality_pick_count = 0
    high_odds_pick_count = 0
    low_tier_keyword_count = 0
    total_odds_cap = min(
        _max_total_odds_for_profile(coupon_size, profile_config["avg_odds_cap"]),
        float(profile_config["max_total_odds_cap"]),
    )
    running_total_odds = 1.0

    for _, row in ranked.iterrows():
        if len(picked_rows) >= max(1, int(coupon_size)):
            break
        league_name = str(row.get("league_name") or "Unknown League")
        home_team = str(row.get("home_team") or "")
        away_team = str(row.get("away_team") or "")
        if league_counts.get(league_name, 0) >= min(max_same_league, int(profile_config["max_same_league"])):
            continue
        if home_team in used_teams or away_team in used_teams:
            continue
        if not _row_passes_coupon_profile(
            row,
            profile_config,
            low_quality_pick_count,
            high_odds_pick_count,
            low_tier_keyword_count,
        ):
            continue
        pick_odds = float(row.get("recommended_odds") or 1.0)
        if (running_total_odds * pick_odds) > total_odds_cap:
            continue
        picked_rows.append(row.to_dict())
        league_counts[league_name] = league_counts.get(league_name, 0) + 1
        used_teams.add(home_team)
        used_teams.add(away_team)
        running_total_odds *= pick_odds
        league_quality = row.get("league_quality_index")
        if pd.notna(league_quality) and float(league_quality) < 0.82:
            low_quality_pick_count += 1
        if pick_odds > float(profile_config["high_odds_cutoff"]):
            high_odds_pick_count += 1
        if _is_low_tier_league_name(league_name):
            low_tier_keyword_count += 1

    if len(picked_rows) < max(1, int(coupon_size)):
        for _, row in ranked.iterrows():
            if len(picked_rows) >= max(1, int(coupon_size)):
                break
            key = (
                row.get("kickoff_time"),
                row.get("home_team"),
                row.get("away_team"),
                row.get("recommended_label"),
            )
            existing_keys = {
                (
                    item.get("kickoff_time"),
                    item.get("home_team"),
                    item.get("away_team"),
                    item.get("recommended_label"),
                )
                for item in picked_rows
            }
            if key in existing_keys:
                continue
            if not _row_passes_coupon_profile(
                row,
                profile_config,
                low_quality_pick_count,
                high_odds_pick_count,
                low_tier_keyword_count,
            ):
                continue
            pick_odds = float(row.get("recommended_odds") or 1.0)
            if (running_total_odds * pick_odds) > total_odds_cap:
                continue
            picked_rows.append(row.to_dict())
            running_total_odds *= pick_odds
            league_quality = row.get("league_quality_index")
            if pd.notna(league_quality) and float(league_quality) < 0.82:
                low_quality_pick_count += 1
            if pick_odds > float(profile_config["high_odds_cutoff"]):
                high_odds_pick_count += 1
            if _is_low_tier_league_name(row.get("league_name")):
                low_tier_keyword_count += 1

    coupon_picks = pd.DataFrame(picked_rows).reset_index(drop=True)
    if coupon_picks.empty:
        empty_summary = _empty_coupon_summary(coupon_size)
        empty_summary.update(
            {
                "requested_coupon_size": int(requested_coupon_size),
                "coupon_profile": coupon_profile,
                "max_total_odds_cap": round(total_odds_cap, 2),
                "selection_summary_lines": [
                    f"Istenen {requested_coupon_size} secim icin filtreleri gecen kupon kalmadi.",
                    f"{coupon_profile} modu toplam oran tavani {round(total_odds_cap, 2)} ve lig/kalite filtreleriyle daha secici davrandi.",
                ],
            }
        )
        return coupon_picks, empty_summary

    total_odds = float(coupon_picks["recommended_odds"].prod()) if not coupon_picks.empty else None
    combined_probability = (
        float(coupon_picks["recommended_probability"].prod()) if not coupon_picks.empty else None
    )
    coupon_expected_roi = (
        (combined_probability * total_odds) - 1.0
        if total_odds is not None and combined_probability is not None
        else None
    )

    coupon_summary = {
        "bet_count": int(len(coupon_picks)),
        "coupon_size": int(coupon_size),
        "requested_coupon_size": int(requested_coupon_size),
        "total_odds": round(total_odds, 4) if total_odds is not None else None,
        "combined_probability": round(combined_probability, 4)
        if combined_probability is not None
        else None,
        "expected_roi": round(coupon_expected_roi, 4)
        if coupon_expected_roi is not None
        else None,
        "bankroll": float(bankroll) if bankroll is not None else None,
        "flat_total_stake": round(float(coupon_picks["flat_stake"].sum()), 2),
        "kelly_total_stake": round(float(coupon_picks["kelly_stake"].sum()), 2),
        "flat_expected_profit": round(float(coupon_picks["flat_expected_profit"].sum()), 2),
        "kelly_expected_profit": round(float(coupon_picks["kelly_expected_profit"].sum()), 2),
        "coupon_profile": coupon_profile,
        "max_total_odds_cap": round(total_odds_cap, 2),
        "distinct_leagues": int(coupon_picks["league_name"].nunique()) if not coupon_picks.empty else 0,
        "high_odds_pick_count": int(high_odds_pick_count),
        "low_quality_pick_count": int(low_quality_pick_count),
        "low_tier_pick_count": int(low_tier_keyword_count),
        "avg_pick_odds": round(float(coupon_picks["recommended_odds"].mean()), 2)
        if not coupon_picks.empty
        else None,
    }
    summary_lines = [
        f"{coupon_summary['bet_count']} mac, {coupon_summary['distinct_leagues']} farkli ligden secildi.",
        f"Ortalama tek mac orani {coupon_summary['avg_pick_odds']:.2f} ve toplam oran tavani {coupon_summary['max_total_odds_cap']:.2f} uygulandi."
        if coupon_summary["avg_pick_odds"] is not None
        else f"Toplam oran tavani {coupon_summary['max_total_odds_cap']:.2f} uygulandi.",
    ]
    if coupon_summary["high_odds_pick_count"] > 0:
        summary_lines.append(
            f"Yuksek oranli secim sayisi {coupon_summary['high_odds_pick_count']} ile sinirlandi."
        )
    else:
        summary_lines.append("Yuksek oranli secim alinmadi; kupon daha dengeli tutuldu.")
    if coupon_summary["low_tier_pick_count"] > 0:
        summary_lines.append(
            f"Alt lig / ozel turnuva secimi {coupon_summary['low_tier_pick_count']} adetle sinirlandi."
        )
    else:
        summary_lines.append("Alt lig / ozel turnuva secimlerinden kacinarak daha guvenli havuz kullanildi.")
    if requested_coupon_size > coupon_summary["bet_count"]:
        summary_lines.append(
            f"Istenen {requested_coupon_size} secim yerine filtreleri gecen {coupon_summary['bet_count']} mac birakildi."
        )
    coupon_summary["selection_summary_lines"] = summary_lines
    return coupon_picks, coupon_summary
