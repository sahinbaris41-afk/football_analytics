from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import joinedload

from football_analytics.db import Match, build_preferred_odds_lookup, get_session, latest_1x2_odds

ROLLING_WINDOW = 5

MODEL_FEATURES = [
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
    "home_home_points_last5",
    "away_away_points_last5",
    "venue_form_diff",
    "home_home_goal_diff_last5",
    "away_away_goal_diff_last5",
    "venue_goal_diff_form_diff",
    "home_home_goals_for_last5",
    "away_away_goals_for_last5",
    "home_home_goals_against_last5",
    "away_away_goals_against_last5",
    "home_home_matches_played",
    "away_away_matches_played",
    "venue_experience_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_matches_played",
    "away_matches_played",
    "imp_home",
    "imp_draw",
    "imp_away",
    "market_overround",
]

OPTIONAL_ODDS_FEATURES = [
    "imp_home",
    "imp_draw",
    "imp_away",
    "market_overround",
]

TRUSTED_LEAGUE_EXCLUDE_TOKENS = [
    "amat",
    "u19",
    "u20",
    "u21",
    "u23",
    "reserves",
    "reserve",
    "genclik",
    "gençlik",
    "women",
    "kadin",
    "kadın",
    "futsal",
    "e-football",
    "e-basket",
    "e-spor",
]

MIN_TRUSTED_LEAGUE_MATCH_COUNT = 120


@dataclass
class TeamState:
    elo: float = 1500.0
    points: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    goal_diff: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    goals_for: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    goals_against: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    home_points: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    away_points: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    home_goal_diff: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    away_goal_diff: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    home_goals_for: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    away_goals_for: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    home_goals_against: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    away_goals_against: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    last_match_date: Optional[datetime] = None
    matches_played: int = 0
    home_matches_played: int = 0
    away_matches_played: int = 0


def _mean(values: Deque[float], default: float = 0.0) -> float:
    return float(np.mean(list(values))) if values else float(default)


def _match_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def implied_probabilities_from_odds(
    home_win_odds: Optional[float],
    draw_odds: Optional[float],
    away_win_odds: Optional[float],
) -> Dict[str, Optional[float]]:
    if not valid_1x2_odds(home_win_odds, draw_odds, away_win_odds):
        return {
            "imp_home": None,
            "imp_draw": None,
            "imp_away": None,
            "market_overround": None,
        }

    inv_home = 1.0 / home_win_odds
    inv_draw = 1.0 / draw_odds
    inv_away = 1.0 / away_win_odds
    total = inv_home + inv_draw + inv_away

    return {
        "imp_home": inv_home / total,
        "imp_draw": inv_draw / total,
        "imp_away": inv_away / total,
        "market_overround": total,
    }


def valid_1x2_odds(
    home_win_odds: Optional[float],
    draw_odds: Optional[float],
    away_win_odds: Optional[float],
) -> bool:
    values = [home_win_odds, draw_odds, away_win_odds]
    if any(value is None for value in values):
        return False

    try:
        home_value, draw_value, away_value = [float(value) for value in values]
    except Exception:
        return False

    if min(home_value, draw_value, away_value) <= 1.01:
        return False
    if max(home_value, draw_value, away_value) > 1000:
        return False
    return True


def valid_simple_odds(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        parsed = float(value)
    except Exception:
        return False
    return 1.01 < parsed <= 1000


def league_quality_index(league_name: Optional[str]) -> float:
    text = str(league_name or "").strip().lower()
    if not text:
        return 0.82

    top_tier_tokens = [
        "premier league",
        "super lig",
        "serie a",
        "la liga",
        "bundesliga",
        "ligue 1",
        "eredivisie",
        "pro league",
        "primeira liga",
        "premiership",
    ]
    second_tier_tokens = [
        "championship",
        "serie b",
        "segunda",
        "bundesliga 2",
        "ligue 2",
        "2.",
    ]
    lower_tier_tokens = [
        "league one",
        "league two",
        "national league",
        "amator",
        "amatör",
        "u19",
        "u20",
        "u21",
        "reserves",
        "reserve",
        "ii",
        "iii",
    ]

    if any(token in text for token in top_tier_tokens):
        return 1.0
    if any(token in text for token in second_tier_tokens):
        return 0.9
    if any(token in text for token in lower_tier_tokens):
        return 0.72
    return 0.82


def is_trusted_iddaa_training_league(league_name: Optional[str]) -> bool:
    text = str(league_name or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in TRUSTED_LEAGUE_EXCLUDE_TOKENS):
        return False
    return league_quality_index(league_name) >= 0.82


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    k: float = 20.0,
    league_quality: float = 0.82,
):
    actual_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
    actual_away = 1.0 - actual_home

    exp_home = expected_score(home_elo + 50.0, away_elo)
    exp_away = 1.0 - exp_home

    goal_margin = abs(home_score - away_score)
    margin_multiplier = 1.0 + min(goal_margin, 3) * 0.08
    quality_multiplier = 0.9 + (league_quality * 0.2)
    adjusted_k = k * margin_multiplier * quality_multiplier

    new_home = home_elo + adjusted_k * (actual_home - exp_home)
    new_away = away_elo + adjusted_k * (actual_away - exp_away)
    return new_home, new_away


def _base_row_from_state(
    home_state: TeamState,
    away_state: TeamState,
    kickoff_time: datetime,
    league_name: Optional[str] = None,
) -> Dict[str, float]:
    home_rest_days = 7.0
    away_rest_days = 7.0

    if home_state.last_match_date is not None:
        home_rest_days = max(1.0, (kickoff_time - home_state.last_match_date).days)
    if away_state.last_match_date is not None:
        away_rest_days = max(1.0, (kickoff_time - away_state.last_match_date).days)

    return {
        "league_quality_index": league_quality_index(league_name),
        "home_elo": home_state.elo,
        "away_elo": away_state.elo,
        "elo_diff": home_state.elo - away_state.elo,
        "home_points_last5": _mean(home_state.points, default=1.2),
        "away_points_last5": _mean(away_state.points, default=1.2),
        "form_diff": _mean(home_state.points, default=1.2) - _mean(away_state.points, default=1.2),
        "home_goal_diff_last5": _mean(home_state.goal_diff, default=0.0),
        "away_goal_diff_last5": _mean(away_state.goal_diff, default=0.0),
        "goal_diff_form_diff": _mean(home_state.goal_diff, default=0.0) - _mean(away_state.goal_diff, default=0.0),
        "home_goals_for_last5": _mean(home_state.goals_for, default=1.0),
        "away_goals_for_last5": _mean(away_state.goals_for, default=1.0),
        "home_goals_against_last5": _mean(home_state.goals_against, default=1.0),
        "away_goals_against_last5": _mean(away_state.goals_against, default=1.0),
        "home_home_points_last5": _mean(home_state.home_points, default=1.3),
        "away_away_points_last5": _mean(away_state.away_points, default=1.1),
        "venue_form_diff": _mean(home_state.home_points, default=1.3)
        - _mean(away_state.away_points, default=1.1),
        "home_home_goal_diff_last5": _mean(home_state.home_goal_diff, default=0.1),
        "away_away_goal_diff_last5": _mean(away_state.away_goal_diff, default=-0.1),
        "venue_goal_diff_form_diff": _mean(home_state.home_goal_diff, default=0.1)
        - _mean(away_state.away_goal_diff, default=-0.1),
        "home_home_goals_for_last5": _mean(home_state.home_goals_for, default=1.2),
        "away_away_goals_for_last5": _mean(away_state.away_goals_for, default=1.0),
        "home_home_goals_against_last5": _mean(home_state.home_goals_against, default=1.0),
        "away_away_goals_against_last5": _mean(away_state.away_goals_against, default=1.1),
        "home_home_matches_played": float(home_state.home_matches_played),
        "away_away_matches_played": float(away_state.away_matches_played),
        "venue_experience_diff": float(home_state.home_matches_played - away_state.away_matches_played),
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "rest_diff": home_rest_days - away_rest_days,
        "home_matches_played": float(home_state.matches_played),
        "away_matches_played": float(away_state.matches_played),
    }


def _apply_match_result(
    home_state: TeamState,
    away_state: TeamState,
    kickoff_time: datetime,
    home_score: int,
    away_score: int,
    league_name: Optional[str] = None,
):
    home_points = _match_points(home_score, away_score)
    away_points = _match_points(away_score, home_score)

    home_state.points.append(home_points)
    away_state.points.append(away_points)

    home_state.goal_diff.append(home_score - away_score)
    away_state.goal_diff.append(away_score - home_score)

    home_state.goals_for.append(home_score)
    home_state.goals_against.append(away_score)
    away_state.goals_for.append(away_score)
    away_state.goals_against.append(home_score)
    home_state.home_points.append(home_points)
    away_state.away_points.append(away_points)
    home_state.home_goal_diff.append(home_score - away_score)
    away_state.away_goal_diff.append(away_score - home_score)
    home_state.home_goals_for.append(home_score)
    home_state.home_goals_against.append(away_score)
    away_state.away_goals_for.append(away_score)
    away_state.away_goals_against.append(home_score)

    home_state.elo, away_state.elo = update_elo(
        home_state.elo,
        away_state.elo,
        home_score,
        away_score,
        league_quality=league_quality_index(league_name),
    )

    home_state.last_match_date = kickoff_time
    away_state.last_match_date = kickoff_time
    home_state.matches_played += 1
    away_state.matches_played += 1
    home_state.home_matches_played += 1
    away_state.away_matches_played += 1


def _completed_matches():
    session = get_session()
    try:
        return (
            session.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
                joinedload(Match.odds_snapshots),
            )
            .filter(
                Match.status == "completed",
                Match.home_score.isnot(None),
                Match.away_score.isnot(None),
            )
            .order_by(Match.kickoff_time.asc())
            .all()
        )
    finally:
        session.close()


def _scheduled_matches():
    cutoff = datetime.utcnow() - timedelta(minutes=5)
    session = get_session()
    try:
        return (
            session.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
                joinedload(Match.odds_snapshots),
            )
            .filter(
                Match.status == "scheduled",
                Match.kickoff_time >= cutoff,
            )
            .order_by(Match.kickoff_time.asc())
            .all()
        )
    finally:
        session.close()


def _live_matches():
    cutoff = datetime.utcnow() - timedelta(hours=4)
    session = get_session()
    try:
        return (
            session.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
                joinedload(Match.odds_snapshots),
            )
            .filter(
                Match.status == "live",
                Match.kickoff_time >= cutoff,
            )
            .order_by(Match.kickoff_time.asc())
            .all()
        )
    finally:
        session.close()


def build_training_dataframe() -> pd.DataFrame:
    matches = _completed_matches()
    preferred_odds_lookup = build_preferred_odds_lookup(matches)
    league_counts: Dict[str, int] = defaultdict(int)
    for match in matches:
        league_name = match.league.name if match.league else None
        if league_name:
            league_counts[league_name] += 1

    matches = [
        match
        for match in matches
        if match.league
        and match.league.name
        and is_trusted_iddaa_training_league(match.league.name)
        and league_counts.get(match.league.name, 0) >= MIN_TRUSTED_LEAGUE_MATCH_COUNT
    ]

    team_states = defaultdict(TeamState)
    rows: List[Dict] = []

    for match in matches:
        home_state = team_states[match.home_team_id]
        away_state = team_states[match.away_team_id]

        row = {
            "match_id": match.id,
            "kickoff_time": match.kickoff_time,
            "league_name": match.league.name if match.league else None,
            "home_team": match.home_team.name,
            "away_team": match.away_team.name,
        }
        row.update(
            _base_row_from_state(
                home_state,
                away_state,
                match.kickoff_time,
                match.league.name if match.league else None,
            )
        )

        odds = preferred_odds_lookup.get(match.id) or {}
        row["market_home_odds"] = odds.get("home_win_odds")
        row["market_draw_odds"] = odds.get("draw_odds")
        row["market_away_odds"] = odds.get("away_win_odds")
        row["market_over_25_odds"] = odds.get("over_25_odds")
        row["market_under_25_odds"] = odds.get("under_25_odds")
        row["market_btts_yes_odds"] = odds.get("btts_yes_odds")
        row["market_btts_no_odds"] = odds.get("btts_no_odds")
        row["market_double_chance_1x_odds"] = odds.get("double_chance_1x_odds")
        row["market_double_chance_12_odds"] = odds.get("double_chance_12_odds")
        row["market_double_chance_x2_odds"] = odds.get("double_chance_x2_odds")
        row["market_handicap_home_10_odds"] = odds.get("handicap_home_10_odds")
        row["market_handicap_draw_10_odds"] = odds.get("handicap_draw_10_odds")
        row["market_handicap_away_10_odds"] = odds.get("handicap_away_10_odds")
        row["odds_source"] = odds.get("odds_source")

        if valid_1x2_odds(
            row["market_home_odds"],
            row["market_draw_odds"],
            row["market_away_odds"],
        ):
            row.update(
                implied_probabilities_from_odds(
                    row["market_home_odds"],
                    row["market_draw_odds"],
                    row["market_away_odds"],
                )
            )
        else:
            row.update(
                {
                    "imp_home": None,
                    "imp_draw": None,
                    "imp_away": None,
                    "market_overround": None,
                }
            )

        if match.home_score > match.away_score:
            row["target"] = 0
            row["target_label"] = "H"
        elif match.home_score == match.away_score:
            row["target"] = 1
            row["target_label"] = "D"
        else:
            row["target"] = 2
            row["target_label"] = "A"

        total_goals = int(match.home_score) + int(match.away_score)
        row["target_total25_label"] = "O25" if total_goals >= 3 else "U25"
        row["target_btts_label"] = "BYES" if min(int(match.home_score), int(match.away_score)) >= 1 else "BNO"
        adjusted_home = int(match.home_score) + 1
        adjusted_away = int(match.away_score)
        if adjusted_home > adjusted_away:
            row["target_handicap_10_label"] = "H10H"
        elif adjusted_home == adjusted_away:
            row["target_handicap_10_label"] = "H10D"
        else:
            row["target_handicap_10_label"] = "H10A"

        rows.append(row)

        _apply_match_result(
            home_state=home_state,
            away_state=away_state,
            kickoff_time=match.kickoff_time,
            home_score=match.home_score,
            away_score=match.away_score,
            league_name=match.league.name if match.league else None,
        )

    return pd.DataFrame(rows)


def available_model_features(df: pd.DataFrame) -> List[str]:
    feature_columns: List[str] = []
    for feature_name in MODEL_FEATURES:
        if feature_name not in df.columns:
            continue
        if df[feature_name].isna().all():
            continue
        feature_columns.append(feature_name)
    return feature_columns


def _build_prediction_feature_dataframe(matches: List[Match]) -> pd.DataFrame:
    completed = _completed_matches()
    preferred_odds_lookup = build_preferred_odds_lookup(matches)

    team_states = defaultdict(TeamState)

    for match in completed:
        home_state = team_states[match.home_team_id]
        away_state = team_states[match.away_team_id]
        _apply_match_result(
            home_state=home_state,
            away_state=away_state,
            kickoff_time=match.kickoff_time,
            home_score=match.home_score,
            away_score=match.away_score,
            league_name=match.league.name if match.league else None,
        )

    rows: List[Dict] = []
    for match in matches:
        home_state = team_states[match.home_team_id]
        away_state = team_states[match.away_team_id]

        row = {
            "match_id": match.id,
            "kickoff_time": match.kickoff_time,
            "league_name": match.league.name if match.league else None,
            "home_team": match.home_team.name,
            "away_team": match.away_team.name,
            "source": match.source,
            "status": match.status,
        }
        row.update(
            _base_row_from_state(
                home_state,
                away_state,
                match.kickoff_time,
                match.league.name if match.league else None,
            )
        )

        odds = preferred_odds_lookup.get(match.id) or {}
        league_known = bool(match.league and match.league.name and match.league.name != "Unknown League")
        valid_market_odds = bool(
            odds
            and valid_1x2_odds(
                odds.get("home_win_odds"),
                odds.get("draw_odds"),
                odds.get("away_win_odds"),
            )
        )
        if odds:
            row["market_home_odds"] = odds.get("home_win_odds")
            row["market_draw_odds"] = odds.get("draw_odds")
            row["market_away_odds"] = odds.get("away_win_odds")
            row["market_over_25_odds"] = odds.get("over_25_odds")
            row["market_under_25_odds"] = odds.get("under_25_odds")
            row["market_btts_yes_odds"] = odds.get("btts_yes_odds")
            row["market_btts_no_odds"] = odds.get("btts_no_odds")
            row["market_double_chance_1x_odds"] = odds.get("double_chance_1x_odds")
            row["market_double_chance_12_odds"] = odds.get("double_chance_12_odds")
            row["market_double_chance_x2_odds"] = odds.get("double_chance_x2_odds")
            row["market_handicap_home_10_odds"] = odds.get("handicap_home_10_odds")
            row["market_handicap_draw_10_odds"] = odds.get("handicap_draw_10_odds")
            row["market_handicap_away_10_odds"] = odds.get("handicap_away_10_odds")
            row["odds_source"] = odds.get("odds_source")
            row["odds_consistency_ok"] = odds.get("odds_consistency_ok", True)
            row["odds_consistency_gap"] = odds.get("odds_consistency_gap")
            row["odds_consistency_market"] = odds.get("odds_consistency_market")
            row["odds_consistency_reference_source"] = odds.get("odds_consistency_reference_source")
            if valid_1x2_odds(
                odds.get("home_win_odds"),
                odds.get("draw_odds"),
                odds.get("away_win_odds"),
            ):
                row.update(
                    implied_probabilities_from_odds(
                        odds.get("home_win_odds"),
                        odds.get("draw_odds"),
                        odds.get("away_win_odds"),
                    )
                )
            else:
                row.update(
                    {
                        "imp_home": None,
                        "imp_draw": None,
                        "imp_away": None,
                        "market_overround": None,
                    }
                )
        else:
            row["market_home_odds"] = None
            row["market_draw_odds"] = None
            row["market_away_odds"] = None
            row["market_over_25_odds"] = None
            row["market_under_25_odds"] = None
            row["market_btts_yes_odds"] = None
            row["market_btts_no_odds"] = None
            row["market_double_chance_1x_odds"] = None
            row["market_double_chance_12_odds"] = None
            row["market_double_chance_x2_odds"] = None
            row["market_handicap_home_10_odds"] = None
            row["market_handicap_draw_10_odds"] = None
            row["market_handicap_away_10_odds"] = None
            row["odds_source"] = None
            row["odds_consistency_ok"] = True
            row["odds_consistency_gap"] = None
            row["odds_consistency_market"] = None
            row["odds_consistency_reference_source"] = None
            row.update(
                {
                    "imp_home": None,
                    "imp_draw": None,
                    "imp_away": None,
                    "market_overround": None,
                }
            )

        row["league_known"] = league_known
        row["has_valid_1x2_odds"] = valid_market_odds
        has_totals_market = bool(
            valid_simple_odds(row.get("market_over_25_odds"))
            or valid_simple_odds(row.get("market_under_25_odds"))
        )
        has_btts_market = bool(
            valid_simple_odds(row.get("market_btts_yes_odds"))
            or valid_simple_odds(row.get("market_btts_no_odds"))
        )
        has_double_chance_market = bool(
            valid_simple_odds(row.get("market_double_chance_1x_odds"))
            or valid_simple_odds(row.get("market_double_chance_12_odds"))
            or valid_simple_odds(row.get("market_double_chance_x2_odds"))
        )
        has_handicap_market = bool(
            valid_simple_odds(row.get("market_handicap_home_10_odds"))
            or valid_simple_odds(row.get("market_handicap_draw_10_odds"))
            or valid_simple_odds(row.get("market_handicap_away_10_odds"))
        )
        row["has_supported_market_odds"] = bool(
            valid_market_odds
            or has_totals_market
            or has_btts_market
            or has_double_chance_market
            or has_handicap_market
        )
        odds_consistency_ok = bool(row.get("odds_consistency_ok", True))
        row["betting_ready"] = bool(
            league_known and row["has_supported_market_odds"] and odds_consistency_ok
        )
        if not league_known and not row["has_supported_market_odds"]:
            row["betting_ready_reason"] = "Lig bilgisi ve desteklenen market oranlari eksik"
        elif not league_known:
            row["betting_ready_reason"] = "Lig bilgisi eksik"
        elif not row["has_supported_market_odds"]:
            row["betting_ready_reason"] = "Desteklenen market oranlari yok"
        elif not odds_consistency_ok:
            gap_value = row.get("odds_consistency_gap")
            market_name = row.get("odds_consistency_market") or "market"
            row["betting_ready_reason"] = (
                f"Kaynaklar arasi oran sapmasi yuksek ({market_name}: {float(gap_value):.2%})"
                if gap_value is not None
                else "Kaynaklar arasi oran sapmasi yuksek"
            )
        else:
            row["betting_ready_reason"] = "Hazir"

        rows.append(row)

    return pd.DataFrame(rows)


def build_upcoming_feature_dataframe() -> pd.DataFrame:
    return _build_prediction_feature_dataframe(_scheduled_matches())


def build_live_feature_dataframe() -> pd.DataFrame:
    return _build_prediction_feature_dataframe(_live_matches())
