import io
import math
import random
import re
import time
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from football_analytics.config import IDDAA_VERIFY_SSL, LOCAL_CSV_IMPORT_DIR
from football_analytics.db import (
    Match,
    add_odds_snapshot,
    cleanup_invalid_active_1x2_odds_snapshots,
    create_tables,
    get_session,
    upsert_match,
)

NESINE_PREMATCH_URL = "https://bulten.nesine.com/api/bulten/getprebultenfull"
NESINE_LIVE_URL = "https://bulten.nesine.com/api/bulten/getlivebultenv3"
IDDAA_EVENTS_URL = "https://sportsbookv2.iddaa.com/sportsbook/events?st=1&type=0&version=0"
IDDAA_COMPETITIONS_URL = "https://sportsbookv2.iddaa.com/sportsbook/competitions"
IDDAA_MARKET_CONFIG_URL = "https://sportsbookv2.iddaa.com/sportsbook/get_market_config"
IDDAA_HIGHLIGHTED_EVENTS_URL = "https://sportsbookv2.iddaa.com/sportsbook/highlighted-events?st=1"
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
FOOTBALLCSV_CACHE_BASE_URL = "https://raw.githubusercontent.com/footballcsv/cache.footballdata/master"
HISTORICAL_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "historical_cache"
HISTORICAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

FOOTBALL_DATA_LEAGUES: Dict[str, Dict[str, str]] = {
    "T1": {"league_name": "Turkey Super Lig", "country": "TR"},
    "E0": {"league_name": "England Premier League", "country": "GB"},
    "E1": {"league_name": "England Championship", "country": "GB"},
    "E2": {"league_name": "England League One", "country": "GB"},
    "E3": {"league_name": "England League Two", "country": "GB"},
    "EC": {"league_name": "England National League", "country": "GB"},
    "SC0": {"league_name": "Scotland Premiership", "country": "GB"},
    "D1": {"league_name": "Germany Bundesliga", "country": "DE"},
    "D2": {"league_name": "Germany Bundesliga 2", "country": "DE"},
    "I1": {"league_name": "Italy Serie A", "country": "IT"},
    "I2": {"league_name": "Italy Serie B", "country": "IT"},
    "SP1": {"league_name": "Spain La Liga", "country": "ES"},
    "SP2": {"league_name": "Spain Segunda", "country": "ES"},
    "F1": {"league_name": "France Ligue 1", "country": "FR"},
    "F2": {"league_name": "France Ligue 2", "country": "FR"},
    "N1": {"league_name": "Netherlands Eredivisie", "country": "NL"},
    "B1": {"league_name": "Belgium Pro League", "country": "BE"},
    "P1": {"league_name": "Portugal Primeira Liga", "country": "PT"},
    "G1": {"league_name": "Greece Super League", "country": "GR"},
    "AT1": {"league_name": "Austria Bundesliga", "country": "AT"},
    "KZ1": {"league_name": "Kazakhstan Premier League", "country": "KZ"},
    "IL1": {"league_name": "Israel Premier League", "country": "IL"},
    "SA1": {"league_name": "Saudi Pro League", "country": "SA"},
    "JP1": {"league_name": "Japan J1 League", "country": "JP"},
    "CN1": {"league_name": "China Super League", "country": "CN"},
    "BR1": {"league_name": "Brazil Serie A", "country": "BR"},
    "MX1": {"league_name": "Mexico Liga MX", "country": "MX"},
    "MLS1": {"league_name": "USA Major League Soccer", "country": "US"},
    "AU1": {"league_name": "Australia A-League", "country": "AU"},
    "EE1": {"league_name": "Estonia Premium Liiga", "country": "EE"},
    "LV1": {"league_name": "Latvia Virsliga", "country": "LV"},
    "HN1": {"league_name": "Honduras Liga Nacional", "country": "HN"},
    "NL2": {"league_name": "Netherlands Eerste Divisie", "country": "NL"},
    "IE1": {"league_name": "Ireland Premier Division", "country": "IE"},
    "IS1": {"league_name": "Iceland Besta Deild", "country": "IS"},
    "HU1": {"league_name": "Hungary NB I", "country": "HU"},
    "DK1": {"league_name": "Denmark Superliga", "country": "DK"},
    "DK2": {"league_name": "Denmark 1st Division", "country": "DK"},
    "FI1": {"league_name": "Finland Veikkausliiga", "country": "FI"},
    "FO1": {"league_name": "Faroe Islands Premier League", "country": "FO"},
    "GE1": {"league_name": "Georgia Erovnuli Liga", "country": "GE"},
    "ME1": {"league_name": "Montenegro First League", "country": "ME"},
    "MT1": {"league_name": "Malta Premier League", "country": "MT"},
    "LU1": {"league_name": "Luxembourg National Division", "country": "LU"},
    "KOS1": {"league_name": "Kosovo Superleague", "country": "XK"},
    "DZ1": {"league_name": "Algeria Ligue 1", "country": "DZ"},
    "LT1": {"league_name": "Lithuania A Lyga", "country": "LT"},
}

FOOTBALLCSV_MIRROR_CODES: Dict[str, str] = {
    "T1": "tr.1",
    "E0": "eng.1",
    "E1": "eng.2",
    "E2": "eng.3",
    "E3": "eng.4",
    "EC": "eng.5",
    "SC0": "sco.1",
    "D1": "de.1",
    "D2": "de.2",
    "I1": "it.1",
    "I2": "it.2",
    "SP1": "es.1",
    "SP2": "es.2",
    "F1": "fr.1",
    "F2": "fr.2",
    "N1": "nl.1",
    "B1": "be.1",
    "P1": "pt.1",
    "G1": "gr.1",
    "AT1": "at.1",
    "KZ1": "kz1",
    "IL1": "il1",
    "SA1": "sa1",
    "JP1": "jp1",
    "CN1": "cn1",
    "BR1": "br.1",
    "MX1": "mx1",
    "MLS1": "mls",
    "AU1": "au1",
    "EE1": "ee1",
    "LV1": "lv1",
    "HN1": "hn1",
    "NL2": "nl.2",
    "IE1": "ie1",
    "IS1": "is1",
    "HU1": "hu1",
    "DK1": "dk1",
    "DK2": "dk2",
    "FI1": "fi1",
    "FO1": "fo1",
    "GE1": "ge1",
    "ME1": "me1",
    "MT1": "mt1",
    "LU1": "lu1",
    "KOS1": "kos1",
    "DZ1": "dz1",
    "LT1": "lt1",
}

ODDS_HOME_COLUMNS = ["AvgCH", "B365CH", "PSCH", "AvgH", "B365H", "PSH"]
ODDS_DRAW_COLUMNS = ["AvgCD", "B365CD", "PSCD", "AvgD", "B365D", "PSD"]
ODDS_AWAY_COLUMNS = ["AvgCA", "B365CA", "PSCA", "AvgA", "B365A", "PSA"]


def current_season_start_year(today: Optional[datetime] = None) -> int:
    today = today or datetime.utcnow()
    return today.year if today.month >= 7 else today.year - 1


def default_historical_season_starts(num_seasons: int = 3) -> List[int]:
    current_start = current_season_start_year()
    return [current_start - offset for offset in reversed(range(num_seasons))]


def default_training_league_codes(profile: str = "broad") -> List[str]:
    if profile == "core":
        return ["T1", "E0", "D1", "I1", "SP1", "F1"]
    return [
        "T1",
        "E0",
        "E1",
        "E2",
        "E3",
        "EC",
        "SC0",
        "D1",
        "D2",
        "I1",
        "I2",
        "SP1",
        "SP2",
        "F1",
        "F2",
        "N1",
        "B1",
        "P1",
        "G1",
        "AT1",
        "KZ1",
        "IL1",
        "SA1",
        "JP1",
        "CN1",
        "BR1",
        "MX1",
        "MLS1",
        "AU1",
    ]


def default_bulk_import_bundle() -> List[tuple[str, int]]:
    return [
        ("E0", 2025),
        ("E1", 2025),
        ("SP1", 2025),
        ("I1", 2025),
        ("D1", 2025),
        ("F1", 2025),
        ("T1", 2025),
        ("N1", 2025),
        ("E0", 2024),
        ("E1", 2024),
        ("SP1", 2024),
        ("I1", 2024),
        ("D1", 2024),
        ("F1", 2024),
        ("T1", 2024),
    ]


def build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        read=4,
        connect=4,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "football-analytics/1.0 (+research-use)",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


def build_iddaa_http_session() -> requests.Session:
    session = build_http_session()
    session.headers.update(
        {
            "Origin": "https://www.iddaa.com",
            "Referer": "https://www.iddaa.com/program/futbol",
        }
    )
    return session


def build_football_data_session() -> requests.Session:
    return build_http_session()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", "."))
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _normalize_local_csv_code_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().upper())
    return cleaned.strip("_")


LOCAL_CSV_LEAGUE_CODE_ALIASES: Dict[str, str] = {
    _normalize_local_csv_code_token(mirror_code): league_code
    for league_code, mirror_code in FOOTBALLCSV_MIRROR_CODES.items()
}
LOCAL_CSV_LEAGUE_CODE_ALIASES.update(
    {
        "CHAMPS": "UCL",
        "1_MLS": "MLS1",
        "MLS": "MLS1",
    }
)


def _extract_season_start_from_text(value: str) -> Optional[int]:
    if not value:
        return None

    range_match = re.search(r"(19|20)\d{2}\s*[-_]\s*\d{2,4}", value)
    if range_match:
        return _safe_int(range_match.group(0)[:4])

    year_match = re.search(r"(19|20)\d{2}", value)
    if year_match:
        return _safe_int(year_match.group(0))

    return None


def _infer_local_csv_metadata(csv_file: Path, root_folder: Path) -> tuple[str, int]:
    stem = csv_file.stem
    stem_parts = stem.split("_")
    inferred_league_code = None
    inferred_season_start = None

    if len(stem_parts) >= 2:
        first_part_season = _extract_season_start_from_text(stem_parts[0])
        second_part_season = _extract_season_start_from_text(stem_parts[1])
        if first_part_season is not None and second_part_season is None:
            inferred_season_start = first_part_season
            inferred_league_code = _normalize_local_csv_code_token(stem_parts[1])
        elif second_part_season is not None and first_part_season is None:
            inferred_season_start = second_part_season
            inferred_league_code = _normalize_local_csv_code_token(stem_parts[0])

    try:
        relative_parts = csv_file.relative_to(root_folder).parts
    except Exception:
        relative_parts = csv_file.parts

    if inferred_season_start is None:
        for part in reversed(relative_parts[:-1]):
            inferred_season_start = _extract_season_start_from_text(part)
            if inferred_season_start is not None:
                break

    normalized_stem = _normalize_local_csv_code_token(stem)
    league_code = LOCAL_CSV_LEAGUE_CODE_ALIASES.get(normalized_stem)
    if not league_code and inferred_league_code:
        league_code = LOCAL_CSV_LEAGUE_CODE_ALIASES.get(inferred_league_code, inferred_league_code)
    if not league_code:
        league_code = normalized_stem or "UNKNOWN"

    season_start = inferred_season_start or current_season_start_year()
    return league_code, season_start


def _first_non_null(row: pd.Series, candidates: Iterable[str]) -> Optional[float]:
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            value = _safe_float(row[column])
            if value is not None:
                return value
    return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, dayfirst=True)
        if pd.isna(ts):
            return None
        return ts.tz_convert(None).to_pydatetime()
    except Exception:
        return None


def _parse_unix_seconds(value: Any) -> Optional[datetime]:
    ivalue = _safe_int(value)
    if ivalue is None:
        return None
    try:
        ts = pd.to_datetime(ivalue, unit="s", utc=True)
        if pd.isna(ts):
            return None
        return ts.tz_convert(None).to_pydatetime()
    except Exception:
        return None


def _parse_unix_millis(value: Any) -> Optional[datetime]:
    ivalue = _safe_int(value)
    if ivalue is None:
        return None
    try:
        ts = pd.to_datetime(ivalue, unit="ms", utc=True)
        if pd.isna(ts):
            return None
        return ts.tz_convert(None).to_pydatetime()
    except Exception:
        return None


def _first_value(d: Dict[str, Any], candidates: Iterable[str]) -> Any:
    lower_map = {str(k).lower(): v for k, v in d.items()}
    for key in candidates:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None


def _extract_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ["name", "teamName", "title", "value"]:
            if key in value and value[key]:
                return str(value[key]).strip()
        for item in value.values():
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _split_match_name(value: Any) -> tuple[Optional[str], Optional[str]]:
    text = _extract_name(value)
    if not text:
        return None, None

    for sep in [" - ", " – ", " v ", " vs ", " / "]:
        if sep in text:
            left, right = text.split(sep, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
    return None, None


def _looks_like_bundle_text(value: Optional[str]) -> bool:
    if not value:
        return False
    text = str(value).strip().lower()
    blocked_fragments = [
        "haftalık bahisler",
        "çeyrek final",
        "ceyrek final",
        "yarı final",
        "yari final",
        "rövanş maçları",
        "rovans maclari",
        "maçları",
        "maclari",
        "uefa avrupa ligi",
        "uefa konferans ligi",
        "konf. ligi",
        "avr. ligi",
        "bahisler",
    ]
    return any(fragment in text for fragment in blocked_fragments)


def _looks_like_real_team_name(value: Optional[str]) -> bool:
    if not value:
        return False
    text = str(value).strip()
    if len(text) < 2:
        return False
    if _looks_like_bundle_text(text):
        return False
    return True


def _looks_like_pure_match_result_market(market_name: str) -> bool:
    text = str(market_name or "").strip().lower()
    if not text:
        return False

    include_tokens = [
        "1x2",
        "maç sonucu",
        "mac sonucu",
        "match result",
        "full time result",
    ]
    exclude_tokens = [
        "handikap",
        "hnd",
        "kg",
        "karsilikli gol",
        "karşılıklı gol",
        "toplam gol",
        "alt/ust",
        "alt üst",
        "korner",
        "ilk yarı",
        "ilk yari",
        "1. yarı",
        "1. yari",
        "ms &",
        "ms+",
        "1-x",
        "1-2",
        "x-2",
        "çifte şans",
        "cifte sans",
        "maç skoru",
        "mac skoru",
    ]

    if any(token in text for token in exclude_tokens):
        return False
    return any(token in text for token in include_tokens)


def _extract_event_teams(node: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    home = _extract_name(
        _first_value(
            node,
            [
                "homeTeam",
                "home_team",
                "homeTeamName",
                "home",
                "evsahibi",
                "team1",
                "participant1",
            ],
        )
    )
    away = _extract_name(
        _first_value(
            node,
            [
                "awayTeam",
                "away_team",
                "awayTeamName",
                "away",
                "deplasman",
                "team2",
                "participant2",
            ],
        )
    )
    if home and away:
        return home, away

    for key in ["matchName", "eventName", "name", "title", "label", "ENO"]:
        parsed_home, parsed_away = _split_match_name(_first_value(node, [key]))
        if parsed_home and parsed_away:
            return parsed_home, parsed_away

    return home, away


def _extract_event_kickoff(node: Dict[str, Any]) -> Optional[datetime]:
    datetime_keys = [
        "matchDate",
        "match_date",
        "kickoff",
        "kickoffTime",
        "eventDate",
        "eventDateTime",
        "date",
        "dateUtc",
        "startDate",
        "start_time",
        "startTime",
        "start",
        "matchTime",
        "D",
    ]
    for key in datetime_keys:
        raw_value = _first_value(node, [key])
        parsed = _parse_datetime(raw_value)
        if parsed is not None:
            return parsed

    for key in ["ESD", "MSD", "MED"]:
        raw_value = _first_value(node, [key])
        parsed = _parse_unix_millis(raw_value)
        if parsed is not None:
            return parsed

    for key in ["ESD", "MSD", "MED"]:
        raw_value = _first_value(node, [key])
        parsed = _parse_unix_seconds(raw_value)
        if parsed is not None:
            return parsed

    return None


def _extract_event_league(node: Dict[str, Any]) -> Optional[str]:
    return _extract_name(
        _first_value(
            node,
            [
                "leagueName",
                "league",
                "league_name",
                "tournamentName",
                "tournament",
                "organizationName",
                "categoryName",
                "sportCategoryName",
                "competitionName",
            ],
        )
    )


def _summarize_nesine_candidate_node(node: Dict[str, Any]) -> Dict[str, Any]:
    home_team, away_team = _extract_event_teams(node)
    kickoff_time = _extract_event_kickoff(node)
    league_name = _extract_event_league(node)
    odds = _extract_1x2_odds_robust(node)
    key_preview = sorted(str(key) for key in node.keys())[:20]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_time": kickoff_time.isoformat() if kickoff_time else None,
        "league_name": league_name,
        "has_any_1x2_odds": any(value is not None for value in odds.values()),
        "odds": odds,
        "keys_preview": key_preview,
    }


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _looks_like_event(d: Dict[str, Any]) -> bool:
    home, away = _extract_event_teams(d)
    kickoff = _extract_event_kickoff(d)
    return bool(
        kickoff
        and _looks_like_real_team_name(home)
        and _looks_like_real_team_name(away)
    )


def build_football_data_url(league_code: str, season_start: int) -> str:
    season_code = f"{str(season_start)[-2:]}{str(season_start + 1)[-2:]}"
    return f"{FOOTBALL_DATA_BASE_URL}/{season_code}/{league_code}.csv"


def _historical_cache_path(league_code: str, season_start: int) -> Path:
    return HISTORICAL_CACHE_DIR / f"{league_code}_{season_start}.csv"


def build_footballcsv_cache_url(league_code: str, season_start: int) -> Optional[str]:
    mirror_code = FOOTBALLCSV_MIRROR_CODES.get(league_code)
    if not mirror_code:
        return None
    season_code = f"{season_start}-{str(season_start + 1)[-2:]}"
    return f"{FOOTBALLCSV_CACHE_BASE_URL}/{season_code}/{mirror_code}.csv"


def _footballcsv_cache_path(league_code: str, season_start: int) -> Path:
    return HISTORICAL_CACHE_DIR / f"footballcsv_{league_code}_{season_start}.csv"


def _parse_football_data_kickoff(row: pd.Series) -> Optional[datetime]:
    date_value = row.get("Date")
    if pd.isna(date_value):
        return None

    time_value = row.get("Time") if "Time" in row.index else None
    if pd.notna(time_value):
        combined = f"{date_value} {time_value}"
        parsed = pd.to_datetime(combined, dayfirst=True, errors="coerce")
    else:
        parsed = pd.to_datetime(date_value, dayfirst=True, errors="coerce")

    if pd.isna(parsed):
        return None

    parsed = parsed.to_pydatetime()
    if time_value is None or pd.isna(time_value):
        return parsed.replace(hour=12, minute=0, second=0, microsecond=0)
    return parsed


def _normalize_football_data_match(
    row: pd.Series,
    league_code: str,
    season_start: int,
) -> Optional[Dict[str, Any]]:
    if any(column not in row.index for column in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]):
        return None

    kickoff_time = _parse_football_data_kickoff(row)
    home_team = _extract_name(row.get("HomeTeam"))
    away_team = _extract_name(row.get("AwayTeam"))
    home_score = _safe_int(row.get("FTHG"))
    away_score = _safe_int(row.get("FTAG"))

    if kickoff_time is None or not home_team or not away_team:
        return None
    if home_score is None or away_score is None:
        return None

    league_meta = FOOTBALL_DATA_LEAGUES.get(
        league_code,
        {"league_name": league_code, "country": None},
    )
    date_slug = kickoff_time.strftime("%Y%m%d")

    return {
        "external_id": f"football_data:{league_code}:{season_start}:{date_slug}:{home_team}:{away_team}",
        "source": "football_data",
        "league_name": league_meta["league_name"],
        "country": league_meta["country"],
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_time": kickoff_time,
        "status": "completed",
        "is_live": False,
        "home_score": home_score,
        "away_score": away_score,
        "notes": f"league_code={league_code} | season_start={season_start}",
        "odds": {
            "home_win_odds": _first_non_null(row, ODDS_HOME_COLUMNS),
            "draw_odds": _first_non_null(row, ODDS_DRAW_COLUMNS),
            "away_win_odds": _first_non_null(row, ODDS_AWAY_COLUMNS),
            "over_25_odds": None,
            "under_25_odds": None,
        },
    }


def _parse_footballcsv_score(value: Any) -> tuple[Optional[int], Optional[int]]:
    if value is None:
        return None, None
    text = str(value).strip()
    for separator in ["-", "–", ":"]:
        if separator in text:
            left, right = text.split(separator, 1)
            return _safe_int(left.strip()), _safe_int(right.strip())
    return None, None


def _is_supported_local_import_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".txt"}:
        return False

    blocked_name_tokens = {"readme", "license", "summary"}
    if path.stem.strip().lower() in blocked_name_tokens:
        return False

    blocked_path_tokens = {
        "cache.footballsquads-master",
        "cache.wfb-master",
    }
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & blocked_path_tokens:
        return False

    return True


def _parse_openfootball_date(line: str, season_start: int, previous_date: Optional[datetime]) -> Optional[datetime]:
    text = re.sub(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+", "", line.strip())
    if not text:
        return None

    parsed = pd.to_datetime(text, format="%b/%d %Y", errors="coerce")
    if pd.notna(parsed):
        return parsed.to_pydatetime().replace(hour=12, minute=0, second=0, microsecond=0)

    guess_year = previous_date.year if previous_date is not None else season_start
    parsed = pd.to_datetime(f"{text} {guess_year}", format="%b/%d %Y", errors="coerce")
    if pd.isna(parsed):
        return None

    candidate = parsed.to_pydatetime().replace(hour=12, minute=0, second=0, microsecond=0)
    if previous_date is not None and candidate < previous_date - timedelta(days=45):
        candidate = candidate.replace(year=candidate.year + 1)
    elif previous_date is None and candidate.month < 7 and season_start >= candidate.year:
        candidate = candidate.replace(year=season_start + 1)
    return candidate


def _parse_openfootball_text_matches(
    text: str,
    league_code: str,
    season_start: int,
) -> List[Dict[str, Any]]:
    league_meta = FOOTBALL_DATA_LEAGUES.get(
        league_code,
        {"league_name": league_code, "country": None},
    )
    rows: List[Dict[str, Any]] = []
    current_date: Optional[datetime] = None

    match_pattern = re.compile(
        r"^\s*(?:(\d{1,2}\.\d{2})\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)\s+\((\d+)-(\d+)\)\s*$"
    )

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("=", "#", "»")):
            continue

        date_candidate = _parse_openfootball_date(stripped, season_start=season_start, previous_date=current_date)
        if date_candidate is not None:
            current_date = date_candidate
            continue

        match = match_pattern.match(line)
        if not match or current_date is None:
            continue

        kickoff_text, home_team, away_team, home_score, away_score, _, _ = match.groups()
        kickoff_time = current_date
        if kickoff_text:
            try:
                hour, minute = kickoff_text.split(".")
                kickoff_time = current_date.replace(
                    hour=int(hour), minute=int(minute), second=0, microsecond=0
                )
            except Exception:
                kickoff_time = current_date

        date_slug = kickoff_time.strftime("%Y%m%d")
        home_team = home_team.strip()
        away_team = away_team.strip()
        rows.append(
            {
                "external_id": f"footballcsvtxt:{league_code}:{season_start}:{date_slug}:{home_team}:{away_team}",
                "source": "footballcsv",
                "league_name": league_meta["league_name"],
                "country": league_meta["country"],
                "home_team": home_team,
                "away_team": away_team,
                "kickoff_time": kickoff_time,
                "status": "completed",
                "is_live": False,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "notes": f"openfootball txt | league_code={league_code} | season_start={season_start}",
                "odds": {
                    "home_win_odds": None,
                    "draw_odds": None,
                    "away_win_odds": None,
                    "over_25_odds": None,
                    "under_25_odds": None,
                },
            }
        )

    return rows


def _normalize_footballcsv_match(
    row: pd.Series,
    league_code: str,
    season_start: int,
) -> Optional[Dict[str, Any]]:
    home_team = _extract_name(
        row.get("Team 1")
        or row.get("Home")
        or row.get("HomeTeam")
        or row.get("home_team")
    )
    away_team = _extract_name(
        row.get("Team 2")
        or row.get("Away")
        or row.get("AwayTeam")
        or row.get("away_team")
    )
    kickoff_time = _parse_datetime(
        row.get("Date")
        or row.get("date")
        or row.get("MatchDate")
    )
    if kickoff_time is None and pd.notna(row.get("Date")):
        parsed = pd.to_datetime(row.get("Date"), dayfirst=True, errors="coerce")
        if pd.notna(parsed):
            kickoff_time = parsed.to_pydatetime().replace(hour=12, minute=0, second=0, microsecond=0)

    home_score, away_score = _parse_footballcsv_score(
        row.get("FT") or row.get("Score") or row.get("score")
    )

    if not home_team or not away_team or kickoff_time is None:
        return None
    if home_score is None or away_score is None:
        return None

    league_meta = FOOTBALL_DATA_LEAGUES.get(
        league_code,
        {"league_name": league_code, "country": None},
    )
    date_slug = kickoff_time.strftime("%Y%m%d")

    return {
        "external_id": f"footballcsv:{league_code}:{season_start}:{date_slug}:{home_team}:{away_team}",
        "source": "footballcsv",
        "league_name": league_meta["league_name"],
        "country": league_meta["country"],
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_time": kickoff_time,
        "status": "completed",
        "is_live": False,
        "home_score": home_score,
        "away_score": away_score,
        "notes": f"footballcsv mirror | league_code={league_code} | season_start={season_start}",
        "odds": {
            "home_win_odds": None,
            "draw_odds": None,
            "away_win_odds": None,
            "over_25_odds": None,
            "under_25_odds": None,
        },
    }


def fetch_historical_results_from_football_data(
    league_codes: Optional[List[str]] = None,
    season_starts: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    league_codes = league_codes or list(FOOTBALL_DATA_LEAGUES.keys())
    season_starts = season_starts or default_historical_season_starts(num_seasons=3)
    session = build_football_data_session()

    try:
        rows: List[Dict[str, Any]] = []
        for league_code in league_codes:
            for season_start in season_starts:
                url = build_football_data_url(league_code=league_code, season_start=season_start)
                cache_path = _historical_cache_path(league_code=league_code, season_start=season_start)
                csv_text: Optional[str] = None

                try:
                    response = session.get(url, timeout=30, verify=IDDAA_VERIFY_SSL)
                    if response.status_code >= 400:
                        if cache_path.exists():
                            csv_text = cache_path.read_text(encoding="utf-8-sig", errors="ignore")
                        else:
                            continue
                    else:
                        csv_text = response.content.decode("utf-8-sig", errors="ignore")
                        cache_path.write_text(csv_text, encoding="utf-8")
                except requests.RequestException:
                    if cache_path.exists():
                        csv_text = cache_path.read_text(encoding="utf-8-sig", errors="ignore")
                    else:
                        continue

                if not csv_text:
                    continue

                dataframe = pd.read_csv(io.StringIO(csv_text))
                if dataframe.empty:
                    continue

                for _, row in dataframe.iterrows():
                    normalized = _normalize_football_data_match(
                        row=row,
                        league_code=league_code,
                        season_start=season_start,
                    )
                    if normalized is not None:
                        rows.append(normalized)

        unique_events: List[Dict[str, Any]] = []
        seen = set()
        for item in rows:
            if item["external_id"] in seen:
                continue
            seen.add(item["external_id"])
            unique_events.append(item)

        return unique_events
    finally:
        session.close()


def fetch_historical_results_from_footballcsv_cache(
    league_codes: Optional[List[str]] = None,
    season_starts: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    league_codes = league_codes or list(FOOTBALLCSV_MIRROR_CODES.keys())
    season_starts = season_starts or default_historical_season_starts(num_seasons=3)
    session = build_football_data_session()

    try:
        rows: List[Dict[str, Any]] = []
        for league_code in league_codes:
            url = build_footballcsv_cache_url(league_code=league_code, season_start=season_starts[0])
            if FOOTBALLCSV_MIRROR_CODES.get(league_code) is None:
                continue

            for season_start in season_starts:
                url = build_footballcsv_cache_url(league_code=league_code, season_start=season_start)
                if not url:
                    continue
                cache_path = _footballcsv_cache_path(league_code=league_code, season_start=season_start)
                csv_text: Optional[str] = None

                try:
                    response = session.get(url, timeout=30, verify=IDDAA_VERIFY_SSL)
                    if response.status_code >= 400:
                        if cache_path.exists():
                            csv_text = cache_path.read_text(encoding="utf-8-sig", errors="ignore")
                        else:
                            continue
                    else:
                        csv_text = response.content.decode("utf-8-sig", errors="ignore")
                        cache_path.write_text(csv_text, encoding="utf-8")
                except requests.RequestException:
                    if cache_path.exists():
                        csv_text = cache_path.read_text(encoding="utf-8-sig", errors="ignore")
                    else:
                        continue

                if not csv_text:
                    continue

                dataframe = pd.read_csv(io.StringIO(csv_text))
                if dataframe.empty:
                    continue

                for _, row in dataframe.iterrows():
                    normalized = _normalize_footballcsv_match(
                        row=row,
                        league_code=league_code,
                        season_start=season_start,
                    )
                    if normalized is not None:
                        rows.append(normalized)

        unique_events: List[Dict[str, Any]] = []
        seen = set()
        for item in rows:
            if item["external_id"] in seen:
                continue
            seen.add(item["external_id"])
            unique_events.append(item)

        return unique_events
    finally:
        session.close()


def persist_historical_results_from_football_data(
    league_codes: Optional[List[str]] = None,
    season_starts: Optional[List[int]] = None,
) -> Dict[str, Any]:
    create_tables()
    session = get_session()
    created = 0

    try:
        matches = fetch_historical_results_from_football_data(
            league_codes=league_codes,
            season_starts=season_starts,
        )
        for match_data in matches:
            match = upsert_match(session, match_data)
            odds = match_data.get("odds") or {}
            if any(value is not None for value in odds.values()):
                add_odds_snapshot(
                    session=session,
                    match=match,
                    bookmaker="football_data",
                    market="1X2",
                    home_win_odds=odds.get("home_win_odds"),
                    draw_odds=odds.get("draw_odds"),
                    away_win_odds=odds.get("away_win_odds"),
                )
            created += 1

        session.commit()
        return {
            "ok": True,
            "inserted_or_updated": created,
            "league_codes": league_codes or list(FOOTBALL_DATA_LEAGUES.keys()),
            "season_starts": season_starts or default_historical_season_starts(num_seasons=3),
        }
    except Exception as exc:
        session.rollback()
        return {"ok": False, "error": str(exc)}
    finally:
        session.close()


def persist_historical_results_from_footballcsv_cache(
    league_codes: Optional[List[str]] = None,
    season_starts: Optional[List[int]] = None,
) -> Dict[str, Any]:
    create_tables()
    session = get_session()
    created = 0

    try:
        matches = fetch_historical_results_from_footballcsv_cache(
            league_codes=league_codes,
            season_starts=season_starts,
        )
        for match_data in matches:
            upsert_match(session, match_data)
            created += 1

        session.commit()
        return {"ok": True, "inserted_or_updated": created, "source": "footballcsv"}
    except Exception as exc:
        session.rollback()
        return {"ok": False, "error": str(exc), "source": "footballcsv"}
    finally:
        session.close()


def import_historical_results_from_local_csv_folder(folder_path: str) -> Dict[str, Any]:
    create_tables()
    session = get_session()
    created = 0
    folder = Path(folder_path)

    try:
        if not folder.exists() or not folder.is_dir():
            return {"ok": False, "error": "Klasor bulunamadi.", "source": "local_csv"}

        import_files = sorted(
            path for path in folder.rglob("*") if path.is_file() and _is_supported_local_import_file(path)
        )
        if not import_files:
            return {"ok": False, "error": "Klasorde desteklenen mac sonucu dosyasi bulunamadi.", "source": "local_csv"}

        imported_files = 0
        skipped_files = 0
        samples = []
        for import_file in import_files:
            parsed_rows = 0
            league_code, season_start = _infer_local_csv_metadata(import_file, folder)
            normalized_rows: List[Dict[str, Any]] = []

            if import_file.suffix.lower() == ".csv":
                try:
                    dataframe = pd.read_csv(import_file, low_memory=False)
                except Exception:
                    skipped_files += 1
                    continue

                for _, row in dataframe.iterrows():
                    normalized = _normalize_football_data_match(
                        row=row,
                        league_code=league_code,
                        season_start=season_start,
                    )
                    if normalized is None:
                        normalized = _normalize_footballcsv_match(
                            row=row,
                            league_code=league_code,
                            season_start=season_start,
                        )
                    if normalized is not None:
                        normalized_rows.append(normalized)
            else:
                try:
                    text = import_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    skipped_files += 1
                    continue
                normalized_rows = _parse_openfootball_text_matches(
                    text=text,
                    league_code=league_code,
                    season_start=season_start,
                )

            for normalized in normalized_rows:
                match = upsert_match(session, normalized)
                odds = normalized.get("odds") or {}
                if any(value is not None for value in odds.values()):
                    add_odds_snapshot(
                        session=session,
                        match=match,
                        bookmaker="local_csv",
                        market="1X2",
                        home_win_odds=odds.get("home_win_odds"),
                        draw_odds=odds.get("draw_odds"),
                        away_win_odds=odds.get("away_win_odds"),
                    )
                created += 1
                parsed_rows += 1

            if parsed_rows > 0:
                imported_files += 1
                if len(samples) < 15:
                    samples.append(
                        {
                            "file": str(import_file.relative_to(folder)),
                            "league_code": league_code,
                            "season_start": season_start,
                            "rows_imported": parsed_rows,
                        }
                    )
            else:
                skipped_files += 1

        session.commit()
        return {
            "ok": True,
            "inserted_or_updated": created,
            "source": "local_csv",
            "import_file_count": len(import_files),
            "imported_file_count": imported_files,
            "skipped_file_count": skipped_files,
            "scanned_root": str(folder),
            "recursive": True,
            "sample_imports": samples,
        }
    except Exception as exc:
        session.rollback()
        return {"ok": False, "error": str(exc), "source": "local_csv"}
    finally:
        session.close()


def download_bulk_import_csv_bundle(
    bundle: Optional[List[tuple[str, int]]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    bundle = bundle or default_bulk_import_bundle()
    session = build_football_data_session()

    downloaded = []
    skipped = []
    failed = []

    try:
        for league_code, season_start in bundle:
            target_path = LOCAL_CSV_IMPORT_DIR / f"{league_code}_{season_start}.csv"
            if target_path.exists() and not overwrite:
                skipped.append(str(target_path.name))
                continue

            csv_text: Optional[str] = None
            source_used: Optional[str] = None

            football_data_url = build_football_data_url(league_code=league_code, season_start=season_start)
            footballcsv_url = build_footballcsv_cache_url(league_code=league_code, season_start=season_start)

            for source_name, url in [("football_data", football_data_url), ("footballcsv", footballcsv_url)]:
                if not url:
                    continue
                try:
                    response = session.get(url, timeout=30, verify=IDDAA_VERIFY_SSL)
                    if response.status_code >= 400:
                        continue
                    csv_text = response.content.decode("utf-8-sig", errors="ignore")
                    if not csv_text.strip():
                        continue
                    source_used = source_name
                    break
                except requests.RequestException:
                    continue

            if csv_text and source_used:
                target_path.write_text(csv_text, encoding="utf-8")
                downloaded.append(
                    {
                        "file": target_path.name,
                        "league_code": league_code,
                        "season_start": season_start,
                        "source": source_used,
                    }
                )
            else:
                failed.append(
                    {
                        "file": target_path.name,
                        "league_code": league_code,
                        "season_start": season_start,
                    }
                )

        return {
            "ok": len(downloaded) > 0,
            "downloaded_count": len(downloaded),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
            "downloaded_files": downloaded,
            "skipped_files": skipped,
            "failed_files": failed,
            "target_folder": str(LOCAL_CSV_IMPORT_DIR),
        }
    finally:
        session.close()


def fetch_iddaa_competitions(
    session: Optional[requests.Session] = None,
) -> Dict[int, Dict[str, Optional[str]]]:
    owns_session = session is None
    session = session or build_iddaa_http_session()

    try:
        response = session.get(
            IDDAA_COMPETITIONS_URL,
            timeout=30,
            verify=IDDAA_VERIFY_SSL,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload)
        if not isinstance(data, list):
            return {}

        competitions: Dict[int, Dict[str, Optional[str]]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            competition_id = _safe_int(_first_value(item, ["i", "id", "competitionId", "ci"]))
            if competition_id is None:
                continue
            competitions[competition_id] = {
                "league_name": _extract_name(_first_value(item, ["n", "name", "sn"])) or "Unknown League",
                "country": _extract_name(_first_value(item, ["cid", "country", "countryCode"])),
            }
        return competitions
    finally:
        if owns_session:
            session.close()


def fetch_iddaa_market_config(
    session: Optional[requests.Session] = None,
) -> Dict[str, Dict[str, Any]]:
    owns_session = session is None
    session = session or build_iddaa_http_session()

    try:
        response = session.get(
            IDDAA_MARKET_CONFIG_URL,
            timeout=30,
            verify=IDDAA_VERIFY_SSL,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload)
        markets = data.get("m", {}) if isinstance(data, dict) else {}
        return markets if isinstance(markets, dict) else {}
    finally:
        if owns_session:
            session.close()


def _market_lookup_key(market_type: Any, state_type: Any, special_value: Any = None) -> List[str]:
    keys = []
    if special_value not in (None, ""):
        keys.append(f"{market_type}_{state_type}_{special_value}")
    keys.append(f"{market_type}_{state_type}")
    return keys


def _resolve_market_name(
    market: Dict[str, Any],
    market_lookup: Dict[str, Dict[str, Any]],
) -> str:
    for key in _market_lookup_key(market.get("t"), market.get("st"), market.get("sov")):
        if key in market_lookup:
            return str(market_lookup[key].get("n", key))
    return f"market_t{market.get('t')}_st{market.get('st')}"


def _extract_iddaa_1x2_odds(
    markets: Any,
    market_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    result = {"home_win_odds": None, "draw_odds": None, "away_win_odds": None}
    if not isinstance(markets, list):
        return result

    prioritized = []
    fallback = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        market_name = _resolve_market_name(market, market_lookup).lower()
        special_value = str(market.get("sov", "")).strip()
        if _safe_int(market.get("t")) == 1 and _safe_int(market.get("st")) == 1 and not special_value:
            prioritized.append(market)
        elif _looks_like_pure_match_result_market(market_name) and not special_value:
            prioritized.append(market)
        else:
            fallback.append(market)

    for market in prioritized + fallback:
        outcomes = market.get("o")
        if not isinstance(outcomes, list):
            continue

        mapping: Dict[str, Optional[float]] = {}
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            label = str(outcome.get("n", "")).strip().upper()
            odd_value = _safe_float(outcome.get("odd"))
            if label in {"1", "0", "X", "2"}:
                mapping[label] = odd_value

        if "1" in mapping and "2" in mapping and ("0" in mapping or "X" in mapping):
            result["home_win_odds"] = mapping.get("1")
            result["draw_odds"] = mapping.get("0") if mapping.get("0") is not None else mapping.get("X")
            result["away_win_odds"] = mapping.get("2")
            return result

    return result


def _extract_iddaa_totals_25_odds(
    markets: Any,
    market_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    result = {"over_25_odds": None, "under_25_odds": None}
    if not isinstance(markets, list):
        return result

    prioritized = []
    fallback = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        market_name = _resolve_market_name(market, market_lookup).lower()
        if str(market.get("sov", "")).strip() == "2.5" and _safe_int(market.get("st")) == 101:
            prioritized.append(market)
        elif "alt" in market_name and "ust" in market_name:
            fallback.append(market)

    for market in prioritized + fallback:
        if str(market.get("sov", "")).strip() != "2.5":
            continue

        outcomes = market.get("o")
        if not isinstance(outcomes, list):
            continue

        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            label = str(outcome.get("n", "")).strip().lower()
            odd_value = _safe_float(outcome.get("odd"))
            if label == "alt":
                result["under_25_odds"] = odd_value
            elif label == "üst":
                result["over_25_odds"] = odd_value

        if result["over_25_odds"] is not None or result["under_25_odds"] is not None:
            return result

    return result


def _extract_iddaa_btts_odds(
    markets: Any,
    market_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    result = {"btts_yes_odds": None, "btts_no_odds": None}
    if not isinstance(markets, list):
        return result

    prioritized = []
    fallback = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        market_name = _resolve_market_name(market, market_lookup).lower()
        normalized_name = (
            market_name.replace("Ã§", "c")
            .replace("Å", "s")
            .replace("Ä±", "i")
            .replace("Ã¼", "u")
            .replace("Ã¶", "o")
            .replace("ÄŸ", "g")
        )
        if "karsilikli" in normalized_name and "gol" in normalized_name:
            prioritized.append(market)
        elif "var" in normalized_name and "yok" in normalized_name:
            fallback.append(market)

    for market in prioritized + fallback:
        outcomes = market.get("o")
        if not isinstance(outcomes, list):
            continue

        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            label = (
                str(outcome.get("n", ""))
                .strip()
                .lower()
                .replace("Ã§", "c")
                .replace("Å", "s")
                .replace("Ä±", "i")
                .replace("Ã¼", "u")
                .replace("Ã¶", "o")
                .replace("ÄŸ", "g")
            )
            odd_value = _safe_float(outcome.get("odd"))
            if label == "var":
                result["btts_yes_odds"] = odd_value
            elif label == "yok":
                result["btts_no_odds"] = odd_value

        if result["btts_yes_odds"] is not None or result["btts_no_odds"] is not None:
            return result

    return result


def _normalized_market_text(value: Any) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("ÃƒÂ§", "c")
        .replace("Ã…", "s")
        .replace("Ã„Â±", "i")
        .replace("ÃƒÂ¼", "u")
        .replace("ÃƒÂ¶", "o")
        .replace("Ã„Å¸", "g")
        .replace("ç", "c")
        .replace("ş", "s")
        .replace("ı", "i")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ğ", "g")
    )


def _extract_iddaa_double_chance_odds(
    markets: Any,
    market_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    result = {
        "double_chance_1x_odds": None,
        "double_chance_12_odds": None,
        "double_chance_x2_odds": None,
    }
    if not isinstance(markets, list):
        return result

    for market in markets:
        if not isinstance(market, dict):
            continue
        market_name = _normalized_market_text(_resolve_market_name(market, market_lookup))
        if "cifte sans" not in market_name:
            continue

        outcomes = market.get("o")
        if not isinstance(outcomes, list):
            continue

        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            label = _normalized_market_text(outcome.get("n"))
            odd_value = _safe_float(outcome.get("odd"))
            compact_label = label.replace(" ", "")
            if compact_label == "1x":
                result["double_chance_1x_odds"] = odd_value
            elif compact_label == "12":
                result["double_chance_12_odds"] = odd_value
            elif compact_label == "x2":
                result["double_chance_x2_odds"] = odd_value

        if any(value is not None for value in result.values()):
            return result

    return result


def _extract_iddaa_handicap_10_odds(
    markets: Any,
    market_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    result = {
        "handicap_home_10_odds": None,
        "handicap_draw_10_odds": None,
        "handicap_away_10_odds": None,
    }
    if not isinstance(markets, list):
        return result

    for market in markets:
        if not isinstance(market, dict):
            continue
        market_name = _normalized_market_text(_resolve_market_name(market, market_lookup))
        special_value = str(market.get("sov", "")).strip()
        if "handikap" not in market_name:
            continue
        if special_value and special_value != "1:0":
            continue
        if special_value == "" and "1:0" not in market_name:
            continue

        outcomes = market.get("o")
        if not isinstance(outcomes, list):
            continue

        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            label = str(outcome.get("n", "")).strip().upper()
            odd_value = _safe_float(outcome.get("odd"))
            if label == "1":
                result["handicap_home_10_odds"] = odd_value
            elif label in {"0", "X"}:
                result["handicap_draw_10_odds"] = odd_value
            elif label == "2":
                result["handicap_away_10_odds"] = odd_value

        if any(value is not None for value in result.values()):
            return result

    return result


def _collect_market_labels(
    markets: Any,
    market_lookup: Dict[str, Dict[str, Any]],
    limit: int = 10,
) -> List[str]:
    if not isinstance(markets, list):
        return []

    labels: List[str] = []
    seen = set()
    for market in markets:
        if not isinstance(market, dict):
            continue
        name = _resolve_market_name(market, market_lookup)
        if name in seen:
            continue
        seen.add(name)
        labels.append(name)
        if len(labels) >= limit:
            break
    return labels


def _normalize_iddaa_event(
    raw_event: Dict[str, Any],
    competitions: Dict[int, Dict[str, Optional[str]]],
    market_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    event_id = _safe_int(_first_value(raw_event, ["i", "id", "eventId"]))
    home_team = _extract_name(_first_value(raw_event, ["hn", "homeTeam", "home"]))
    away_team = _extract_name(_first_value(raw_event, ["an", "awayTeam", "away"]))
    kickoff_time = _parse_unix_seconds(_first_value(raw_event, ["d", "date", "kickoff"]))

    if event_id is None or not home_team or not away_team or kickoff_time is None:
        return None
    if not _looks_like_real_team_name(home_team) or not _looks_like_real_team_name(away_team):
        return None

    competition_id = _safe_int(_first_value(raw_event, ["ci", "competitionId"]))
    competition = competitions.get(competition_id or -1, {})
    league_name = competition.get("league_name") or f"Competition {competition_id}"
    if _looks_like_bundle_text(league_name):
        return None
    market_names = _collect_market_labels(raw_event.get("m"), market_lookup)

    odds_1x2 = _extract_iddaa_1x2_odds(raw_event.get("m"), market_lookup)
    odds_totals = _extract_iddaa_totals_25_odds(raw_event.get("m"), market_lookup)
    odds_btts = _extract_iddaa_btts_odds(raw_event.get("m"), market_lookup)
    odds_double_chance = _extract_iddaa_double_chance_odds(raw_event.get("m"), market_lookup)
    odds_handicap = _extract_iddaa_handicap_10_odds(raw_event.get("m"), market_lookup)

    is_live = bool(raw_event.get("il"))
    status = "live" if is_live else "scheduled"

    notes_parts = []
    if competition_id is not None:
        notes_parts.append(f"iddaa_competition_id={competition_id}")
    if market_names:
        notes_parts.append("markets=" + ", ".join(market_names))

    return {
        "external_id": f"iddaa:{event_id}",
        "source": "iddaa",
        "league_name": league_name,
        "country": competition.get("country"),
        "home_team": home_team,
        "away_team": away_team,
        "kickoff_time": kickoff_time,
        "status": status,
        "is_live": is_live,
        "home_score": None,
        "away_score": None,
        "notes": " | ".join(notes_parts) if notes_parts else None,
        "odds": {
            **odds_1x2,
            **odds_totals,
            **odds_btts,
            **odds_double_chance,
            **odds_handicap,
        },
    }


def _is_suspicious_upcoming_match(match: Match) -> bool:
    if match.status not in {"scheduled", "live"}:
        return False

    home_name = match.home_team.name if match.home_team else None
    away_name = match.away_team.name if match.away_team else None
    league_name = match.league.name if match.league else None
    notes_text = str(match.notes or "")

    if not _looks_like_real_team_name(home_name) or not _looks_like_real_team_name(away_name):
        return True
    if league_name and _looks_like_bundle_text(league_name):
        return True
    if notes_text and _looks_like_bundle_text(notes_text):
        return True
    return False


def cleanup_suspicious_upcoming_matches() -> Dict[str, Any]:
    create_tables()
    session = get_session()
    removed_count = 0
    examples: List[Dict[str, Any]] = []

    try:
        matches = (
            session.query(Match)
            .filter(Match.status.in_(["scheduled", "live"]))
            .all()
        )

        for match in matches:
            if not _is_suspicious_upcoming_match(match):
                continue

            if len(examples) < 10:
                examples.append(
                    {
                        "match_id": match.id,
                        "source": match.source,
                        "status": match.status,
                        "league_name": match.league.name if match.league else None,
                        "home_team": match.home_team.name if match.home_team else None,
                        "away_team": match.away_team.name if match.away_team else None,
                    }
                )

            session.delete(match)
            removed_count += 1

        session.commit()
        return {
            "ok": True,
            "removed_count": removed_count,
            "examples": examples,
        }
    except Exception as exc:
        session.rollback()
        return {"ok": False, "error": str(exc)}
    finally:
        session.close()


def fetch_iddaa_upcoming(include_highlighted: bool = True) -> List[Dict[str, Any]]:
    session = build_iddaa_http_session()

    try:
        competitions = fetch_iddaa_competitions(session=session)
        market_lookup = fetch_iddaa_market_config(session=session)

        responses = []
        urls = [IDDAA_EVENTS_URL]
        if include_highlighted:
            urls.append(IDDAA_HIGHLIGHTED_EVENTS_URL)

        for url in urls:
            response = session.get(url, timeout=30, verify=IDDAA_VERIFY_SSL)
            response.raise_for_status()
            responses.append(response.json())

        normalized: List[Dict[str, Any]] = []
        for payload in responses:
            data = payload.get("data", payload)
            if isinstance(data, dict):
                events = data.get("events", [])
            elif isinstance(data, list):
                events = data
            else:
                events = []

            for raw_event in events:
                if not isinstance(raw_event, dict):
                    continue
                event = _normalize_iddaa_event(
                    raw_event,
                    competitions=competitions,
                    market_lookup=market_lookup,
                )
                if event is not None:
                    normalized.append(event)

        unique_events: List[Dict[str, Any]] = []
        seen = set()
        for item in normalized:
            if item["external_id"] in seen:
                continue
            seen.add(item["external_id"])
            unique_events.append(item)

        return unique_events
    finally:
        session.close()


def persist_upcoming_from_iddaa(include_highlighted: bool = True) -> Dict[str, Any]:
    create_tables()
    session = get_session()
    created = 0

    try:
        events = fetch_iddaa_upcoming(include_highlighted=include_highlighted)
        for event in events:
            match = upsert_match(session, event)
            odds = event.get("odds") or {}
            if any(value is not None for value in odds.values()):
                add_odds_snapshot(
                    session=session,
                    match=match,
                    bookmaker="iddaa",
                    market="1X2",
                    home_win_odds=odds.get("home_win_odds"),
                    draw_odds=odds.get("draw_odds"),
                    away_win_odds=odds.get("away_win_odds"),
                    over_25_odds=odds.get("over_25_odds"),
                    under_25_odds=odds.get("under_25_odds"),
                    btts_yes_odds=odds.get("btts_yes_odds"),
                    btts_no_odds=odds.get("btts_no_odds"),
                    double_chance_1x_odds=odds.get("double_chance_1x_odds"),
                    double_chance_12_odds=odds.get("double_chance_12_odds"),
                    double_chance_x2_odds=odds.get("double_chance_x2_odds"),
                    handicap_home_10_odds=odds.get("handicap_home_10_odds"),
                    handicap_draw_10_odds=odds.get("handicap_draw_10_odds"),
                    handicap_away_10_odds=odds.get("handicap_away_10_odds"),
                )
            created += 1

        session.commit()
        cleanup_result = cleanup_suspicious_upcoming_matches()
        cleanup_session = get_session()
        try:
            odds_cleanup_result = cleanup_invalid_active_1x2_odds_snapshots(cleanup_session)
            cleanup_session.commit()
        finally:
            cleanup_session.close()
        return {
            "ok": True,
            "inserted_or_updated": created,
            "cleanup_result": cleanup_result,
            "odds_cleanup_result": odds_cleanup_result,
        }
    except Exception as exc:
        session.rollback()
        return {"ok": False, "error": str(exc)}
    finally:
        session.close()


def _extract_1x2_odds(event: Dict[str, Any]) -> Dict[str, Optional[float]]:
    result = {"home_win_odds": None, "draw_odds": None, "away_win_odds": None}

    for node in _walk(event):
        if not isinstance(node, dict):
            continue

        market_name = _first_value(node, ["marketName", "name", "title", "desc", "description"])
        market_name_str = str(market_name).lower() if market_name is not None else ""

        if not _looks_like_pure_match_result_market(market_name_str):
            continue

        selections = _first_value(node, ["selections", "outcomes", "options", "items"])
        if not isinstance(selections, list):
            continue

        for selection in selections:
            if not isinstance(selection, dict):
                continue
            label = _first_value(selection, ["label", "name", "code", "title", "result"])
            label = str(label).strip().upper() if label is not None else ""
            odd_value = _safe_float(_first_value(selection, ["odd", "odds", "price", "value", "rate"]))

            if label == "1":
                result["home_win_odds"] = odd_value
            elif label in {"X", "0"}:
                result["draw_odds"] = odd_value
            elif label == "2":
                result["away_win_odds"] = odd_value

        if any(value is not None for value in result.values()):
            return result

    return result


def _extract_1x2_odds_robust(event: Dict[str, Any]) -> Dict[str, Optional[float]]:
    result = _extract_1x2_odds(event)
    if any(value is not None for value in result.values()):
        return result

    fallback = {"home_win_odds": None, "draw_odds": None, "away_win_odds": None}
    for node in _walk(event):
        if not isinstance(node, dict):
            continue

        market_name = _first_value(node, ["marketName", "name", "title", "desc", "description"])
        market_name_str = str(market_name).lower() if market_name is not None else ""
        if not _looks_like_pure_match_result_market(market_name_str):
            continue

        selections = _first_value(node, ["selections", "outcomes", "options", "items"])
        if not isinstance(selections, list):
            continue

        for selection in selections:
            if not isinstance(selection, dict):
                continue
            label = _first_value(selection, ["label", "name", "code", "title", "result"])
            label = str(label).strip().upper() if label is not None else ""
            odd_value = _safe_float(_first_value(selection, ["odd", "odds", "price", "value", "rate"]))

            if label in {"1", "MS1", "HOME"}:
                fallback["home_win_odds"] = odd_value
            elif label in {"X", "0", "MSX", "DRAW"}:
                fallback["draw_odds"] = odd_value
            elif label in {"2", "MS2", "AWAY"}:
                fallback["away_win_odds"] = odd_value

        if any(value is not None for value in fallback.values()):
            return fallback

    for node in _walk(event):
        if not isinstance(node, dict):
            continue

        for list_key in ["selections", "outcomes", "options", "items", "OC", "S"]:
            selections = _first_value(node, [list_key])
            if not isinstance(selections, list):
                continue

            pattern_result = {"home_win_odds": None, "draw_odds": None, "away_win_odds": None}
            for selection in selections:
                if not isinstance(selection, dict):
                    continue

                label = _first_value(
                    selection,
                    ["label", "name", "code", "title", "result", "N", "NO"],
                )
                odd_value = _safe_float(
                    _first_value(selection, ["odd", "odds", "price", "value", "rate", "O"])
                )
                if label is None or odd_value is None:
                    continue

                normalized_label = str(label).strip().upper()
                if normalized_label in {"1", "MS1", "HOME"}:
                    pattern_result["home_win_odds"] = odd_value
                elif normalized_label in {"X", "0", "MSX", "DRAW"}:
                    pattern_result["draw_odds"] = odd_value
                elif normalized_label in {"2", "MS2", "AWAY"}:
                    pattern_result["away_win_odds"] = odd_value

            if all(value is not None for value in pattern_result.values()):
                return pattern_result

    return fallback


def _fetch_nesine_payload(session: requests.Session) -> tuple[Any, Dict[str, Any]]:
    last_error: Optional[Exception] = None
    endpoints = [
        {
            "name": "prematch",
            "url": NESINE_PREMATCH_URL,
            "params": None,
        },
        {
            "name": "live",
            "url": NESINE_LIVE_URL,
            "params": {
                "eventVersion": 0,
                "oddVersion": 0,
                "_": int(time.time() * 1000),
            },
        },
    ]

    for endpoint in endpoints:
        try:
            response = session.get(endpoint["url"], params=endpoint["params"], timeout=20)
            response.raise_for_status()
            payload = response.json()
            meta = {
                "endpoint_name": endpoint["name"],
                "fetched_url": response.url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "response_preview": response.text[:1200],
            }
            return payload, meta
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Nesine payload alinamadi.")


def inspect_nesine_upcoming(sample_limit: int = 5) -> Dict[str, Any]:
    session = build_http_session()
    try:
        payload, meta = _fetch_nesine_payload(session)

        dict_count = 0
        list_count = 0
        candidate_count = 0
        sample_candidates: List[Dict[str, Any]] = []
        normalized_preview: List[Dict[str, Any]] = []

        for node in _walk(payload):
            if isinstance(node, dict):
                dict_count += 1
                if _looks_like_event(node):
                    candidate_count += 1
                    if len(sample_candidates) < sample_limit:
                        sample_candidates.append(_summarize_nesine_candidate_node(node))
            elif isinstance(node, list):
                list_count += 1

        normalized_events = fetch_nesine_upcoming()
        for event in normalized_events[:sample_limit]:
            normalized_preview.append(
                {
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "kickoff_time": event.get("kickoff_time").isoformat()
                    if event.get("kickoff_time")
                    else None,
                    "league_name": event.get("league_name"),
                    "has_any_1x2_odds": any(
                        value is not None for value in (event.get("odds") or {}).values()
                    ),
                    "source": event.get("source"),
                }
            )

        top_level_type = type(payload).__name__
        if isinstance(payload, dict):
            top_level_keys = sorted(str(key) for key in payload.keys())[:40]
        else:
            top_level_keys = []

        return {
            "ok": True,
            "source": "nesine",
            "status_code": meta.get("status_code"),
            "content_type": meta.get("content_type"),
            "endpoint_name": meta.get("endpoint_name"),
            "fetched_url": meta.get("fetched_url"),
            "top_level_type": top_level_type,
            "top_level_keys": top_level_keys,
            "walk_dict_count": dict_count,
            "walk_list_count": list_count,
            "candidate_event_count": candidate_count,
            "normalized_event_count": len(normalized_events),
            "sample_candidates": sample_candidates,
            "normalized_preview": normalized_preview,
            "response_preview": meta.get("response_preview"),
        }
    except Exception as exc:
        return {"ok": False, "source": "nesine", "error": str(exc)}
    finally:
        session.close()


def fetch_nesine_upcoming() -> List[Dict[str, Any]]:
    session = build_http_session()
    try:
        payload, _meta = _fetch_nesine_payload(session)

        normalized: List[Dict[str, Any]] = []

        for node in _walk(payload):
            if not isinstance(node, dict):
                continue
            if not _looks_like_event(node):
                continue

            home_team, away_team = _extract_event_teams(node)
            kickoff_time = _extract_event_kickoff(node)

            if not home_team or not away_team or kickoff_time is None:
                continue
            if not _looks_like_real_team_name(home_team) or not _looks_like_real_team_name(away_team):
                continue

            league_name = _extract_event_league(node)
            external_id = _first_value(node, ["id", "matchId", "eventId", "code"])
            if external_id is None:
                external_id = f"nesine:{home_team}:{away_team}:{kickoff_time.isoformat()}"

            odds = _extract_1x2_odds_robust(node)

            normalized.append(
                {
                    "external_id": str(external_id),
                    "source": "nesine",
                    "league_name": str(league_name).strip() if league_name else "Unknown League",
                    "country": None,
                    "home_team": home_team,
                    "away_team": away_team,
                    "kickoff_time": kickoff_time,
                    "status": "scheduled",
                    "is_live": False,
                    "home_score": None,
                    "away_score": None,
                    "odds": odds,
                }
            )

        unique_events = []
        seen = set()
        for item in normalized:
            if item["external_id"] in seen:
                continue
            seen.add(item["external_id"])
            unique_events.append(item)

        return unique_events
    finally:
        session.close()


def persist_upcoming_from_nesine() -> Dict[str, Any]:
    create_tables()
    session = get_session()
    created = 0

    try:
        events = fetch_nesine_upcoming()
        for event in events:
            match = upsert_match(session, event)
            odds = event.get("odds") or {}
            if any(odds.values()):
                add_odds_snapshot(
                    session=session,
                    match=match,
                    bookmaker="nesine",
                    market="1X2",
                    home_win_odds=odds.get("home_win_odds"),
                    draw_odds=odds.get("draw_odds"),
                    away_win_odds=odds.get("away_win_odds"),
                )
            created += 1

        session.commit()
        cleanup_result = cleanup_suspicious_upcoming_matches()
        cleanup_session = get_session()
        try:
            odds_cleanup_result = cleanup_invalid_active_1x2_odds_snapshots(cleanup_session)
            cleanup_session.commit()
        finally:
            cleanup_session.close()
        result = {
            "ok": True,
            "inserted_or_updated": created,
            "fetched_event_count": len(events),
            "source": "nesine",
            "cleanup_result": cleanup_result,
            "odds_cleanup_result": odds_cleanup_result,
        }
        if len(events) == 0:
            result["diagnostics"] = inspect_nesine_upcoming(sample_limit=5)
        return result
    except Exception as exc:
        session.rollback()
        return {"ok": False, "error": str(exc)}
    finally:
        session.close()


def _three_way_probs(home_strength: float, away_strength: float) -> Dict[str, float]:
    diff = home_strength - away_strength + 0.18
    base_home = 1 / (1 + math.exp(-1.25 * diff))
    draw = max(0.18, 0.28 - abs(diff) * 0.06)
    remain = 1.0 - draw
    home = remain * base_home
    away = remain - home

    total = home + draw + away
    return {"home": home / total, "draw": draw / total, "away": away / total}


def _odds_from_probs(probs: Dict[str, float], overround: float = 1.07) -> Dict[str, float]:
    return {
        "home_win_odds": round(overround / probs["home"], 2),
        "draw_odds": round(overround / probs["draw"], 2),
        "away_win_odds": round(overround / probs["away"], 2),
    }


def _simulate_goals(home_strength: float, away_strength: float) -> Dict[str, int]:
    home_xg = max(0.25, 1.35 + 0.45 * (home_strength - away_strength) + 0.22)
    away_xg = max(0.25, 1.15 + 0.45 * (away_strength - home_strength))

    return {
        "home_score": int(np.random.poisson(home_xg)),
        "away_score": int(np.random.poisson(away_xg)),
    }


def seed_demo_data(force: bool = False) -> Dict[str, Any]:
    create_tables()
    session = get_session()

    try:
        existing = session.query(Match).count()
        if existing > 0 and not force:
            return {
                "ok": True,
                "message": "Veritabani zaten veri iceriyor. Yeniden uretmek icin force=True kullan.",
                "match_count": existing,
            }

        if force:
            session.query(Match).delete()
            session.commit()

        random.seed(42)
        np.random.seed(42)

        leagues = {
            "Turkey Super Lig": [
                "Galatasaray",
                "Fenerbahce",
                "Besiktas",
                "Trabzonspor",
                "Basaksehir",
                "Samsunspor",
                "Konyaspor",
                "Kasimpasa",
            ],
            "England Premier League": [
                "Arsenal",
                "Liverpool",
                "Manchester City",
                "Chelsea",
                "Tottenham",
                "Newcastle",
                "Aston Villa",
                "Brighton",
            ],
        }

        team_strength = {}
        for league_name, teams in leagues.items():
            for team in teams:
                team_strength[team] = random.uniform(-0.45, 0.45)

        created_matches = 0
        start_date = datetime.utcnow() - timedelta(days=260)

        for league_name, teams in leagues.items():
            fixtures = list(permutations(teams, 2))
            random.shuffle(fixtures)

            for idx, (home, away) in enumerate(fixtures[:80]):
                kickoff = start_date + timedelta(days=idx * 2 + random.randint(0, 1))
                probs = _three_way_probs(team_strength[home], team_strength[away])
                scores = _simulate_goals(team_strength[home], team_strength[away])

                match_data = {
                    "external_id": f"demo:{league_name}:{home}:{away}:{kickoff.isoformat()}",
                    "source": "demo",
                    "league_name": league_name,
                    "country": "Demo",
                    "home_team": home,
                    "away_team": away,
                    "kickoff_time": kickoff,
                    "status": "completed",
                    "is_live": False,
                    "home_score": scores["home_score"],
                    "away_score": scores["away_score"],
                }
                match = upsert_match(session, match_data)
                odds = _odds_from_probs(probs)
                add_odds_snapshot(
                    session=session,
                    match=match,
                    bookmaker="demo_book",
                    market="1X2",
                    home_win_odds=odds["home_win_odds"],
                    draw_odds=odds["draw_odds"],
                    away_win_odds=odds["away_win_odds"],
                )
                created_matches += 1

        for league_name, teams in leagues.items():
            for i in range(0, len(teams), 2):
                home = teams[i]
                away = teams[(i + 1) % len(teams)]
                kickoff = datetime.utcnow() + timedelta(days=(i // 2) + 1, hours=20)

                probs = _three_way_probs(team_strength[home], team_strength[away])
                odds = _odds_from_probs(probs)

                match_data = {
                    "external_id": f"demo-upcoming:{league_name}:{home}:{away}:{kickoff.isoformat()}",
                    "source": "demo",
                    "league_name": league_name,
                    "country": "Demo",
                    "home_team": home,
                    "away_team": away,
                    "kickoff_time": kickoff,
                    "status": "scheduled",
                    "is_live": False,
                    "home_score": None,
                    "away_score": None,
                }
                match = upsert_match(session, match_data)
                add_odds_snapshot(
                    session=session,
                    match=match,
                    bookmaker="demo_book",
                    market="1X2",
                    home_win_odds=odds["home_win_odds"],
                    draw_odds=odds["draw_odds"],
                    away_win_odds=odds["away_win_odds"],
                )
                created_matches += 1

        session.commit()
        return {"ok": True, "created": created_matches}
    except Exception as exc:
        session.rollback()
        return {"ok": False, "error": str(exc)}
    finally:
        session.close()
