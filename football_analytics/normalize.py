import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, Optional, Tuple

TEAM_ALIAS_OVERRIDES = {
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "man city": "Manchester City",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "psg": "Paris SG",
    "paris st g": "Paris SG",
    "ath madrid": "Atletico Madrid",
    "athletico madrid": "Atletico Madrid",
    "besiktas jk": "Besiktas",
    "fenerbahce sk": "Fenerbahce",
    "galatasaray sk": "Galatasaray",
    "tottenham hotspur": "Tottenham",
    "newcastle utd": "Newcastle",
    "qpr": "QPR",
    "sheff utd": "Sheffield United",
    "sheff wed": "Sheffield Wednesday",
    "spal 2013": "SPAL",
}

GENERIC_TOKENS = {
    "fc",
    "cf",
    "ac",
    "sc",
    "sk",
    "fk",
    "afc",
    "bk",
    "jk",
    "ud",
    "cd",
    "club",
    "de",
    "the",
}


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_team_key(name: str) -> str:
    text = _strip_accents((name or "").strip().lower())
    text = text.replace("&", " and ")
    # Keep reserve-team suffixes like II/III because dropping them causes
    # senior and reserve squads to collapse onto the same normalized key.
    text = re.sub(r"\b(u23|u21|u20|u19|u18)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [token for token in text.split() if token not in GENERIC_TOKENS]
    compact = " ".join(tokens).strip()
    return compact or text


def canonical_team_name(name: str) -> str:
    cleaned = " ".join((name or "").strip().split())
    normalized = normalize_team_key(cleaned)
    override = TEAM_ALIAS_OVERRIDES.get(normalized)
    return override or cleaned


def best_team_alias_match(
    raw_name: str,
    candidates: Iterable[Tuple[int, str]],
    threshold: float = 0.94,
) -> Optional[int]:
    target = normalize_team_key(raw_name)
    best_id = None
    best_score = 0.0

    for team_id, candidate_name in candidates:
        candidate_key = normalize_team_key(candidate_name)
        if target == candidate_key and candidate_key:
            return team_id

        score = SequenceMatcher(a=target, b=candidate_key).ratio()
        if score > best_score:
            best_score = score
            best_id = team_id

    if best_score >= threshold:
        return best_id
    return None
