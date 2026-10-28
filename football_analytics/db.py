from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import os

from pathlib import Path

# Projenin ana dizinini bul (football_analytics klasörünün bir üstü)
BASE_DIR = Path(__file__).resolve().parent.parent

# 'data' klasörünü otomatik oluştur (yoksa)
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import Session, declarative_base, joinedload, relationship, sessionmaker

from football_analytics.config import DATABASE_URL
from football_analytics.normalize import best_team_alias_match, canonical_team_name, normalize_team_key

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False,
        "timeout": 60,
    }

engine = create_engine(
    DATABASE_URL,
    future=True,
    echo=False,
    connect_args=connect_args,
)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=60000;")
        cursor.execute("PRAGMA temp_store=MEMORY;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)
Base = declarative_base()


class League(Base):
    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    country = Column(String(255), nullable=True)

    matches = relationship("Match", back_populates="league")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)

    home_matches = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )
    aliases = relationship("TeamAlias", back_populates="team", cascade="all, delete-orphan")


class TeamAlias(Base):
    __tablename__ = "team_aliases"
    __table_args__ = (UniqueConstraint("alias", name="uq_team_alias_alias"),)

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    alias = Column(String(255), nullable=False)
    normalized_alias = Column(String(255), nullable=False, index=True)
    source = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    team = relationship("Team", back_populates="aliases")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    external_id = Column(String(255), unique=True, nullable=False, index=True)
    source = Column(String(100), nullable=False, default="demo")
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    kickoff_time = Column(DateTime, nullable=False, index=True)
    status = Column(String(50), nullable=False, default="scheduled")
    is_live = Column(Boolean, nullable=False, default=False)

    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    venue = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    odds_snapshots = relationship(
        "OddsSnapshot", back_populates="match", cascade="all, delete-orphan"
    )
    predictions = relationship(
        "Prediction", back_populates="match", cascade="all, delete-orphan"
    )


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)

    bookmaker = Column(String(100), nullable=False, default="unknown")
    market = Column(String(50), nullable=False, default="1X2")

    home_win_odds = Column(Float, nullable=True)
    draw_odds = Column(Float, nullable=True)
    away_win_odds = Column(Float, nullable=True)

    over_25_odds = Column(Float, nullable=True)
    under_25_odds = Column(Float, nullable=True)
    btts_yes_odds = Column(Float, nullable=True)
    btts_no_odds = Column(Float, nullable=True)
    double_chance_1x_odds = Column(Float, nullable=True)
    double_chance_12_odds = Column(Float, nullable=True)
    double_chance_x2_odds = Column(Float, nullable=True)
    handicap_home_10_odds = Column(Float, nullable=True)
    handicap_draw_10_odds = Column(Float, nullable=True)
    handicap_away_10_odds = Column(Float, nullable=True)

    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    match = relationship("Match", back_populates="odds_snapshots")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)

    prob_home = Column(Float, nullable=False)
    prob_draw = Column(Float, nullable=False)
    prob_away = Column(Float, nullable=False)

    predicted_label = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)

    edge_home = Column(Float, nullable=True)
    edge_draw = Column(Float, nullable=True)
    edge_away = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    match = relationship("Match", back_populates="predictions")


class SavedCoupon(Base):
    __tablename__ = "saved_coupons"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="open", index=True)
    segment = Column(String(50), nullable=True)
    source_mode = Column(String(100), nullable=True)
    total_odds = Column(Float, nullable=True)
    flat_total_stake = Column(Float, nullable=True)
    expected_profit = Column(Float, nullable=True)
    actual_total_odds = Column(Float, nullable=True)
    actual_total_stake = Column(Float, nullable=True)
    realized_return = Column(Float, nullable=True)
    settled_profit = Column(Float, nullable=True)
    settled_at = Column(DateTime, nullable=True)
    is_manual = Column(Boolean, nullable=False, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    items = relationship(
        "SavedCouponItem",
        back_populates="coupon",
        cascade="all, delete-orphan",
        order_by="SavedCouponItem.selection_order.asc()",
    )


class SavedCouponItem(Base):
    __tablename__ = "saved_coupon_items"

    id = Column(Integer, primary_key=True)
    coupon_id = Column(Integer, ForeignKey("saved_coupons.id"), nullable=False, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=True, index=True)
    selection_order = Column(Integer, nullable=False, default=1)
    kickoff_time = Column(DateTime, nullable=True)
    league_name = Column(String(255), nullable=True)
    home_team = Column(String(255), nullable=False)
    away_team = Column(String(255), nullable=False)
    recommended_label = Column(String(20), nullable=False)
    recommended_market = Column(String(100), nullable=False)
    recommended_odds = Column(Float, nullable=True)
    recommended_probability = Column(Float, nullable=True)
    recommended_edge = Column(Float, nullable=True)
    flat_stake = Column(Float, nullable=True)
    actual_odds = Column(Float, nullable=True)
    source = Column(String(100), nullable=True)
    odds_source = Column(String(100), nullable=True)
    result_status = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)

    coupon = relationship("SavedCoupon", back_populates="items")
    match = relationship("Match")


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith("sqlite"):
        with engine.begin() as connection:
            existing_columns = {
                row[1]
                for row in connection.exec_driver_sql("PRAGMA table_info(odds_snapshots)").fetchall()
            }
            if "btts_yes_odds" not in existing_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE odds_snapshots ADD COLUMN btts_yes_odds FLOAT"
                )
            if "btts_no_odds" not in existing_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE odds_snapshots ADD COLUMN btts_no_odds FLOAT"
                )
            if "double_chance_1x_odds" not in existing_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE odds_snapshots ADD COLUMN double_chance_1x_odds FLOAT"
                )
            if "double_chance_12_odds" not in existing_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE odds_snapshots ADD COLUMN double_chance_12_odds FLOAT"
                )
            if "double_chance_x2_odds" not in existing_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE odds_snapshots ADD COLUMN double_chance_x2_odds FLOAT"
                )
            if "handicap_home_10_odds" not in existing_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE odds_snapshots ADD COLUMN handicap_home_10_odds FLOAT"
                )
            if "handicap_draw_10_odds" not in existing_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE odds_snapshots ADD COLUMN handicap_draw_10_odds FLOAT"
                )
            if "handicap_away_10_odds" not in existing_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE odds_snapshots ADD COLUMN handicap_away_10_odds FLOAT"
                )

            saved_coupon_columns = {
                row[1]
                for row in connection.exec_driver_sql("PRAGMA table_info(saved_coupons)").fetchall()
            }
            if "actual_total_odds" not in saved_coupon_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE saved_coupons ADD COLUMN actual_total_odds FLOAT"
                )
            if "actual_total_stake" not in saved_coupon_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE saved_coupons ADD COLUMN actual_total_stake FLOAT"
                )
            if "realized_return" not in saved_coupon_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE saved_coupons ADD COLUMN realized_return FLOAT"
                )
            if "settled_profit" not in saved_coupon_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE saved_coupons ADD COLUMN settled_profit FLOAT"
                )
            if "settled_at" not in saved_coupon_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE saved_coupons ADD COLUMN settled_at DATETIME"
                )
            if "is_manual" not in saved_coupon_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE saved_coupons ADD COLUMN is_manual BOOLEAN DEFAULT 0"
                )

            saved_coupon_item_columns = {
                row[1]
                for row in connection.exec_driver_sql("PRAGMA table_info(saved_coupon_items)").fetchall()
            }
            if "actual_odds" not in saved_coupon_item_columns:
                connection.exec_driver_sql(
                    "ALTER TABLE saved_coupon_items ADD COLUMN actual_odds FLOAT"
                )


def get_session() -> Session:
    return SessionLocal()


def _clean_name(value: str) -> str:
    return " ".join((value or "").strip().split())


def _first_alias_by_alias(session: Session, alias: str) -> Optional["TeamAlias"]:
    return (
        session.query(TeamAlias)
        .filter(TeamAlias.alias == alias)
        .order_by(TeamAlias.id.asc())
        .first()
    )


def _first_alias_by_normalized_alias(
    session: Session,
    normalized_alias: str,
) -> Optional["TeamAlias"]:
    return (
        session.query(TeamAlias)
        .filter(TeamAlias.normalized_alias == normalized_alias)
        .order_by(TeamAlias.id.asc())
        .first()
    )


def _first_team_by_name(session: Session, team_name: str) -> Optional["Team"]:
    return session.query(Team).filter(Team.name == team_name).order_by(Team.id.asc()).first()


def _first_league_by_name(session: Session, league_name: str) -> Optional["League"]:
    return (
        session.query(League)
        .filter(League.name == league_name)
        .order_by(League.id.asc())
        .first()
    )


def _first_match_by_external_id(session: Session, external_id: str) -> Optional["Match"]:
    return (
        session.query(Match)
        .filter(Match.external_id == external_id)
        .order_by(Match.id.asc())
        .first()
    )


def ensure_team_alias(
    session: Session,
    team: Team,
    alias: str,
    source: Optional[str] = None,
) -> None:
    alias = _clean_name(alias)
    if not alias:
        return

    existing = _first_alias_by_alias(session, alias)
    if existing:
        if existing.team_id != team.id:
            return
        if source and not existing.source:
            existing.source = source
        return

    session.add(
        TeamAlias(
            team_id=team.id,
            alias=alias,
            normalized_alias=normalize_team_key(alias),
            source=source,
        )
    )
    session.flush()


def list_team_aliases(session: Session) -> list[dict[str, Any]]:
    aliases = (
        session.query(TeamAlias)
        .join(Team, Team.id == TeamAlias.team_id)
        .order_by(Team.name.asc(), TeamAlias.alias.asc())
        .all()
    )
    return [
        {
            "alias_id": alias.id,
            "team_id": alias.team_id,
            "team_name": alias.team.name if alias.team else None,
            "alias": alias.alias,
            "normalized_alias": alias.normalized_alias,
            "source": alias.source,
            "created_at": alias.created_at,
        }
        for alias in aliases
    ]


def list_teams(session: Session) -> list[dict[str, Any]]:
    teams = session.query(Team).order_by(Team.name.asc()).all()
    return [{"team_id": team.id, "team_name": team.name} for team in teams]


def create_manual_team_alias(
    session: Session,
    team_id: int,
    alias: str,
    source: str = "manual",
) -> TeamAlias:
    team = session.query(Team).filter(Team.id == team_id).one()
    ensure_team_alias(session, team, alias, source=source)
    alias_row = _first_alias_by_alias(session, _clean_name(alias))
    if alias_row is None:
        raise ValueError("Alias olusturulamadi.")
    return alias_row


def update_team_alias(
    session: Session,
    alias_id: int,
    alias: Optional[str] = None,
    team_id: Optional[int] = None,
    source: Optional[str] = None,
) -> TeamAlias:
    alias_row = session.query(TeamAlias).filter(TeamAlias.id == alias_id).first()
    if alias_row is None:
        raise ValueError("Alias kaydi bulunamadi.")
    new_alias = _clean_name(alias if alias is not None else alias_row.alias)
    if not new_alias:
        raise ValueError("Alias bos olamaz.")

    target_team_id = team_id if team_id is not None else alias_row.team_id
    session.query(Team).filter(Team.id == target_team_id).one()

    conflict = (
        session.query(TeamAlias)
        .filter(TeamAlias.alias == new_alias, TeamAlias.id != alias_row.id)
        .order_by(TeamAlias.id.asc())
        .first()
    )
    if conflict:
        raise ValueError("Bu alias zaten baska bir takim kaydinda kullaniliyor.")

    alias_row.alias = new_alias
    alias_row.normalized_alias = normalize_team_key(new_alias)
    alias_row.team_id = target_team_id
    if source is not None:
        alias_row.source = _clean_name(source) or None
    session.flush()
    return alias_row


def delete_team_alias(session: Session, alias_id: int) -> None:
    alias_row = session.query(TeamAlias).filter(TeamAlias.id == alias_id).first()
    if alias_row is None:
        raise ValueError("Alias kaydi bulunamadi.")
    session.delete(alias_row)
    session.flush()


def get_or_create_team(session: Session, team_name: str, source: Optional[str] = None) -> Team:
    team_name = _clean_name(team_name)
    canonical_name = canonical_team_name(team_name)

    team = _first_team_by_name(session, canonical_name)
    if team:
        ensure_team_alias(session, team, team_name, source=source)
        return team

    alias = _first_alias_by_alias(session, team_name)
    if alias:
        if source and not alias.source:
            alias.source = source
        return alias.team

    normalized_alias = normalize_team_key(team_name)
    alias = _first_alias_by_normalized_alias(session, normalized_alias)
    if alias:
        ensure_team_alias(session, alias.team, team_name, source=source)
        return alias.team

    all_candidates = session.query(Team.id, Team.name).all()
    best_match_id = best_team_alias_match(team_name, all_candidates)
    if best_match_id is not None:
        matched_team = session.query(Team).filter(Team.id == best_match_id).one()
        ensure_team_alias(session, matched_team, team_name, source=source)
        return matched_team

    team = Team(name=canonical_name)
    session.add(team)
    session.flush()
    ensure_team_alias(session, team, canonical_name, source=source)
    if team_name != canonical_name:
        ensure_team_alias(session, team, team_name, source=source)
    return team


def get_or_create_league(
    session: Session, league_name: str, country: Optional[str] = None
) -> League:
    league_name = _clean_name(league_name)
    league = _first_league_by_name(session, league_name)
    if league:
        if country and not league.country:
            league.country = country
        return league

    league = League(name=league_name, country=country)
    session.add(league)
    session.flush()
    return league


def upsert_match(session: Session, match_data: Dict[str, Any]) -> Match:
    external_id = match_data["external_id"]
    match = _first_match_by_external_id(session, external_id)

    league = None
    if match_data.get("league_name"):
        league = get_or_create_league(
            session,
            league_name=match_data["league_name"],
            country=match_data.get("country"),
        )

    source = match_data.get("source")
    home_team = get_or_create_team(session, match_data["home_team"], source=source)
    away_team = get_or_create_team(session, match_data["away_team"], source=source)

    if match is None:
        match = Match(
            external_id=external_id,
            source=match_data.get("source", "demo"),
            league_id=league.id if league else None,
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            kickoff_time=match_data["kickoff_time"],
            status=match_data.get("status", "scheduled"),
            is_live=bool(match_data.get("is_live", False)),
            home_score=match_data.get("home_score"),
            away_score=match_data.get("away_score"),
            venue=match_data.get("venue"),
            notes=match_data.get("notes"),
        )
        session.add(match)
        session.flush()
        return match

    match.source = match_data.get("source", match.source)
    match.league_id = league.id if league else match.league_id
    match.home_team_id = home_team.id
    match.away_team_id = away_team.id
    match.kickoff_time = match_data["kickoff_time"]
    match.status = match_data.get("status", match.status)
    match.is_live = bool(match_data.get("is_live", match.is_live))
    match.home_score = match_data.get("home_score")
    match.away_score = match_data.get("away_score")
    match.venue = match_data.get("venue")
    match.notes = match_data.get("notes")
    session.flush()
    return match


def add_odds_snapshot(
    session: Session,
    match: Match,
    bookmaker: str,
    market: str = "1X2",
    home_win_odds: Optional[float] = None,
    draw_odds: Optional[float] = None,
    away_win_odds: Optional[float] = None,
    over_25_odds: Optional[float] = None,
    under_25_odds: Optional[float] = None,
    btts_yes_odds: Optional[float] = None,
    btts_no_odds: Optional[float] = None,
    double_chance_1x_odds: Optional[float] = None,
    double_chance_12_odds: Optional[float] = None,
    double_chance_x2_odds: Optional[float] = None,
    handicap_home_10_odds: Optional[float] = None,
    handicap_draw_10_odds: Optional[float] = None,
    handicap_away_10_odds: Optional[float] = None,
) -> OddsSnapshot:
    snapshot = OddsSnapshot(
        match_id=match.id,
        bookmaker=bookmaker,
        market=market,
        home_win_odds=home_win_odds,
        draw_odds=draw_odds,
        away_win_odds=away_win_odds,
        over_25_odds=over_25_odds,
        under_25_odds=under_25_odds,
        btts_yes_odds=btts_yes_odds,
        btts_no_odds=btts_no_odds,
        double_chance_1x_odds=double_chance_1x_odds,
        double_chance_12_odds=double_chance_12_odds,
        double_chance_x2_odds=double_chance_x2_odds,
        handicap_home_10_odds=handicap_home_10_odds,
        handicap_draw_10_odds=handicap_draw_10_odds,
        handicap_away_10_odds=handicap_away_10_odds,
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def _valid_1x2_triplet(
    home_win_odds: Optional[float],
    draw_odds: Optional[float],
    away_win_odds: Optional[float],
) -> bool:
    values = [home_win_odds, draw_odds, away_win_odds]
    if any(value is None for value in values):
        return False
    try:
        parsed = [float(value) for value in values]
    except Exception:
        return False
    return min(parsed) > 1.01 and max(parsed) <= 1000


def _valid_simple_price(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        parsed = float(value)
    except Exception:
        return False
    return 1.01 < parsed <= 1000


def _snapshot_has_valid_1x2(snapshot: "OddsSnapshot") -> bool:
    return _valid_1x2_triplet(
        snapshot.home_win_odds,
        snapshot.draw_odds,
        snapshot.away_win_odds,
    )


def _snapshot_has_valid_totals(snapshot: "OddsSnapshot") -> bool:
    return _valid_simple_price(snapshot.over_25_odds) and _valid_simple_price(snapshot.under_25_odds)


def _snapshot_has_valid_btts(snapshot: "OddsSnapshot") -> bool:
    return _valid_simple_price(snapshot.btts_yes_odds) and _valid_simple_price(snapshot.btts_no_odds)


def _snapshot_has_valid_double_chance(snapshot: "OddsSnapshot") -> bool:
    return all(
        _valid_simple_price(value)
        for value in (
            snapshot.double_chance_1x_odds,
            snapshot.double_chance_12_odds,
            snapshot.double_chance_x2_odds,
        )
    )


def _snapshot_has_valid_handicap_10(snapshot: "OddsSnapshot") -> bool:
    return _valid_1x2_triplet(
        snapshot.handicap_home_10_odds,
        snapshot.handicap_draw_10_odds,
        snapshot.handicap_away_10_odds,
    )


def _relative_gap(value_a: Optional[float], value_b: Optional[float]) -> Optional[float]:
    if not _valid_simple_price(value_a) or not _valid_simple_price(value_b):
        return None
    a = float(value_a)
    b = float(value_b)
    floor = min(a, b)
    if floor <= 0:
        return None
    return abs(a - b) / floor


def latest_1x2_odds(match: Match) -> Optional[OddsSnapshot]:
    candidates = [snapshot for snapshot in match.odds_snapshots if snapshot.market == "1X2"]
    if not candidates:
        return None
    valid_candidates = [snapshot for snapshot in candidates if _snapshot_has_valid_1x2(snapshot)]
    pool = valid_candidates or candidates
    return max(pool, key=lambda x: x.fetched_at or datetime.min)


def build_preferred_odds_lookup(
    matches: list[Match],
    preferred_bookmakers: tuple[str, ...] = ("iddaa", "nesine"),
) -> Dict[int, Dict[str, Any]]:
    grouped_matches: Dict[tuple[str, str, str], list[Match]] = {}
    for match in matches or []:
        home_name = match.home_team.name if match.home_team else ""
        away_name = match.away_team.name if match.away_team else ""
        kickoff = match.kickoff_time
        kickoff_key = ""
        if kickoff is not None:
            kickoff_key = kickoff.strftime("%Y-%m-%d")
        signature = (
            normalize_team_key(home_name),
            normalize_team_key(away_name),
            kickoff_key,
        )
        grouped_matches.setdefault(signature, []).append(match)

    lookup: Dict[int, Dict[str, Any]] = {}
    for grouped in grouped_matches.values():
        snapshots_by_bookmaker: Dict[str, list[OddsSnapshot]] = {}
        all_snapshots: list[OddsSnapshot] = []

        for match in grouped:
            for snapshot in match.odds_snapshots:
                bookmaker = str(snapshot.bookmaker or "unknown").strip().lower()
                snapshots_by_bookmaker.setdefault(bookmaker, []).append(snapshot)
                all_snapshots.append(snapshot)

        all_snapshots.sort(key=lambda item: item.fetched_at or datetime.min, reverse=True)

        for bookmaker, snapshots in snapshots_by_bookmaker.items():
            snapshots.sort(key=lambda item: item.fetched_at or datetime.min, reverse=True)

        def pick_snapshot(
            validator,
            family_fields: tuple[str, ...],
        ) -> tuple[dict[str, Any], Optional[str]]:
            for bookmaker in preferred_bookmakers:
                snapshots = snapshots_by_bookmaker.get(bookmaker, [])
                for snapshot in snapshots:
                    if validator(snapshot):
                        return (
                            {field_name: getattr(snapshot, field_name, None) for field_name in family_fields},
                            bookmaker,
                        )

            for snapshot in all_snapshots:
                if validator(snapshot):
                    source_name = str(snapshot.bookmaker or "unknown").strip().lower()
                    return (
                        {field_name: getattr(snapshot, field_name, None) for field_name in family_fields},
                        source_name,
                    )

            return ({field_name: None for field_name in family_fields}, None)

        def snapshot_field_map(
            snapshot: Optional["OddsSnapshot"],
            family_fields: tuple[str, ...],
        ) -> dict[str, Any]:
            return {
                field_name: getattr(snapshot, field_name, None) if snapshot is not None else None
                for field_name in family_fields
            }

        def best_snapshot_for_bookmaker(
            bookmaker: str,
            validator,
        ) -> Optional["OddsSnapshot"]:
            for snapshot in snapshots_by_bookmaker.get(bookmaker, []):
                if validator(snapshot):
                    return snapshot
            return None

        def first_alternative_snapshot(
            selected_source: Optional[str],
            validator,
        ) -> tuple[Optional["OddsSnapshot"], Optional[str]]:
            if selected_source is None:
                return None, None

            ordered_sources = list(preferred_bookmakers)
            for bookmaker in snapshots_by_bookmaker.keys():
                if bookmaker not in ordered_sources:
                    ordered_sources.append(bookmaker)

            for bookmaker in ordered_sources:
                if bookmaker == selected_source:
                    continue
                snapshot = best_snapshot_for_bookmaker(bookmaker, validator)
                if snapshot is not None:
                    return snapshot, bookmaker
            return None, None

        def compute_family_gap(
            chosen_payload: dict[str, Any],
            alternative_snapshot: Optional["OddsSnapshot"],
            family_fields: tuple[str, ...],
        ) -> Optional[float]:
            if alternative_snapshot is None:
                return None

            gaps: list[float] = []
            alternative_payload = snapshot_field_map(alternative_snapshot, family_fields)
            for field_name in family_fields:
                gap = _relative_gap(chosen_payload.get(field_name), alternative_payload.get(field_name))
                if gap is not None:
                    gaps.append(gap)
            if not gaps:
                return None
            return max(gaps)

        merged = {
            "home_win_odds": None,
            "draw_odds": None,
            "away_win_odds": None,
            "over_25_odds": None,
            "under_25_odds": None,
            "btts_yes_odds": None,
            "btts_no_odds": None,
            "double_chance_1x_odds": None,
            "double_chance_12_odds": None,
            "double_chance_x2_odds": None,
            "handicap_home_10_odds": None,
            "handicap_draw_10_odds": None,
            "handicap_away_10_odds": None,
            "odds_source": None,
            "odds_consistency_ok": True,
            "odds_consistency_gap": None,
            "odds_consistency_market": None,
            "odds_consistency_reference_source": None,
        }

        source_hits: list[str] = []
        family_payload, family_source = pick_snapshot(
            _snapshot_has_valid_1x2,
            ("home_win_odds", "draw_odds", "away_win_odds"),
        )
        merged.update(family_payload)
        if family_source:
            source_hits.append(family_source)
        family_alt_snapshot, family_alt_source = first_alternative_snapshot(
            family_source,
            _snapshot_has_valid_1x2,
        )
        family_gap = compute_family_gap(
            family_payload,
            family_alt_snapshot,
            ("home_win_odds", "draw_odds", "away_win_odds"),
        )

        totals_payload, totals_source = pick_snapshot(
            _snapshot_has_valid_totals,
            ("over_25_odds", "under_25_odds"),
        )
        merged.update(totals_payload)
        if totals_source:
            source_hits.append(totals_source)
        totals_alt_snapshot, totals_alt_source = first_alternative_snapshot(
            totals_source,
            _snapshot_has_valid_totals,
        )
        totals_gap = compute_family_gap(
            totals_payload,
            totals_alt_snapshot,
            ("over_25_odds", "under_25_odds"),
        )

        btts_payload, btts_source = pick_snapshot(
            _snapshot_has_valid_btts,
            ("btts_yes_odds", "btts_no_odds"),
        )
        merged.update(btts_payload)
        if btts_source:
            source_hits.append(btts_source)
        btts_alt_snapshot, btts_alt_source = first_alternative_snapshot(
            btts_source,
            _snapshot_has_valid_btts,
        )
        btts_gap = compute_family_gap(
            btts_payload,
            btts_alt_snapshot,
            ("btts_yes_odds", "btts_no_odds"),
        )

        dc_payload, dc_source = pick_snapshot(
            _snapshot_has_valid_double_chance,
            ("double_chance_1x_odds", "double_chance_12_odds", "double_chance_x2_odds"),
        )
        merged.update(dc_payload)
        if dc_source:
            source_hits.append(dc_source)
        dc_alt_snapshot, dc_alt_source = first_alternative_snapshot(
            dc_source,
            _snapshot_has_valid_double_chance,
        )
        dc_gap = compute_family_gap(
            dc_payload,
            dc_alt_snapshot,
            ("double_chance_1x_odds", "double_chance_12_odds", "double_chance_x2_odds"),
        )

        handicap_payload, handicap_source = pick_snapshot(
            _snapshot_has_valid_handicap_10,
            ("handicap_home_10_odds", "handicap_draw_10_odds", "handicap_away_10_odds"),
        )
        merged.update(handicap_payload)
        if handicap_source:
            source_hits.append(handicap_source)
        handicap_alt_snapshot, handicap_alt_source = first_alternative_snapshot(
            handicap_source,
            _snapshot_has_valid_handicap_10,
        )
        handicap_gap = compute_family_gap(
            handicap_payload,
            handicap_alt_snapshot,
            ("handicap_home_10_odds", "handicap_draw_10_odds", "handicap_away_10_odds"),
        )

        if source_hits:
            unique_sources: list[str] = []
            for source_name in source_hits:
                if source_name not in unique_sources:
                    unique_sources.append(source_name)
            merged["odds_source"] = " > ".join(unique_sources)

        family_gaps = [
            ("1X2", family_gap, family_alt_source),
            ("Alt/Ust", totals_gap, totals_alt_source),
            ("KG", btts_gap, btts_alt_source),
            ("Cifte Sans", dc_gap, dc_alt_source),
            ("Handikap 1:0", handicap_gap, handicap_alt_source),
        ]
        family_gaps = [item for item in family_gaps if item[1] is not None]
        if family_gaps:
            biggest_market, biggest_gap, ref_source = max(family_gaps, key=lambda item: item[1])
            merged["odds_consistency_gap"] = round(float(biggest_gap), 4)
            merged["odds_consistency_market"] = biggest_market
            merged["odds_consistency_reference_source"] = ref_source
            merged["odds_consistency_ok"] = float(biggest_gap) <= 0.18

        for match in grouped:
            lookup[match.id] = merged.copy()

    return lookup


def cleanup_invalid_active_1x2_odds_snapshots(session: Session) -> Dict[str, Any]:
    snapshots = (
        session.query(OddsSnapshot)
        .join(Match, Match.id == OddsSnapshot.match_id)
        .filter(
            OddsSnapshot.market == "1X2",
            Match.status.in_(["scheduled", "live"]),
        )
        .all()
    )

    removed_count = 0
    examples: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if _snapshot_has_valid_1x2(snapshot):
            continue

        if len(examples) < 10:
            examples.append(
                {
                    "snapshot_id": snapshot.id,
                    "bookmaker": snapshot.bookmaker,
                    "match_id": snapshot.match_id,
                    "home_win_odds": snapshot.home_win_odds,
                    "draw_odds": snapshot.draw_odds,
                    "away_win_odds": snapshot.away_win_odds,
                }
            )
        session.delete(snapshot)
        removed_count += 1

    session.flush()
    return {"ok": True, "removed_count": removed_count, "examples": examples}


def create_saved_coupon(
    session: Session,
    name: str,
    picks,
    summary: Dict[str, Any],
    segment: str,
    source_mode: str,
    actual_total_odds: Optional[float] = None,
    actual_total_stake: Optional[float] = None,
    is_manual: bool = False,
    notes: Optional[str] = None,
) -> SavedCoupon:
    resolved_total_odds = actual_total_odds if actual_total_odds is not None else summary.get("total_odds")
    resolved_total_stake = (
        actual_total_stake if actual_total_stake is not None else summary.get("flat_total_stake")
    )
    coupon = SavedCoupon(
        name=name,
        status="open",
        segment=segment,
        source_mode=source_mode,
        total_odds=summary.get("total_odds"),
        flat_total_stake=summary.get("flat_total_stake"),
        expected_profit=summary.get("flat_expected_profit"),
        actual_total_odds=resolved_total_odds,
        actual_total_stake=resolved_total_stake,
        is_manual=bool(is_manual),
        notes=notes,
    )
    session.add(coupon)
    session.flush()

    if picks is not None:
        for selection_order, (_, row) in enumerate(picks.iterrows(), start=1):
            session.add(
                SavedCouponItem(
                    coupon_id=coupon.id,
                    match_id=int(row["match_id"]) if row.get("match_id") is not None else None,
                    selection_order=selection_order,
                    kickoff_time=row.get("kickoff_time"),
                    league_name=row.get("league_name"),
                    home_team=str(row.get("home_team") or ""),
                    away_team=str(row.get("away_team") or ""),
                    recommended_label=str(row.get("recommended_label") or ""),
                    recommended_market=str(row.get("recommended_market") or ""),
                    recommended_odds=float(row["recommended_odds"])
                    if row.get("recommended_odds") is not None
                    else None,
                    recommended_probability=float(row["recommended_probability"])
                    if row.get("recommended_probability") is not None
                    else None,
                    recommended_edge=float(row["recommended_edge"])
                    if row.get("recommended_edge") is not None
                    else None,
                    flat_stake=float(row["flat_stake"]) if row.get("flat_stake") is not None else None,
                    actual_odds=float(row["actual_odds"]) if row.get("actual_odds") is not None else (
                        float(row["recommended_odds"]) if row.get("recommended_odds") is not None else None
                    ),
                    source=str(row.get("source") or ""),
                    odds_source=str(row.get("odds_source") or ""),
                    result_status="open",
                )
            )

    session.flush()
    return coupon


def _saved_coupon_item_result(item: SavedCouponItem) -> str:
    if item.match is None:
        return str(item.result_status or "open").strip().lower() or "open"

    match = item.match
    if match.home_score is None or match.away_score is None:
        return "open"

    home_score = int(match.home_score)
    away_score = int(match.away_score)
    market = str(item.recommended_market or "").strip().lower()
    label = str(item.recommended_label or "").strip().upper()

    if market in {"ev sahibi", "beraberlik", "deplasman"} or label in {"H", "D", "A"}:
        if label == "H":
            return "won" if home_score > away_score else "lost"
        if label == "D":
            return "won" if home_score == away_score else "lost"
        if label == "A":
            return "won" if away_score > home_score else "lost"
    if market == "2.5 alt" or label == "U25":
        return "won" if (home_score + away_score) <= 2 else "lost"
    if market == "2.5 ust" or label == "O25":
        return "won" if (home_score + away_score) >= 3 else "lost"
    if market == "kg var":
        return "won" if home_score > 0 and away_score > 0 else "lost"
    if market == "kg yok":
        return "won" if home_score == 0 or away_score == 0 else "lost"
    if market == "cifte sans" or label in {"1X", "12", "X2"}:
        if label == "1X":
            return "won" if home_score >= away_score else "lost"
        if label == "12":
            return "won" if home_score != away_score else "lost"
        if label == "X2":
            return "won" if away_score >= home_score else "lost"
    if "handikap" in market or label in {"H10H", "H10D", "H10A"}:
        adjusted_home = home_score + 1
        adjusted_away = away_score
        if label == "H10H":
            return "won" if adjusted_home > adjusted_away else "lost"
        if label == "H10D":
            return "won" if adjusted_home == adjusted_away else "lost"
        if label == "H10A":
            return "won" if adjusted_home < adjusted_away else "lost"
    return str(item.result_status or "open").strip().lower() or "open"


def settle_saved_coupon(session: Session, coupon_id: int) -> Optional[SavedCoupon]:
    coupon = get_saved_coupon_with_items(session, coupon_id)
    if coupon is None:
        return None

    item_statuses: list[str] = []
    resolved_odds_product = 1.0
    has_settled_item = False

    for item in coupon.items:
        new_status = _saved_coupon_item_result(item)
        item.result_status = new_status
        item_statuses.append(new_status)
        if new_status == "won":
            item_odds = item.actual_odds or item.recommended_odds
            if item_odds:
                resolved_odds_product *= float(item_odds)
            has_settled_item = True
        elif new_status == "lost":
            has_settled_item = True

    actual_stake = float(coupon.actual_total_stake or coupon.flat_total_stake or 0.0)

    if any(status == "lost" for status in item_statuses):
        coupon.status = "lost"
        coupon.realized_return = 0.0
        coupon.settled_profit = round(-actual_stake, 2)
        coupon.settled_at = datetime.utcnow()
    elif item_statuses and all(status == "won" for status in item_statuses):
        coupon.status = "won"
        payout_odds = float(coupon.actual_total_odds or resolved_odds_product or coupon.total_odds or 0.0)
        realized_return = round(actual_stake * payout_odds, 2)
        coupon.realized_return = realized_return
        coupon.settled_profit = round(realized_return - actual_stake, 2)
        coupon.settled_at = datetime.utcnow()
    elif has_settled_item:
        coupon.status = "open"
        coupon.realized_return = None
        coupon.settled_profit = None
        coupon.settled_at = None
    else:
        coupon.status = "open"
        coupon.realized_return = None
        coupon.settled_profit = None
        coupon.settled_at = None

    session.flush()
    return coupon


def summarize_saved_coupon_performance(session: Session) -> Dict[str, Any]:
    coupons = session.query(SavedCoupon).order_by(SavedCoupon.created_at.desc()).all()
    settled = [coupon for coupon in coupons if str(coupon.status).lower() in {"won", "lost"}]
    total_staked = sum(float(coupon.actual_total_stake or coupon.flat_total_stake or 0.0) for coupon in settled)
    total_profit = sum(float(coupon.settled_profit or 0.0) for coupon in settled)
    roi = (total_profit / total_staked) if total_staked > 0 else None
    return {
        "total_coupons": len(coupons),
        "open_coupons": sum(1 for coupon in coupons if str(coupon.status).lower() == "open"),
        "won_coupons": sum(1 for coupon in coupons if str(coupon.status).lower() == "won"),
        "lost_coupons": sum(1 for coupon in coupons if str(coupon.status).lower() == "lost"),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(float(roi), 4) if roi is not None else None,
    }


def summarize_saved_coupon_preferences(session: Session) -> Dict[str, list[dict[str, Any]]]:
    coupons = session.query(SavedCoupon).order_by(SavedCoupon.created_at.desc()).all()
    item_rows: list[dict[str, Any]] = []
    for coupon in coupons:
        for item in coupon.items:
            item_rows.append(
                {
                    "league_name": item.league_name,
                    "home_team": item.home_team,
                    "away_team": item.away_team,
                    "recommended_market": item.recommended_market,
                    "recommended_label": item.recommended_label,
                    "result_status": item.result_status,
                }
            )

    if not item_rows:
        return {"top_leagues": [], "top_teams": [], "top_markets": []}

    frame = pd.DataFrame(item_rows)

    top_leagues = (
        frame.groupby("league_name", as_index=False)
        .agg(
            picks=("league_name", "count"),
            wins=("result_status", lambda series: int(series.fillna("").eq("won").sum())),
        )
        .sort_values(["picks", "wins"], ascending=[False, False])
        .head(8)
    )
    top_leagues["hit_rate"] = np.where(
        top_leagues["picks"] > 0,
        top_leagues["wins"] / top_leagues["picks"],
        np.nan,
    )

    team_rows = []
    for _, row in frame.iterrows():
        for team_name in [row.get("home_team"), row.get("away_team")]:
            if team_name:
                team_rows.append({"team_name": team_name, "result_status": row.get("result_status")})
    team_frame = pd.DataFrame(team_rows)
    top_teams = pd.DataFrame()
    if not team_frame.empty:
        top_teams = (
            team_frame.groupby("team_name", as_index=False)
            .agg(
                picks=("team_name", "count"),
                wins=("result_status", lambda series: int(series.fillna("").eq("won").sum())),
            )
            .sort_values(["picks", "wins"], ascending=[False, False])
            .head(8)
        )
        top_teams["hit_rate"] = np.where(
            top_teams["picks"] > 0,
            top_teams["wins"] / top_teams["picks"],
            np.nan,
        )

    top_markets = (
        frame.groupby(["recommended_market", "recommended_label"], as_index=False)
        .agg(
            picks=("recommended_market", "count"),
            wins=("result_status", lambda series: int(series.fillna("").eq("won").sum())),
        )
        .sort_values(["picks", "wins"], ascending=[False, False])
        .head(8)
    )
    top_markets["hit_rate"] = np.where(
        top_markets["picks"] > 0,
        top_markets["wins"] / top_markets["picks"],
        np.nan,
    )

    return {
        "top_leagues": top_leagues.to_dict("records"),
        "top_teams": top_teams.to_dict("records") if not top_teams.empty else [],
        "top_markets": top_markets.to_dict("records"),
    }


def list_saved_coupons(session: Session) -> list[dict]:
    coupons = session.query(SavedCoupon).order_by(SavedCoupon.created_at.desc()).all()
    rows: list[dict] = []
    for coupon in coupons:
        rows.append(
            {
                "coupon_id": coupon.id,
                "name": coupon.name,
                "status": coupon.status,
                "segment": coupon.segment,
                "source_mode": coupon.source_mode,
                "total_odds": coupon.total_odds,
                "flat_total_stake": coupon.flat_total_stake,
                "expected_profit": coupon.expected_profit,
                "actual_total_odds": coupon.actual_total_odds,
                "actual_total_stake": coupon.actual_total_stake,
                "realized_return": coupon.realized_return,
                "settled_profit": coupon.settled_profit,
                "settled_at": coupon.settled_at,
                "notes": coupon.notes,
                "is_manual": coupon.is_manual,
                "item_count": len(coupon.items),
                "created_at": coupon.created_at,
                "updated_at": coupon.updated_at,
            }
        )
    return rows


def get_saved_coupon_with_items(session: Session, coupon_id: int) -> Optional[SavedCoupon]:
    return (
        session.query(SavedCoupon)
        .options(joinedload(SavedCoupon.items).joinedload(SavedCouponItem.match))
        .filter(SavedCoupon.id == coupon_id)
        .order_by(SavedCoupon.id.asc())
        .first()
    )


def update_saved_coupon_status(session: Session, coupon_id: int, status: str) -> Optional[SavedCoupon]:
    coupon = get_saved_coupon_with_items(session, coupon_id)
    if coupon is None:
        return None
    coupon.status = str(status or "open").strip().lower()
    if coupon.status == "open":
        coupon.realized_return = None
        coupon.settled_profit = None
        coupon.settled_at = None
    session.flush()
    return coupon


def save_prediction(
    session: Session,
    match_id: int,
    model_name: str,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    predicted_label: str,
    confidence: float,
    edge_home: Optional[float] = None,
    edge_draw: Optional[float] = None,
    edge_away: Optional[float] = None,
) -> Prediction:
    prediction = Prediction(
        match_id=match_id,
        model_name=model_name,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        predicted_label=predicted_label,
        confidence=confidence,
        edge_home=edge_home,
        edge_draw=edge_draw,
        edge_away=edge_away,
    )
    session.add(prediction)
    return prediction
