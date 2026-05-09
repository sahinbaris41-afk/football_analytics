from datetime import datetime
import html as html_lib
import textwrap
from contextlib import contextmanager
from typing import Any, Dict
from zoneinfo import ZoneInfo
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy.orm import joinedload

from football_analytics.audit import build_system_health_report
from football_analytics.config import LOCAL_CSV_IMPORT_DIR
from football_analytics.backtest import compare_backtest_strategies, compare_model_backtests, run_backtest
from football_analytics.data_ingest import (
    FOOTBALL_DATA_LEAGUES,
    cleanup_suspicious_upcoming_matches,
    default_training_league_codes,
    default_historical_season_starts,
    download_bulk_import_csv_bundle,
    import_historical_results_from_local_csv_folder,
    persist_historical_results_from_football_data,
    persist_upcoming_from_iddaa,
    seed_demo_data,
)
from football_analytics.db import (
    Match,
    Prediction,
    build_preferred_odds_lookup,
    cleanup_invalid_active_1x2_odds_snapshots,
    create_manual_team_alias,
    create_saved_coupon,
    create_tables,
    delete_team_alias,
    get_saved_coupon_with_items,
    get_session,
    list_saved_coupons,
    settle_saved_coupon,
    summarize_saved_coupon_performance,
    summarize_saved_coupon_preferences,
    update_saved_coupon_status,
    list_team_aliases,
    list_teams,
    update_team_alias,
)
from football_analytics.explain import compute_model_explanations
from football_analytics.model_compare import train_and_compare_models
from football_analytics.normalize import normalize_team_key
from football_analytics.predict import predict_live_matches, predict_upcoming_matches
from football_analytics.recommend import (
    build_recommendation_audit,
    build_betting_recommendations,
    build_coupon_from_recommendations,
)
from football_analytics.train import MODEL_PATH, train_and_save_model


APP_TIMEZONE = ZoneInfo("Europe/Istanbul")
MATCH_DRIVER_LABELS = {
    "elo_diff": "Elo farki",
    "form_diff": "Form farki (son 5 mac puan)",
    "rest_diff": "Dinlenme gunu farki",
    "imp_home": "Piyasa ev sahibi olasiligi",
    "imp_draw": "Piyasa beraberlik olasiligi",
    "imp_away": "Piyasa deplasman olasiligi",
    "market_overround": "Piyasa marji",
    "league_quality_index": "Lig kalite katsayisi",
    "home_points_last5": "Ev sahibi son 5 puan ort.",
    "away_points_last5": "Deplasman son 5 puan ort.",
}
BRAND_TITLE = "GABRELA IDDAA TAHMIN SISTEMI"
BRAND_SUBTITLE = "Iddaa odakli tahmin, kupon takibi ve model analizi paneli"


def inject_app_shell_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --gabrela-green: #1fa866;
            --gabrela-green-2: #13824f;
            --gabrela-yellow: #ffd23f;
            --gabrela-ink: #f4f7fb;
            --gabrela-soft: #0f1217;
            --gabrela-panel: #151a21;
            --gabrela-panel-2: #1b222c;
            --gabrela-border: rgba(255,255,255,0.08);
            --gabrela-danger: #c1121f;
            --gabrela-muted: #9aa6b2;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(31,168,102,0.14) 0%, rgba(15,18,23,0) 28%),
                linear-gradient(180deg, #0c1015 0%, #121820 45%, #111822 100%);
            color: var(--gabrela-ink);
        }
        header[data-testid="stHeader"] {
            display: none !important;
            background: #0c1015 !important;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        div[data-testid="stDecoration"] {
            display: none !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
            background: transparent !important;
        }
        [data-testid="stHeaderActionElements"],
        [data-testid="stStatusWidget"] {
            display: none !important;
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 0.75rem !important;
        }
        .stApp > header {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0d11 0%, #10151d 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stCaption {
            color: var(--gabrela-ink);
        }
        section[data-testid="stSidebar"] [data-testid="stForm"] {
            background: rgba(21,26,33,0.96);
            border: 1px solid var(--gabrela-border);
            border-radius: 18px;
            padding: 14px 14px 10px 14px;
            box-shadow: 0 18px 34px rgba(0, 0, 0, 0.28);
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: linear-gradient(180deg, rgba(17,22,29,0.98), rgba(13,18,24,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 16px 28px rgba(0, 0, 0, 0.24);
            margin-bottom: 12px;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
            background: linear-gradient(135deg, rgba(255,210,63,0.08), rgba(255,210,63,0.03));
            border-bottom: 1px solid rgba(255,255,255,0.06);
            padding-top: 2px;
            padding-bottom: 2px;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {
            color: #f7fbff !important;
            font-weight: 800 !important;
            letter-spacing: 0.01em;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary svg {
            fill: var(--gabrela-yellow) !important;
        }
        .gabrela-sidebar-flow {
            display: grid;
            gap: 10px;
            margin: 0 0 14px 0;
            padding: 0 2px;
        }
        .gabrela-sidebar-flow-item {
            display: grid;
            grid-template-columns: 28px 28px 1fr;
            align-items: start;
            gap: 10px;
            color: #d6dde7;
            font-size: 0.9rem;
            padding: 10px 12px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            transition: transform 0.14s ease, border-color 0.14s ease, background 0.14s ease;
        }
        .gabrela-sidebar-flow-item:hover {
            transform: translateY(-1px);
            border-color: rgba(255,210,63,0.18);
            background: rgba(255,255,255,0.05);
        }
        .gabrela-sidebar-flow-item.is-active {
            background: linear-gradient(135deg, rgba(31,168,102,0.16), rgba(31,168,102,0.06));
            border-color: rgba(31,168,102,0.34);
            box-shadow: inset 3px 0 0 rgba(31,168,102,0.95);
        }
        .gabrela-sidebar-flow-item.is-active .gabrela-sidebar-flow-title {
            color: #ffffff;
        }
        .gabrela-sidebar-flow-index {
            width: 24px;
            height: 24px;
            border-radius: 999px;
            background: rgba(255,210,63,0.12);
            border: 1px solid rgba(255,210,63,0.22);
            color: var(--gabrela-yellow);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            flex: 0 0 auto;
        }
        .gabrela-sidebar-flow-icon {
            width: 24px;
            height: 24px;
            border-radius: 8px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            color: #eef4fb;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.90rem;
            line-height: 1;
            flex: 0 0 auto;
        }
        .gabrela-sidebar-flow-item.is-active .gabrela-sidebar-flow-icon {
            background: rgba(31,168,102,0.22);
            border-color: rgba(31,168,102,0.38);
            color: #ffffff;
        }
        .gabrela-sidebar-flow-copy {
            min-width: 0;
        }
        .gabrela-sidebar-flow-title {
            color: #f7fbff;
            font-weight: 700;
            margin-bottom: 2px;
        }
        .gabrela-sidebar-flow-sub {
            color: var(--gabrela-muted);
            font-size: 0.8rem;
            line-height: 1.35;
        }
        .gabrela-sidebar-mini-note {
            color: var(--gabrela-muted);
            font-size: 0.82rem;
            margin: 2px 0 10px 0;
        }
        .gabrela-sidebar-brand {
            background:
                linear-gradient(145deg, rgba(255,210,63,0.08), rgba(255,210,63,0.03)),
                linear-gradient(145deg, rgba(14,18,24,0.98), rgba(20,26,34,0.98));
            border-radius: 22px;
            color: white;
            padding: 14px 14px 12px 14px;
            margin: 4px 0 14px 0;
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.34);
            border: 1px solid rgba(255,210,63,0.14);
        }
        .gabrela-sidebar-brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }
        .gabrela-sidebar-brand-badge {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: linear-gradient(145deg, rgba(255,210,63,0.95), rgba(250,187,33,0.82));
            color: #11161d;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            font-weight: 900;
            box-shadow: 0 12px 24px rgba(255,210,63,0.18);
            flex: 0 0 auto;
        }
        .gabrela-sidebar-brand-copy {
            min-width: 0;
        }
        .gabrela-sidebar-brand-title {
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
            color: var(--gabrela-yellow);
        }
        .gabrela-sidebar-brand-sub {
            font-size: 0.88rem;
            color: #d6dde7;
        }
        .gabrela-sidebar-brand-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding-top: 10px;
            margin-top: 8px;
            border-top: 1px solid rgba(255,255,255,0.06);
            color: #cfe6d8;
            font-size: 0.82rem;
        }
        .gabrela-sidebar-status-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #21d07a;
            box-shadow: 0 0 0 4px rgba(33,208,122,0.12);
            flex: 0 0 auto;
        }
        .gabrela-sidebar-section {
            margin: 10px 0 8px 0;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255,210,63,0.1);
            color: var(--gabrela-yellow);
            border: 1px solid rgba(255,210,63,0.18);
            font-weight: 800;
            letter-spacing: 0.01em;
        }
        .gabrela-hero {
            background:
                radial-gradient(circle at top right, rgba(255,210,63,0.18), rgba(255,210,63,0) 30%),
                linear-gradient(135deg, rgba(13,17,23,0.98), rgba(19,24,31,0.98));
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 24px;
            padding: 22px 24px;
            margin: 2px 0 16px 0;
            box-shadow: 0 22px 44px rgba(0, 0, 0, 0.36);
            color: white;
        }
        .gabrela-hero-title {
            font-size: 2.05rem;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: 0.02em;
            margin-bottom: 6px;
            color: #ffffff;
        }
        .gabrela-hero-subtitle {
            font-size: 1rem;
            color: #d1d9e6;
            margin-bottom: 16px;
        }
        .gabrela-hero-chip-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .gabrela-chip {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,210,63,0.16);
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 0.9rem;
            color: #f5f8fb;
        }
        .gabrela-kpi-card {
            background: linear-gradient(180deg, rgba(23,29,37,0.98), rgba(18,24,32,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 14px 16px;
            min-height: 118px;
            box-shadow: 0 14px 24px rgba(0, 0, 0, 0.22);
            position: relative;
            overflow: hidden;
        }
        .gabrela-kpi-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, rgba(255,210,63,0.92), rgba(31,168,102,0.92));
        }
        .gabrela-kpi-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 10px;
        }
        .gabrela-kpi-label {
            color: #92a0ae;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .gabrela-kpi-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            color: var(--gabrela-yellow);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 800;
            flex: 0 0 auto;
        }
        .gabrela-kpi-value {
            color: #ffffff;
            font-size: 1.8rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .gabrela-kpi-sub {
            color: #d4dde8;
            font-size: 0.86rem;
        }
        .gabrela-kpi-accent {
            color: #2ee47e;
            font-weight: 700;
        }
        .gabrela-panel-card {
            background: linear-gradient(180deg, rgba(23,29,37,0.98), rgba(18,24,32,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 16px 18px;
            box-shadow: 0 14px 24px rgba(0, 0, 0, 0.22);
            margin-bottom: 14px;
        }
        .gabrela-panel-title {
            color: #f6f8fb;
            font-size: 0.96rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .gabrela-section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 6px 0 12px 0;
            color: #f6f8fb;
            font-size: 1.02rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .gabrela-section-title-badge {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            background: rgba(255,210,63,0.1);
            border: 1px solid rgba(255,210,63,0.16);
            color: var(--gabrela-yellow);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 900;
            flex: 0 0 auto;
        }
        .gabrela-panel-subtitle {
            color: #93a0ae;
            font-size: 0.88rem;
            margin-bottom: 12px;
        }
        .gabrela-note-list {
            display: grid;
            gap: 10px;
        }
        .gabrela-note-item {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 12px 14px;
        }
        .gabrela-note-item b {
            color: var(--gabrela-yellow);
        }
        .gabrela-inline-stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 8px;
        }
        .gabrela-inline-stat {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 12px 14px;
        }
        .gabrela-inline-stat-label {
            color: #92a0ae;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }
        .gabrela-inline-stat-value {
            color: #ffffff;
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.05;
        }
        .gabrela-dashboard-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(300px, 0.75fr);
            gap: 18px;
            align-items: start;
        }
        .gabrela-stack {
            display: grid;
            gap: 14px;
        }
        .gabrela-list-card {
            background: linear-gradient(180deg, rgba(23,29,37,0.98), rgba(18,24,32,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 16px 28px rgba(0, 0, 0, 0.22);
        }
        .gabrela-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .gabrela-pill {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            color: #e8eef5;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.82rem;
        }
        .gabrela-bulletin-board {
            display: grid;
            gap: 10px;
        }
        .gabrela-bulletin-row {
            display: grid;
            grid-template-columns: 64px 78px minmax(0, 1.5fr) minmax(0, 1fr) 78px 78px;
            gap: 8px;
            align-items: center;
            padding: 10px 12px;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
            border: 1px solid rgba(255,255,255,0.07);
        }
        .gabrela-bulletin-row.gabrela-bulletin-head {
            padding-top: 8px;
            padding-bottom: 8px;
            background: transparent;
            border: 0;
            color: var(--gabrela-muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .gabrela-bulletin-time {
            color: #f7fbff;
            font-weight: 700;
            font-size: 0.84rem;
        }
        .gabrela-bulletin-mbs {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 56px;
            padding: 5px 7px;
            border-radius: 10px;
            background: rgba(255,210,63,0.10);
            border: 1px solid rgba(255,210,63,0.18);
            color: var(--gabrela-yellow);
            font-size: 0.71rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }
        .gabrela-bulletin-match {
            min-width: 0;
        }
        .gabrela-bulletin-match-title {
            color: #ffffff;
            font-weight: 700;
            font-size: 0.92rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .gabrela-bulletin-match-sub {
            color: var(--gabrela-muted);
            font-size: 0.74rem;
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .gabrela-market-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 4px 9px;
            border-radius: 999px;
            background: rgba(255,210,63,0.1);
            border: 1px solid rgba(255,210,63,0.2);
            color: var(--gabrela-yellow);
            font-size: 0.75rem;
            font-weight: 800;
        }
        .gabrela-market-chip.inactive {
            background: rgba(148,163,184,0.10);
            border: 1px solid rgba(148,163,184,0.18);
            color: #7f8b99;
        }
        .gabrela-favorite-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 4px 9px;
            border-radius: 999px;
            background: rgba(22,163,74,0.14);
            border: 1px solid rgba(74,222,128,0.28);
            color: #86efac;
            font-size: 0.72rem;
            font-weight: 800;
            margin-left: 6px;
        }
        .gabrela-favorite-chip.medium {
            background: rgba(59,130,246,0.12);
            border-color: rgba(96,165,250,0.22);
            color: #93c5fd;
        }
        .gabrela-bulletin-number {
            color: #ffffff;
            font-weight: 800;
            font-size: 0.88rem;
        }
        .gabrela-bulletin-meta {
            color: var(--gabrela-muted);
            font-size: 0.70rem;
            margin-top: 2px;
        }
        .gabrela-match-board {
            display: grid;
            gap: 8px;
        }
        .gabrela-match-head,
        .gabrela-match-row {
            display: grid;
            grid-template-columns: 88px 82px minmax(0, 2.4fr) 76px repeat(5, 76px);
            gap: 8px;
            align-items: center;
            padding: 10px 12px;
            border-radius: 16px;
        }
        .gabrela-match-head {
            background: transparent;
            border: 0;
            color: var(--gabrela-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
        }
        .gabrela-match-row {
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
            border: 1px solid rgba(255,255,255,0.07);
        }
        .gabrela-match-time {
            color: #f7fbff;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .gabrela-match-status {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 68px;
            padding: 5px 7px;
            border-radius: 10px;
            background: rgba(255,210,63,0.10);
            border: 1px solid rgba(255,210,63,0.18);
            color: var(--gabrela-yellow);
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.05em;
        }
        .gabrela-match-title-wrap {
            min-width: 0;
        }
        .gabrela-match-main {
            color: #ffffff;
            font-weight: 800;
            font-size: 0.90rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .gabrela-match-sub,
        .gabrela-match-source {
            color: var(--gabrela-muted);
            font-size: 0.72rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .gabrela-match-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 6px;
        }
        .gabrela-match-odd-cell {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 50px;
            padding: 7px 6px;
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
        }
        .gabrela-match-odd-cell span {
            color: var(--gabrela-muted);
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .gabrela-match-odd-cell strong {
            color: #ffffff;
            margin-top: 3px;
            font-size: 0.84rem;
        }
        div[data-baseweb="tab-list"] {
            gap: 10px;
            padding-bottom: 8px;
        }
        button[data-baseweb="tab"] {
            background: linear-gradient(180deg, #181e27, #131921);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 999px !important;
            color: #eef4fb;
            padding: 10px 16px !important;
            min-height: 44px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
            transition: all 0.18s ease;
        }
        button[data-baseweb="tab"]:hover {
            border-color: rgba(255,210,63,0.28);
            color: var(--gabrela-yellow);
            transform: translateY(-1px);
        }
        button[aria-selected="true"][data-baseweb="tab"] {
            background: linear-gradient(135deg, #1a8a53, #0f6e42) !important;
            color: white !important;
            border-color: rgba(255,210,63,0.34) !important;
            box-shadow: 0 12px 28px rgba(10, 91, 54, 0.32);
        }
        @media (max-width: 1100px) {
            .gabrela-dashboard-grid {
                grid-template-columns: 1fr;
            }
            .gabrela-bulletin-row {
                grid-template-columns: 1fr;
            }
            .gabrela-match-head,
            .gabrela-match-row {
                grid-template-columns: 1fr;
            }
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(23,29,37,0.98), rgba(18,24,32,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 12px 14px;
            box-shadow: 0 16px 28px rgba(0, 0, 0, 0.24);
        }
        div[data-testid="stMetricLabel"] {
            color: #9aa6b2 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fbff;
        }
        div[data-testid="stExpander"] {
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(20,26,34,0.92);
            color: #eef4fb;
        }
        .stMarkdown, .stText, p, label, div {
            color: #eef4fb;
        }
        pre, code {
            background: #141a22 !important;
            color: #dbe7f3 !important;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, #151b23, #10161d);
            color: #eef4fb;
            font-weight: 700;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.24);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(255,210,63,0.28);
            color: var(--gabrela-yellow);
        }
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #1a8a53, #0f6e42) !important;
            color: white !important;
            border-color: rgba(255,210,63,0.22) !important;
        }
        .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(10, 91, 54, 0.28);
        }
        .stSelectbox [data-baseweb="select"],
        .stMultiSelect [data-baseweb="select"],
        .stNumberInput input,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"],
        section[data-testid="stSidebar"] .stNumberInput input {
            border-radius: 14px !important;
        }
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        .stSelectbox [data-baseweb="select"] input,
        .stMultiSelect [data-baseweb="select"] input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] input,
        section[data-testid="stSidebar"] .stNumberInput input {
            background: #f8fafc !important;
            color: #12263a !important;
            -webkit-text-fill-color: #12263a !important;
        }
        .stSelectbox [data-baseweb="select"] span,
        .stMultiSelect [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] svg,
        .stMultiSelect [data-baseweb="select"] svg,
        .stSelectbox [data-baseweb="select"] div,
        .stMultiSelect [data-baseweb="select"] div,
        .stSelectbox [data-baseweb="select"] p,
        .stMultiSelect [data-baseweb="select"] p,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] svg {
            color: #12263a !important;
            fill: #12263a !important;
            -webkit-text-fill-color: #12263a !important;
        }
        .stSelectbox [data-baseweb="select"] *,
        .stMultiSelect [data-baseweb="select"] *,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] *,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] * {
            color: #12263a !important;
            fill: #12263a !important;
            -webkit-text-fill-color: #12263a !important;
            opacity: 1 !important;
        }
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        .stSelectbox [data-baseweb="select"] > div > div,
        .stMultiSelect [data-baseweb="select"] > div > div,
        .stSelectbox [data-baseweb="select"] input,
        .stMultiSelect [data-baseweb="select"] input,
        .stSelectbox [data-baseweb="select"] input[readonly],
        .stMultiSelect [data-baseweb="select"] input[readonly] {
            color: #12263a !important;
            background: #f8fafc !important;
            -webkit-text-fill-color: #12263a !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }
        .stSelectbox [data-baseweb="select"] input::placeholder,
        .stMultiSelect [data-baseweb="select"] input::placeholder {
            color: #5b6672 !important;
            -webkit-text-fill-color: #5b6672 !important;
            opacity: 1 !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        div[data-baseweb="select"] + div {
            color: #12263a !important;
        }
        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        div[data-baseweb="select"] + div * {
            color: #12263a !important;
            fill: #12263a !important;
            -webkit-text-fill-color: #12263a !important;
            opacity: 1 !important;
        }
        div[role="listbox"],
        ul[role="listbox"] {
            background: #f8fafc !important;
            color: #12263a !important;
            border: 1px solid rgba(18,38,58,0.14) !important;
        }
        div[role="option"],
        li[role="option"] {
            background: #f8fafc !important;
            color: #12263a !important;
            -webkit-text-fill-color: #12263a !important;
        }
        div[role="option"]:hover,
        li[role="option"]:hover {
            background: #e9f5ee !important;
            color: #0f6e42 !important;
        }
        div[role="option"] *,
        li[role="option"] *,
        div[role="listbox"] *,
        ul[role="listbox"] * {
            color: #12263a !important;
            -webkit-text-fill-color: #12263a !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: rgba(18,24,32,0.96);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            overflow: hidden;
        }
        .stAlert {
            border-radius: 16px;
        }
        .gabrela-shell-note {
            color: var(--gabrela-muted);
            font-size: 0.9rem;
        }
        .gabrela-activity-chip {
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:8px 12px;
            border-radius:999px;
            background:rgba(31,168,102,0.16);
            border:1px solid rgba(31,168,102,0.34);
            color:#effcf4;
            font-size:0.84rem;
            font-weight:700;
        }
        .gabrela-activity-dot {
            width:10px;
            height:10px;
            border-radius:999px;
            background:#32d27a;
            box-shadow:0 0 0 0 rgba(50,210,122,0.8);
            animation: gabrelaPulse 1.4s ease-in-out infinite;
        }
        @keyframes gabrelaPulse {
            0% { box-shadow:0 0 0 0 rgba(50,210,122,0.7); }
            70% { box-shadow:0 0 0 10px rgba(50,210,122,0); }
            100% { box-shadow:0 0 0 0 rgba(50,210,122,0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    current_local = datetime.now(APP_TIMEZONE).strftime("%d.%m.%Y %H:%M")
    activity = st.session_state.get("runtime_activity") or {}
    activity_chip = ""
    if activity.get("state") == "running":
        activity_chip = (
            "<div class='gabrela-activity-chip'>"
            "<span class='gabrela-activity-dot'></span>"
            f"<span>{html_lib.escape(str(activity.get('kind', 'Sistem')))} aktif: "
            f"{html_lib.escape(str(activity.get('text', 'islem suruyor')))}</span>"
            "</div>"
        )
    st.markdown(
        textwrap.dedent(
            f"""
        <div class="gabrela-hero">
            <div class="gabrela-hero-title">{BRAND_TITLE}</div>
            <div class="gabrela-hero-subtitle">{BRAND_SUBTITLE}</div>
            <div class="gabrela-hero-chip-row">
                <div class="gabrela-chip">TR Saati: {current_local}</div>
                <div class="gabrela-chip">iddaa odakli fiyatlama</div>
                <div class="gabrela-chip">Kupon ve sonuc takibi</div>
                {activity_chip}
            </div>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_sidebar_brand_card() -> None:
    st.markdown(
        textwrap.dedent(
            """
        <div class="gabrela-sidebar-brand">
            <div class="gabrela-sidebar-brand-title">GABRELA</div>
            <div class="gabrela-sidebar-brand-sub">
                iddaa odakli tahmin, kupon ve performans paneli
            </div>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_sidebar_section(label: str) -> None:
    st.markdown(
        f"<div class='gabrela-sidebar-section'>{html_lib.escape(label)}</div>",
        unsafe_allow_html=True,
    )


def render_sidebar_flow() -> None:
    st.markdown(
        textwrap.dedent(
            """
        <div class="gabrela-sidebar-flow">
            <div class="gabrela-sidebar-flow-item is-active">
                <span class="gabrela-sidebar-flow-index">1</span>
                <span class="gabrela-sidebar-flow-icon">⌂</span>
                <div class="gabrela-sidebar-flow-copy">
                    <div class="gabrela-sidebar-flow-title">Kontrol Merkezi</div>
                    <div class="gabrela-sidebar-flow-sub">Kupon, bankroll ve risk tercihlerini ayarla.</div>
                </div>
            </div>
            <div class="gabrela-sidebar-flow-item">
                <span class="gabrela-sidebar-flow-index">2</span>
                <span class="gabrela-sidebar-flow-icon">↻</span>
                <div class="gabrela-sidebar-flow-copy">
                    <div class="gabrela-sidebar-flow-title">Veri Akisi</div>
                    <div class="gabrela-sidebar-flow-sub">iddaa veri guncellemesi ve veri temizligi buradan ilerler.</div>
                </div>
            </div>
            <div class="gabrela-sidebar-flow-item">
                <span class="gabrela-sidebar-flow-index">3</span>
                <span class="gabrela-sidebar-flow-icon">⚙</span>
                <div class="gabrela-sidebar-flow-copy">
                    <div class="gabrela-sidebar-flow-title">Tahmin Motoru</div>
                    <div class="gabrela-sidebar-flow-sub">Modeli egit, tahmin uret ve kupon akisini baslat.</div>
                </div>
            </div>
            <div class="gabrela-sidebar-flow-item">
                <span class="gabrela-sidebar-flow-index">4</span>
                <span class="gabrela-sidebar-flow-icon">▣</span>
                <div class="gabrela-sidebar-flow-copy">
                    <div class="gabrela-sidebar-flow-title">Kupon ve Takip</div>
                    <div class="gabrela-sidebar-flow-sub">Bugunun kuponu, manuel secim ve kupon performansi burada.</div>
                </div>
            </div>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_sidebar_brand_shell() -> None:
    activity = st.session_state.get("runtime_activity") or {}
    status_line = (
        "<span class='gabrela-sidebar-status-dot'></span><span>Sistem durumu: cevrimici</span>"
        if activity.get("state") != "running"
        else
        "<span class='gabrela-sidebar-status-dot'></span>"
        f"<span>{html_lib.escape(str(activity.get('kind', 'Sistem')))} calisiyor</span>"
    )
    st.markdown(
        textwrap.dedent(
            f"""
        <div class="gabrela-sidebar-brand">
            <div class="gabrela-sidebar-brand-row">
                <div class="gabrela-sidebar-brand-badge">G</div>
                <div class="gabrela-sidebar-brand-copy">
                    <div class="gabrela-sidebar-brand-title">GABRELA</div>
                    <div class="gabrela-sidebar-brand-sub">
                        iddaa odakli tahmin, kupon ve performans paneli
                    </div>
                </div>
            </div>
            <div class="gabrela-sidebar-brand-status">
                {status_line}
            </div>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def metric_text(value, kind: str = "number") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if kind == "pct":
        return f"{float(value):.2%}"
    if kind == "float":
        return f"{float(value):.3f}"
    if kind == "int":
        try:
            return f"{int(value):,}".replace(",", ".")
        except Exception:
            return str(value)
    return str(value)


@st.cache_data(show_spinner=False, ttl=60)
def load_matches_dataframe(status_filter=None) -> pd.DataFrame:
    return _load_matches_dataframe_impl(status_filter=status_filter, limit=None, include_odds=True)


@st.cache_data(show_spinner=False, ttl=60)
def load_match_counts() -> dict:
    session = get_session()
    try:
        total = session.query(Match.id).count()
        completed = session.query(Match.id).filter(Match.status == "completed").count()
        upcoming = (
            session.query(Match.id)
            .filter(Match.status.in_(["scheduled", "live"]))
            .count()
        )
        live = session.query(Match.id).filter(Match.status == "live").count()
        return {
            "total": int(total or 0),
            "completed": int(completed or 0),
            "upcoming": int(upcoming or 0),
            "live": int(live or 0),
        }
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=60)
def load_match_center_preview(status_filter=None, limit: int = 500) -> pd.DataFrame:
    return _load_matches_dataframe_impl(status_filter=status_filter, limit=limit, include_odds=True)


def _load_matches_dataframe_impl(status_filter=None, limit: int | None = None, include_odds: bool = True) -> pd.DataFrame:
    session = get_session()
    try:
        query = (
            session.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
            )
            .order_by(Match.kickoff_time.desc())
        )
        if include_odds:
            query = query.options(joinedload(Match.odds_snapshots))

        if status_filter:
            query = query.filter(Match.status.in_(status_filter))
        if limit:
            query = query.limit(int(limit))

        matches = query.all()
        preferred_odds_lookup = build_preferred_odds_lookup(matches)
        rows = []
        for match in matches:
            odds = preferred_odds_lookup.get(match.id) or {}
            rows.append(
                {
                    "match_id": match.id,
                    "kickoff_time": match.kickoff_time,
                    "league": match.league.name if match.league else None,
                    "home_team": match.home_team.name,
                    "away_team": match.away_team.name,
                    "status": match.status,
                    "source": match.source,
                    "odds_source": odds.get("odds_source"),
                    "score": (
                        f"{match.home_score}-{match.away_score}"
                        if match.home_score is not None and match.away_score is not None
                        else None
                    ),
                    "home_odds": odds.get("home_win_odds"),
                    "draw_odds": odds.get("draw_odds"),
                    "away_odds": odds.get("away_win_odds"),
                    "over_25_odds": odds.get("over_25_odds"),
                    "under_25_odds": odds.get("under_25_odds"),
                    "double_chance_1x_odds": odds.get("double_chance_1x_odds"),
                    "double_chance_12_odds": odds.get("double_chance_12_odds"),
                    "double_chance_x2_odds": odds.get("double_chance_x2_odds"),
                    "handicap_home_10_odds": odds.get("handicap_home_10_odds"),
                    "handicap_draw_10_odds": odds.get("handicap_draw_10_odds"),
                    "handicap_away_10_odds": odds.get("handicap_away_10_odds"),
                }
            )
        return pd.DataFrame(rows)
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=30)
def load_latest_predictions() -> pd.DataFrame:
    session = get_session()
    try:
        predictions = (
            session.query(Prediction)
            .options(
                joinedload(Prediction.match).joinedload(Match.home_team),
                joinedload(Prediction.match).joinedload(Match.away_team),
                joinedload(Prediction.match).joinedload(Match.league),
            )
            .order_by(Prediction.created_at.desc())
            .limit(100)
            .all()
        )

        rows = []
        for prediction in predictions:
            match = prediction.match
            rows.append(
                {
                    "created_at": prediction.created_at,
                    "kickoff_time": match.kickoff_time if match else None,
                    "league": match.league.name if match and match.league else None,
                    "home_team": match.home_team.name if match and match.home_team else None,
                    "away_team": match.away_team.name if match and match.away_team else None,
                    "prob_home": round(prediction.prob_home, 4),
                    "prob_draw": round(prediction.prob_draw, 4),
                    "prob_away": round(prediction.prob_away, 4),
                    "predicted_label": prediction.predicted_label,
                    "confidence": round(prediction.confidence, 4),
                    "edge_home": round(prediction.edge_home, 4)
                    if prediction.edge_home is not None
                    else None,
                    "edge_draw": round(prediction.edge_draw, 4)
                    if prediction.edge_draw is not None
                    else None,
                    "edge_away": round(prediction.edge_away, 4)
                    if prediction.edge_away is not None
                    else None,
                    "model_name": prediction.model_name,
                }
            )

        return pd.DataFrame(rows)
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=60)
def load_source_summary() -> pd.DataFrame:
    session = get_session()
    try:
        rows = []
        for source, status, count in (
            session.query(Match.source, Match.status, Match.id)
            .order_by(Match.source.asc(), Match.status.asc())
            .all()
        ):
            rows.append({"source": source, "status": status, "match_id": count})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return (
            df.groupby(["source", "status"], as_index=False)["match_id"]
            .count()
            .rename(columns={"match_id": "match_count"})
        )
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=60)
def load_team_alias_dataframe() -> pd.DataFrame:
    session = get_session()
    try:
        return pd.DataFrame(list_team_aliases(session))
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=300)
def load_team_options() -> list[dict]:
    session = get_session()
    try:
        return list_teams(session)
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=15)
def load_saved_coupons_dataframe() -> pd.DataFrame:
    session = get_session()
    try:
        return pd.DataFrame(list_saved_coupons(session))
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=15)
def load_saved_coupon_items(coupon_id: int) -> tuple[dict | None, pd.DataFrame]:
    session = get_session()
    try:
        coupon = get_saved_coupon_with_items(session, int(coupon_id))
        if coupon is None:
            return None, pd.DataFrame()

        header = {
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
            "is_manual": coupon.is_manual,
            "notes": coupon.notes,
            "created_at": coupon.created_at,
            "updated_at": coupon.updated_at,
        }
        items = []
        for item in coupon.items:
            items.append(
                {
                    "selection_order": item.selection_order,
                    "kickoff_time": item.kickoff_time,
                    "league_name": item.league_name,
                    "home_team": item.home_team,
                    "away_team": item.away_team,
                    "recommended_label": item.recommended_label,
                    "recommended_market": item.recommended_market,
                    "recommended_odds": item.recommended_odds,
                    "recommended_probability": item.recommended_probability,
                    "recommended_edge": item.recommended_edge,
                    "flat_stake": item.flat_stake,
                    "actual_odds": item.actual_odds,
                    "source": item.source,
                    "odds_source": item.odds_source,
                    "result_status": item.result_status,
                }
            )
        return header, pd.DataFrame(items)
    finally:
        session.close()


def save_coupon_to_library(
    picks: pd.DataFrame,
    summary: dict,
    segment: str,
    source_mode: str,
    actual_total_odds: float | None = None,
    actual_total_stake: float | None = None,
    is_manual: bool = False,
    notes: str | None = None,
) -> tuple[bool, str]:
    if picks is None or picks.empty:
        return False, "Kaydedilecek kupon bulunamadi."

    now_text = datetime.now(APP_TIMEZONE).strftime("%d.%m %H:%M")
    coupon_name = f"{segment} kuponu {now_text}"
    session = get_session()
    try:
        create_saved_coupon(
            session=session,
            name=coupon_name,
            picks=picks,
            summary=summary or {},
            segment=segment,
            source_mode=source_mode,
            actual_total_odds=actual_total_odds,
            actual_total_stake=actual_total_stake,
            is_manual=is_manual,
            notes=notes,
        )
        session.commit()
        clear_runtime_caches()
        return True, f"Kupon kaydedildi: {coupon_name}"
    except Exception as exc:
        session.rollback()
        return False, f"Kupon kaydedilemedi: {exc}"
    finally:
        session.close()


def update_coupon_status_in_library(coupon_id: int, status: str) -> tuple[bool, str]:
    session = get_session()
    try:
        coupon = update_saved_coupon_status(session, coupon_id=int(coupon_id), status=status)
        if coupon is None:
            session.rollback()
            return False, "Kupon bulunamadi."
        session.commit()
        clear_runtime_caches()
        return True, f"Kupon durumu guncellendi: {status}"
    except Exception as exc:
        session.rollback()
        return False, f"Kupon durumu guncellenemedi: {exc}"
    finally:
        session.close()


def settle_coupon_in_library(coupon_id: int) -> tuple[bool, str]:
    session = get_session()
    try:
        coupon = settle_saved_coupon(session, coupon_id=int(coupon_id))
        if coupon is None:
            session.rollback()
            return False, "Kupon bulunamadi."
        session.commit()
        clear_runtime_caches()
        status_text = str(coupon.status or "open").upper()
        return True, f"Kupon mac sonuclarina gore guncellendi: {status_text}"
    except Exception as exc:
        session.rollback()
        return False, f"Kupon sonucu islenemedi: {exc}"
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=15)
def load_saved_coupon_performance() -> dict:
    session = get_session()
    try:
        return summarize_saved_coupon_performance(session)
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=15)
def load_saved_coupon_preferences() -> dict:
    session = get_session()
    try:
        return summarize_saved_coupon_preferences(session)
    finally:
        session.close()


def clear_runtime_caches() -> None:
    for cached_fn in [
        load_matches_dataframe,
        load_match_counts,
        load_match_center_preview,
        load_latest_predictions,
        load_source_summary,
        load_team_alias_dataframe,
        load_team_options,
        load_saved_coupons_dataframe,
        load_saved_coupon_items,
        load_saved_coupon_performance,
        load_saved_coupon_preferences,
    ]:
        try:
            cached_fn.clear()
        except Exception:
            pass


def build_manual_coupon_label(row: pd.Series | dict, idx: int) -> str:
    row_obj = row if isinstance(row, dict) else row.to_dict()
    return (
        f"{idx}. {format_kickoff(row_obj.get('kickoff_time'))} | "
        f"{row_obj.get('home_team', '')} - {row_obj.get('away_team', '')} | "
        f"{row_obj.get('recommended_market', '')} | "
        f"{number_text(row_obj.get('recommended_odds'))}"
    )


def build_manual_coupon_options(picks: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
    if picks is None or picks.empty:
        return {}, pd.DataFrame()
    working = picks.copy()
    working["kickoff_time"] = pd.to_datetime(working["kickoff_time"], errors="coerce", utc=True)
    labels: dict[str, dict] = {}
    for idx, (_, row) in enumerate(working.iterrows(), start=1):
        label = build_manual_coupon_label(row, idx)
        labels[label] = row.to_dict()
    return labels, working


def render_period_performance(title: str, period_df: pd.DataFrame) -> None:
    st.write(title)
    if not isinstance(period_df, pd.DataFrame) or period_df.empty:
        st.info("Bu gorunum icin yeterli veri yok.")
        return

    display_df = period_df.copy().sort_values("period_start")
    render_table(display_df, height=260)

    roi_chart = display_df.set_index("period_start")["roi"]
    pnl_chart = display_df.set_index("period_start")["pnl"]
    render_bar_chart_dark(pnl_chart, height=220, color="#1fa866")
    render_line_chart_dark(roi_chart, height=220, color="#69b0ff")


def pct_text(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2%}"


def number_text(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def render_table(df: pd.DataFrame, height: int = 320) -> None:
    view = df.copy()
    for column in view.columns:
        name = str(column).lower()
        if any(token in name for token in ["kickoff_time", "created_at", "updated_at", "period_start", "fetched_at"]):
            parsed = pd.to_datetime(view[column], errors="coerce", utc=True, dayfirst=True)
            if not parsed.isna().all():
                view[column] = parsed.apply(
                    lambda value: "-"
                    if pd.isna(value)
                    else value.tz_convert(APP_TIMEZONE).strftime("%d.%m.%Y %H:%M")
                )
    styler = (
        view.style.hide(axis="index")
        .set_properties(
            **{
                "background-color": "#121820",
                "color": "#eef4fb",
                "border-color": "rgba(255,255,255,0.06)",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#171e27"),
                        ("color", "#9fb0c2"),
                        ("border-color", "rgba(255,255,255,0.06)"),
                        ("font-weight", "700"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("background-color", "#121820"),
                        ("color", "#eef4fb"),
                        ("border-color", "rgba(255,255,255,0.05)"),
                    ],
                },
            ]
        )
    )
    st.dataframe(
        styler,
        width="stretch",
        height=height,
    )


def _prepare_chart_frame(series_or_df) -> tuple[pd.DataFrame, str, str, str | None]:
    if isinstance(series_or_df, pd.Series):
        frame = series_or_df.reset_index()
        frame.columns = ["x_value", "metric_value"]
        return frame, "x_value", "metric_value", None
    if isinstance(series_or_df, pd.DataFrame):
        frame = series_or_df.copy()
        if frame.index.name or not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index()
        value_cols = [col for col in frame.columns if col != frame.columns[0]]
        if len(value_cols) == 1:
            return frame.rename(columns={frame.columns[0]: "x_value", value_cols[0]: "metric_value"}), "x_value", "metric_value", None
        melted = frame.rename(columns={frame.columns[0]: "x_value"}).melt(
            id_vars=["x_value"],
            var_name="series_name",
            value_name="metric_value",
        )
        return melted, "x_value", "metric_value", "series_name"
    return pd.DataFrame(), "x_value", "metric_value", None


def render_line_chart_dark(series_or_df, height: int = 260, color: str = "#69b0ff") -> None:
    frame, x_col, y_col, color_col = _prepare_chart_frame(series_or_df)
    if frame.empty:
        st.info("Bu grafik icin veri yok.")
        return
    base = alt.Chart(frame).mark_line(strokeWidth=2.5).encode(
        x=alt.X(x_col, title=None),
        y=alt.Y(y_col, title=None),
    )
    if color_col:
        base = base.encode(color=alt.Color(color_col, legend=alt.Legend(title=None)))
    else:
        base = base.encode(color=alt.value(color))
    chart = (
        base.properties(height=height)
        .configure(background="#121820")
        .configure_view(stroke=None, fill="#121820")
        .configure_axis(
            labelColor="#b9c7d6",
            titleColor="#8fa0b3",
            gridColor="#253040",
            domainColor="#253040",
            tickColor="#253040",
        )
        .configure_legend(labelColor="#d7e3ef", titleColor="#8fa0b3")
    )
    st.altair_chart(chart, use_container_width=True)


def render_bar_chart_dark(series_or_df, height: int = 240, color: str = "#1fa866") -> None:
    frame, x_col, y_col, _ = _prepare_chart_frame(series_or_df)
    if frame.empty:
        st.info("Bu grafik icin veri yok.")
        return
    chart = (
        alt.Chart(frame)
        .mark_bar(color=color)
        .encode(
            x=alt.X(x_col, title=None),
            y=alt.Y(y_col, title=None),
        )
        .properties(height=height)
        .configure(background="#121820")
        .configure_view(stroke=None, fill="#121820")
        .configure_axis(
            labelColor="#b9c7d6",
            titleColor="#8fa0b3",
            gridColor="#253040",
            domainColor="#253040",
            tickColor="#253040",
        )
    )
    st.altair_chart(chart, use_container_width=True)


def format_kickoff(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return "-"
    return ts.tz_convert(APP_TIMEZONE).strftime("%d.%m %H:%M")


def local_calendar_date(value):
    if value is None or pd.isna(value):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_convert(APP_TIMEZONE).date()


def local_timestamp(value):
    if value is None or pd.isna(value):
        return pd.NaT
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(APP_TIMEZONE)


def format_local_datetime_series(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype="object")
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    formatted = ts.apply(
        lambda value: "-"
        if pd.isna(value)
        else value.tz_convert(APP_TIMEZONE).strftime("%d.%m.%Y %H:%M")
    )
    return formatted.astype("object")


def build_match_driver_view(
    row: pd.Series,
    global_importance: pd.DataFrame | None = None,
    limit: int = 6,
) -> pd.DataFrame:
    candidate_features = [
        "elo_diff",
        "form_diff",
        "rest_diff",
        "imp_home",
        "imp_draw",
        "imp_away",
        "market_overround",
        "league_quality_index",
        "home_points_last5",
        "away_points_last5",
    ]

    ranked_features = candidate_features
    if isinstance(global_importance, pd.DataFrame) and not global_importance.empty:
        ordered = global_importance["feature"].astype(str).tolist()
        ranked_features = [feature for feature in ordered if feature in candidate_features]
        for feature in candidate_features:
            if feature not in ranked_features:
                ranked_features.append(feature)

    predicted_label = str(row.get("predicted_label", "")).upper()
    items = []
    for feature in ranked_features:
        value = row.get(feature)
        if value is None or pd.isna(value):
            continue

        comment = ""
        numeric_value = float(value)
        if feature == "elo_diff":
            if numeric_value > 0:
                comment = "Ev sahibi rating olarak onde."
            elif numeric_value < 0:
                comment = "Deplasman rating olarak onde."
            else:
                comment = "Takim ratingleri dengeli."
        elif feature == "form_diff":
            if numeric_value > 0:
                comment = "Ev sahibi son formda daha iyi."
            elif numeric_value < 0:
                comment = "Deplasman son formda daha iyi."
            else:
                comment = "Son form dengeli."
        elif feature == "rest_diff":
            if numeric_value > 0:
                comment = "Ev sahibi daha uzun dinlenmis."
            elif numeric_value < 0:
                comment = "Deplasman daha uzun dinlenmis."
            else:
                comment = "Dinlenme farki yok."
        elif feature == "imp_home":
            comment = "Piyasa ev sahibini boyle fiyatliyor."
        elif feature == "imp_draw":
            comment = "Piyasa beraberlik riskini boyle fiyatliyor."
        elif feature == "imp_away":
            comment = "Piyasa deplasmani boyle fiyatliyor."
        elif feature == "market_overround":
            comment = "Bookmaker marji / piyasa sikiligi."
        elif feature == "league_quality_index":
            comment = "Lig seviye agirligi."
        elif feature == "home_points_last5":
            comment = "Ev sahibi son 5 mac puan ortalamasi."
        elif feature == "away_points_last5":
            comment = "Deplasman son 5 mac puan ortalamasi."

        alignment = ""
        if predicted_label == "H" and feature in {"elo_diff", "form_diff", "rest_diff"}:
            alignment = "Tahmini destekliyor" if numeric_value > 0 else "Tahminle zayif uyumlu"
        elif predicted_label == "A" and feature in {"elo_diff", "form_diff", "rest_diff"}:
            alignment = "Tahmini destekliyor" if numeric_value < 0 else "Tahminle zayif uyumlu"
        elif predicted_label == "D" and feature in {"elo_diff", "form_diff"}:
            alignment = "Tahminle uyumlu" if abs(numeric_value) < 0.15 else "Tahminle zayif uyumlu"

        items.append(
            {
                "feature": MATCH_DRIVER_LABELS.get(feature, feature),
                "value": round(numeric_value, 4),
                "yorum": comment,
                "tahmin_uyumu": alignment,
            }
        )

        if len(items) >= limit:
            break

    return pd.DataFrame(items)


def build_error_analysis_views(predictions_df: pd.DataFrame) -> dict:
    if predictions_df is None or predictions_df.empty:
        return {}

    frame = predictions_df.copy()
    frame["incorrect"] = frame["predicted_label"] != frame["target_label"]
    frame["confidence_band"] = pd.cut(
        frame["confidence"],
        bins=[0.0, 0.45, 0.55, 0.65, 0.75, 1.01],
        labels=["0-45%", "45-55%", "55-65%", "65-75%", "75%+"],
        include_lowest=True,
    )

    league_summary = (
        frame.groupby("league_name", as_index=False)
        .agg(
            matches=("incorrect", "count"),
            errors=("incorrect", "sum"),
            avg_confidence=("confidence", "mean"),
        )
    )
    league_summary["error_rate"] = np.where(
        league_summary["matches"] > 0,
        league_summary["errors"] / league_summary["matches"],
        np.nan,
    )
    league_summary = league_summary.sort_values(
        ["error_rate", "matches"],
        ascending=[False, False],
    ).reset_index(drop=True)

    confidence_summary = (
        frame.groupby("confidence_band", as_index=False)
        .agg(
            matches=("incorrect", "count"),
            errors=("incorrect", "sum"),
            avg_confidence=("confidence", "mean"),
        )
    )
    confidence_summary["error_rate"] = np.where(
        confidence_summary["matches"] > 0,
        confidence_summary["errors"] / confidence_summary["matches"],
        np.nan,
    )

    label_summary = (
        frame.groupby("predicted_label", as_index=False)
        .agg(
            matches=("incorrect", "count"),
            errors=("incorrect", "sum"),
            avg_confidence=("confidence", "mean"),
        )
    )
    label_summary["error_rate"] = np.where(
        label_summary["matches"] > 0,
        label_summary["errors"] / label_summary["matches"],
        np.nan,
    )

    worst_rows = frame[frame["incorrect"]].copy().sort_values(
        ["confidence", "kickoff_time"],
        ascending=[False, False],
    )

    return {
        "league_summary": league_summary.head(20),
        "confidence_summary": confidence_summary,
        "label_summary": label_summary,
        "worst_rows": worst_rows.head(50),
    }


def market_family_from_row(row: pd.Series) -> str:
    label = str(row.get("recommended_label", "")).upper()
    market = str(row.get("recommended_market", "")).strip().lower()
    if label in {"H", "D", "A"}:
        return "1X2"
    if label in {"1X", "12", "X2"} or "cifte sans" in market:
        return "Cifte Sans"
    if label in {"H10H", "H10D", "H10A"} or "handikap" in market:
        return "Handikap"
    if label in {"O25", "U25"} or "2.5" in market or "alt" in market or "ust" in market:
        return "Alt/Ust"
    if label in {"BYES", "BNO"} or "kg" in market or "var" in market or "yok" in market:
        return "KG"
    return "Diger"


def build_market_mix_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    view = frame.copy()
    view["market_family"] = view.apply(market_family_from_row, axis=1)
    summary = (
        view.groupby("market_family", as_index=False)
        .agg(
            picks=("recommended_label", "count"),
            avg_odds=("recommended_odds", "mean"),
            avg_confidence=("model_confidence", "mean"),
            avg_edge=("recommended_edge", "mean"),
        )
        .sort_values(["picks", "avg_edge"], ascending=[False, False])
    )
    return summary


def build_market_segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    view = frame.copy()
    view["market_family"] = view.apply(market_family_from_row, axis=1)
    summary = (
        view.groupby(["segment", "market_family"], as_index=False)
        .agg(
            picks=("recommended_label", "count"),
            avg_odds=("recommended_odds", "mean"),
            avg_confidence=("model_confidence", "mean"),
            avg_edge=("recommended_edge", "mean"),
        )
        .sort_values(["segment", "picks", "avg_edge"], ascending=[True, False, False])
    )
    return summary


def pretty_market_family(value: Any) -> str:
    raw = str(value or "").strip().upper()
    mapping = {
        "1X2": "1X2",
        "TOTALS": "Alt/Ust",
        "BTTS": "KG",
        "DOUBLE_CHANCE": "Cifte Sans",
        "HANDICAP10": "Handikap 1:0",
        "ALT/UST": "Alt/Ust",
        "KG": "KG",
        "CIFTE SANS": "Cifte Sans",
        "HANDIKAP": "Handikap 1:0",
    }
    return mapping.get(raw, str(value or "Diger"))


def build_backtest_market_performance(bets_df: pd.DataFrame) -> pd.DataFrame:
    if bets_df is None or bets_df.empty:
        return pd.DataFrame()

    view = bets_df.copy()
    if "market_family" not in view.columns:
        view["market_family"] = view.apply(market_family_from_row, axis=1)

    view["market_family"] = view["market_family"].map(pretty_market_family)
    summary = (
        view.groupby("market_family", as_index=False)
        .agg(
            bets=("selected_label", "count"),
            wins=("won", "sum"),
            total_stake=("stake", "sum"),
            total_pnl=("pnl", "sum"),
            avg_odds=("odds", "mean"),
            avg_edge=("edge", "mean"),
            avg_probability=("probability", "mean"),
        )
        .sort_values(["bets", "total_pnl"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["hit_rate"] = np.where(summary["bets"] > 0, summary["wins"] / summary["bets"], np.nan)
    summary["roi"] = np.where(
        summary["total_stake"] > 0,
        summary["total_pnl"] / summary["total_stake"],
        np.nan,
    )
    drawdown_rows = []
    for (market_family, league_name), group in view.groupby(["market_family", "league_name"]):
        segment = group.copy()
        if "kickoff_time" in segment.columns:
            segment = segment.sort_values("kickoff_time")
        bankroll_curve = 1000.0 + pd.to_numeric(segment["pnl"], errors="coerce").fillna(0.0).cumsum()
        rolling_peak = bankroll_curve.cummax()
        drawdown = np.where(rolling_peak > 0, (bankroll_curve - rolling_peak) / rolling_peak, 0.0)
        max_drawdown = abs(float(np.nanmin(drawdown))) if len(drawdown) else 0.0
        drawdown_rows.append(
            {
                "market_family": market_family,
                "league_name": league_name,
                "max_drawdown": max_drawdown,
            }
        )
    summary = summary.merge(
        pd.DataFrame(drawdown_rows) if drawdown_rows else pd.DataFrame(columns=["market_family", "league_name", "max_drawdown"]),
        on=["market_family", "league_name"],
        how="left",
    )
    return summary[
        [
            "market_family",
            "bets",
            "wins",
            "hit_rate",
            "roi",
            "avg_odds",
            "avg_edge",
            "avg_probability",
            "max_drawdown",
            "total_stake",
            "total_pnl",
        ]
    ]


def build_backtest_market_league_performance(
    bets_df: pd.DataFrame,
    min_bets: int = 3,
) -> pd.DataFrame:
    if bets_df is None or bets_df.empty:
        return pd.DataFrame()

    view = bets_df.copy()
    if "market_family" not in view.columns:
        view["market_family"] = view.apply(market_family_from_row, axis=1)

    view["market_family"] = view["market_family"].map(pretty_market_family)
    if "kickoff_time" in view.columns:
        view["kickoff_time"] = pd.to_datetime(view["kickoff_time"], errors="coerce", utc=True)
    summary = (
        view.groupby(["market_family", "league_name"], as_index=False)
        .agg(
            bets=("selected_label", "count"),
            wins=("won", "sum"),
            total_stake=("stake", "sum"),
            total_pnl=("pnl", "sum"),
            avg_odds=("odds", "mean"),
            avg_edge=("edge", "mean"),
            avg_probability=("probability", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["hit_rate"] = np.where(summary["bets"] > 0, summary["wins"] / summary["bets"], np.nan)
    summary["roi"] = np.where(
        summary["total_stake"] > 0,
        summary["total_pnl"] / summary["total_stake"],
        np.nan,
    )
    drawdown_rows = []
    for (market_family, league_name), seg in view.groupby(["market_family", "league_name"]):
        seg_view = seg.copy()
        if "kickoff_time" in seg_view.columns:
            seg_view = seg_view.sort_values("kickoff_time", kind="stable")
        bankroll_curve = 1000.0 + pd.to_numeric(seg_view["pnl"], errors="coerce").fillna(0.0).cumsum()
        rolling_peak = bankroll_curve.cummax()
        drawdown_curve = np.where(rolling_peak > 0, (bankroll_curve - rolling_peak) / rolling_peak, 0.0)
        max_drawdown = abs(float(np.nanmin(drawdown_curve))) if len(drawdown_curve) else 0.0
        drawdown_rows.append(
            {
                "market_family": market_family,
                "league_name": league_name,
                "max_drawdown": round(max_drawdown, 4),
            }
        )
    if drawdown_rows:
        summary = summary.merge(pd.DataFrame(drawdown_rows), on=["market_family", "league_name"], how="left")
    else:
        summary["max_drawdown"] = np.nan
    summary = summary[summary["bets"] >= int(min_bets)].copy()
    summary = summary.sort_values(
        ["market_family", "roi", "bets"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return summary[
        [
            "market_family",
            "league_name",
            "bets",
            "wins",
            "hit_rate",
            "roi",
            "max_drawdown",
            "avg_odds",
            "avg_edge",
            "avg_probability",
            "total_stake",
            "total_pnl",
        ]
    ]


def build_segment_strength_lookup(
    bets_df: pd.DataFrame,
    min_bets: int = 3,
) -> pd.DataFrame:
    summary = build_backtest_market_league_performance(
        bets_df,
        min_bets=max(1, int(min_bets)),
    )
    if summary.empty:
        return pd.DataFrame()

    view = summary.copy()
    roi = view["roi"].fillna(-1.0)
    hit_rate = view["hit_rate"].fillna(0.0)
    max_drawdown = view["max_drawdown"].fillna(1.0)
    sample_bonus = np.clip(view["bets"].astype(float) / 10.0, 0.0, 1.0) * 0.10
    view["segment_score"] = (
        (roi * 0.52)
        + (hit_rate * 0.33)
        - (max_drawdown * 0.18)
        + sample_bonus
    ).round(4)
    conditions = [
        (view["bets"] >= max(6, int(min_bets))) & (view["roi"] >= 0.01) & (view["hit_rate"] >= 0.39) & (view["max_drawdown"] <= 0.30),
        (view["bets"] >= max(4, int(min_bets))) & ((view["roi"] >= -0.01) | (view["hit_rate"] >= 0.35)) & (view["max_drawdown"] <= 0.40),
    ]
    choices = ["strong", "medium"]
    view["segment_fit"] = np.select(conditions, choices, default="weak")
    view["segment_priority"] = view["segment_fit"].map(
        {"strong": 2, "medium": 1, "weak": 0}
    ).fillna(0).astype(int)
    return view[
        [
            "market_family",
            "league_name",
            "bets",
            "hit_rate",
            "roi",
            "max_drawdown",
            "segment_score",
            "segment_fit",
            "segment_priority",
        ]
    ].copy()


def apply_segment_strength_filter(
    frame: pd.DataFrame,
    segment_lookup: pd.DataFrame,
    target_coupon_size: int,
    strong_only: bool = False,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if frame is None or frame.empty or segment_lookup is None or segment_lookup.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(), {
            "enabled": False,
            "strong_pool": 0,
            "medium_pool": 0,
            "used_min_priority": None,
            "negative_roi_removed_count": 0,
            "used_roi_floor": None,
            "high_drawdown_removed_count": 0,
            "used_drawdown_cap": None,
        }

    view = frame.copy()
    view["market_family"] = view.apply(market_family_from_row, axis=1)
    view["market_family"] = view["market_family"].map(pretty_market_family)

    merge_cols = [
        "market_family",
        "league_name",
        "segment_score",
        "segment_fit",
        "segment_priority",
        "roi",
        "hit_rate",
        "max_drawdown",
    ]
    merged = view.merge(
        segment_lookup[merge_cols],
        on=["market_family", "league_name"],
        how="left",
        suffixes=("", "_segment"),
    )
    merged["segment_fit"] = merged["segment_fit"].fillna("unknown")
    merged["segment_priority"] = merged["segment_priority"].fillna(0).astype(int)
    merged["segment_score"] = merged["segment_score"].fillna(-1.0)
    merged["roi"] = pd.to_numeric(merged["roi"], errors="coerce")
    merged["max_drawdown"] = pd.to_numeric(merged["max_drawdown"], errors="coerce")
    negative_roi_mask = merged["roi"].notna() & (merged["roi"] < 0)
    negative_roi_removed_count = int(negative_roi_mask.sum())
    used_roi_floor = None
    if negative_roi_removed_count > 0:
        merged = merged[~negative_roi_mask].copy()
        used_roi_floor = 0.0
    high_drawdown_mask = merged["max_drawdown"].notna() & (merged["max_drawdown"] > 0.40)
    high_drawdown_removed_count = int(high_drawdown_mask.sum())
    used_drawdown_cap = None
    if high_drawdown_removed_count > 0:
        merged = merged[~high_drawdown_mask].copy()
        used_drawdown_cap = 0.40

    requested = max(1, int(target_coupon_size))
    strong_pool = int((merged["segment_priority"] >= 2).sum())
    medium_pool = int((merged["segment_priority"] >= 1).sum())
    used_min_priority = None

    if strong_pool >= requested:
        merged = merged[merged["segment_priority"] >= 2].copy()
        used_min_priority = 2
    elif strong_only:
        merged = merged[merged["segment_priority"] >= 2].copy()
        used_min_priority = 2 if not merged.empty else None
    elif medium_pool >= requested:
        merged = merged[merged["segment_priority"] >= 1].copy()
        used_min_priority = 1

    return merged, {
        "enabled": True,
        "strong_pool": strong_pool,
        "medium_pool": medium_pool,
        "used_min_priority": used_min_priority,
        "strong_only": bool(strong_only),
        "negative_roi_removed_count": negative_roi_removed_count,
        "used_roi_floor": used_roi_floor,
        "high_drawdown_removed_count": high_drawdown_removed_count,
        "used_drawdown_cap": used_drawdown_cap,
    }


def annotate_segment_badges(
    frame: pd.DataFrame,
    segment_lookup: pd.DataFrame,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    view = frame.copy()
    if "market_family" not in view.columns:
        view["market_family"] = view.apply(market_family_from_row, axis=1)
        view["market_family"] = view["market_family"].map(pretty_market_family)

    if segment_lookup is None or segment_lookup.empty:
        view["segment_fit"] = view.get("segment_fit", "unknown")
        existing_priority = (
            view["segment_priority"]
            if "segment_priority" in view.columns
            else pd.Series([0] * len(view), index=view.index, dtype=float)
        )
        view["segment_priority"] = pd.to_numeric(
            existing_priority, errors="coerce"
        ).fillna(0).astype(int)
        view["segment_badge"] = ""
        return view

    merge_cols = [
        "market_family",
        "league_name",
        "segment_fit",
        "segment_priority",
        "segment_score",
        "roi",
        "hit_rate",
        "max_drawdown",
        "bets",
    ]
    merge_view = segment_lookup[merge_cols].copy()
    view = view.merge(
        merge_view,
        on=["market_family", "league_name"],
        how="left",
        suffixes=("", "_segment_lookup"),
    )

    view["segment_fit"] = view["segment_fit"].fillna("unknown")
    view["segment_priority"] = pd.to_numeric(
        view["segment_priority"], errors="coerce"
    ).fillna(0).astype(int)
    view["segment_badge"] = np.select(
        [
            view["segment_priority"] >= 2,
            view["segment_priority"] >= 1,
        ],
        [
            "Favori Segment",
            "Izleme",
        ],
        default="",
    )
    return view


def build_coupon_with_smart_fallback(
    recommended_bets: pd.DataFrame,
    coupon_size: int,
    bankroll: float,
    coupon_profile: str,
    sort_by: str = "recommended_edge",
    fallback_enabled: bool = True,
) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    coupon, summary = build_coupon_from_recommendations(
        recommended_bets=recommended_bets,
        coupon_size=min(int(coupon_size), len(recommended_bets)) if len(recommended_bets) else int(coupon_size),
        bankroll=float(bankroll),
        sort_by=sort_by,
        coupon_profile=coupon_profile,
    )
    fallback_meta = {
        "used": False,
        "from_profile": coupon_profile,
        "to_profile": None,
        "reason": None,
        "requested_coupon_size": int(coupon_size),
    }
    if str(coupon_profile) != "Temkinli" or not bool(fallback_enabled):
        return coupon, summary, fallback_meta
    if recommended_bets is None or recommended_bets.empty or len(recommended_bets) < 2:
        return coupon, summary, fallback_meta
    if int(summary.get("bet_count", 0)) >= 2:
        return coupon, summary, fallback_meta

    fallback_attempts = []
    full_size = min(int(coupon_size), len(recommended_bets))
    if full_size >= 2:
        fallback_attempts.append(("Dengeli", full_size))
    reduced_size = min(2, len(recommended_bets))
    if reduced_size >= 2 and ("Dengeli", reduced_size) not in fallback_attempts:
        fallback_attempts.append(("Dengeli", reduced_size))

    for fallback_profile, fallback_size in fallback_attempts:
        alt_coupon, alt_summary = build_coupon_from_recommendations(
            recommended_bets=recommended_bets,
            coupon_size=fallback_size,
            bankroll=float(bankroll),
            sort_by=sort_by,
            coupon_profile=fallback_profile,
        )
        if int(alt_summary.get("bet_count", 0)) >= 2:
            alt_summary = dict(alt_summary)
            alt_summary["fallback_from_profile"] = coupon_profile
            alt_summary["fallback_profile_used"] = fallback_profile
            fallback_meta = {
                "used": True,
                "from_profile": coupon_profile,
                "to_profile": fallback_profile,
                "reason": "Temkinli mod kupon cikaramadigi icin kontrollu fallback kullanildi.",
                "requested_coupon_size": int(coupon_size),
            }
            return alt_coupon, alt_summary, fallback_meta

    return coupon, summary, fallback_meta


def render_match_center_board(matches_df: pd.DataFrame, limit: int = 120) -> None:
    if matches_df is None or matches_df.empty:
        st.info("Mac Merkezi icin gosterilecek kayit yok.")
        return

    rows = []
    for _, row in matches_df.head(limit).iterrows():
        status_text = str(row.get("status") or "-").upper()
        source_text = str(row.get("odds_source") or row.get("source") or "-").upper()
        market_specs = [
            (
                "1X2",
                ["home_odds", "draw_odds", "away_odds"],
            ),
            (
                "2.5",
                ["under_25_odds", "over_25_odds"],
            ),
            (
                "C.Sans",
                [
                    "double_chance_1x_odds",
                    "double_chance_12_odds",
                    "double_chance_x2_odds",
                    "market_double_chance_1x_odds",
                    "market_double_chance_12_odds",
                    "market_double_chance_x2_odds",
                ],
            ),
            (
                "H1:0",
                [
                    "handicap_home_10_odds",
                    "handicap_draw_10_odds",
                    "handicap_away_10_odds",
                    "market_handicap_home_10_odds",
                    "market_handicap_draw_10_odds",
                    "market_handicap_away_10_odds",
                ],
            ),
        ]
        badges_html = "".join(
            f"<span class='gabrela-market-chip{' inactive' if not any(pd.notna(row.get(col)) for col in cols) else ''}'>{html_lib.escape(label)}</span>"
            for label, cols in market_specs
        )
        rows.append(
            textwrap.dedent(
                f"""
                <div class="gabrela-match-row">
                    <div class="gabrela-match-time">{html_lib.escape(format_kickoff(row.get('kickoff_time')))}</div>
                    <div class="gabrela-match-status">{html_lib.escape(status_text)}</div>
                    <div class="gabrela-match-title-wrap">
                        <div class="gabrela-match-main">{html_lib.escape(str(row.get('home_team', '-')))} - {html_lib.escape(str(row.get('away_team', '-')))}</div>
                        <div class="gabrela-match-sub">{html_lib.escape(str(row.get('league', '-')))}</div>
                        <div class="gabrela-match-badges">{badges_html}</div>
                    </div>
                    <div class="gabrela-match-source">{html_lib.escape(source_text)}</div>
                    <div class="gabrela-match-odd-cell"><span>1</span><strong>{html_lib.escape(number_text(row.get('home_odds')))}</strong></div>
                    <div class="gabrela-match-odd-cell"><span>X</span><strong>{html_lib.escape(number_text(row.get('draw_odds')))}</strong></div>
                    <div class="gabrela-match-odd-cell"><span>2</span><strong>{html_lib.escape(number_text(row.get('away_odds')))}</strong></div>
                    <div class="gabrela-match-odd-cell"><span>Alt</span><strong>{html_lib.escape(number_text(row.get('under_25_odds')))}</strong></div>
                    <div class="gabrela-match-odd-cell"><span>Ust</span><strong>{html_lib.escape(number_text(row.get('over_25_odds')))}</strong></div>
                </div>
                """
            ).strip()
        )

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="gabrela-match-board">
                <div class="gabrela-match-head">
                    <div>Saat</div>
                    <div>Durum</div>
                    <div>Mac</div>
                    <div>Kaynak</div>
                    <div>1</div>
                    <div>X</div>
                    <div>2</div>
                    <div>Alt</div>
                    <div>Ust</div>
                </div>
                {''.join(rows)}
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def format_pick_label(row: pd.Series) -> str:
    label = str(row.get("recommended_label", "")).upper()
    market = str(row.get("recommended_market", "")).strip()
    if label == "H":
        return "Mac Sonucu 1"
    if label == "D":
        return "Mac Sonucu X"
    if label == "A":
        return "Mac Sonucu 2"
    if label == "O25":
        return "2.5 Ust"
    if label == "U25":
        return "2.5 Alt"
    if label == "BYES":
        return "KG Var"
    if label == "BNO":
        return "KG Yok"
    if label == "1X":
        return "Cifte Sans 1X"
    if label == "12":
        return "Cifte Sans 12"
    if label == "X2":
        return "Cifte Sans X2"
    if label == "H10H":
        return "Handikap 1:0 / 1"
    if label == "H10D":
        return "Handikap 1:0 / 0"
    if label == "H10A":
        return "Handikap 1:0 / 2"
    return market or label or "-"


def render_simple_pick_cards(picks: pd.DataFrame) -> None:
    if picks is None or picks.empty:
        st.info("Bu kutu icin uygun mac bulunmadi.")
        return

    for idx, (_, pick) in enumerate(picks.iterrows(), start=1):
        source_name = str(pick.get("source", "")).lower()
        source_label = str(pick.get("source", "-") or "-").upper()
        odds_label = str(pick.get("odds_source", "-") or "-").upper()
        mbs_value = pick.get("mbs")
        if mbs_value is None or pd.isna(mbs_value) or str(mbs_value).strip() == "":
            mbs_value = pick.get("recommended_mbs")
        if mbs_value is None or pd.isna(mbs_value) or str(mbs_value).strip() == "":
            mbs_value = idx
        segment_priority = int(pd.to_numeric(pd.Series([pick.get("segment_priority")]), errors="coerce").fillna(0).iloc[0])
        segment_badge = str(pick.get("segment_badge") or "").strip()
        search_prefix = (
            "iddaa'da ara:"
            if source_name == "iddaa"
            else "Sitede birebir bulunmayabilir. Arama metni:"
        )
        with st.container():
            top_cols = st.columns([1.0, 3.8, 1.6], gap="small")
            top_cols[0].markdown(f"**MBS {mbs_value}**")
            top_cols[1].markdown(f"**{pick.get('home_team', '-')} - {pick.get('away_team', '-')}**")
            badge_text = format_pick_label(pick)
            if segment_badge:
                badge_text = f"{badge_text} | {segment_badge}"
            top_cols[2].markdown(f"**{badge_text}**")

            st.caption(
                f"Lig: {pick.get('league_name', '-')} | Kaynak: {source_label} | Oran kaynagi: {odds_label}"
            )

            cell_cols = st.columns(5, gap="small")
            cell_cols[0].metric("Saat", format_kickoff(pick.get("kickoff_time")))
            cell_cols[1].metric("MBS", str(mbs_value))
            cell_cols[2].metric("Market", format_pick_label(pick))
            cell_cols[3].metric("Oran", number_text(pick.get("recommended_odds")))
            cell_cols[4].metric("Stake", number_text(pick.get("flat_stake")))

            st.caption(
                f"{search_prefix} {pick.get('home_team', '-')} - {pick.get('away_team', '-')}"
            )
            st.divider()


def render_compact_pick_board(
    picks: pd.DataFrame,
    limit: int = 24,
    selection_state_key: str | None = None,
) -> None:
    if picks is None or picks.empty:
        st.info("Bugun icin filtreleri gecen mac yok.")
        return

    header_cols = st.columns([1.0, 1.0, 3.2, 1.8, 1.0, 1.0, 1.2], gap="small")
    header_labels = ["SAAT", "MBS", "MAC", "TAHMIN", "ORAN", "GUVEN", "KUPON"]
    selected_labels = (
        list(st.session_state.get(selection_state_key, []))
        if selection_state_key
        else []
    )
    for col, label in zip(header_cols, header_labels):
        col.markdown(f"**{label}**")

    for idx, (_, pick) in enumerate(picks.head(limit).iterrows(), start=1):
        manual_label = build_manual_coupon_label(pick, idx)
        mbs_value = pick.get("mbs")
        if mbs_value is None or pd.isna(mbs_value) or str(mbs_value).strip() == "":
            mbs_value = pick.get("recommended_mbs")
        if mbs_value is None or pd.isna(mbs_value) or str(mbs_value).strip() == "":
            mbs_value = idx
        segment_priority = int(pd.to_numeric(pd.Series([pick.get("segment_priority")]), errors="coerce").fillna(0).iloc[0])
        segment_badge = str(pick.get("segment_badge") or "").strip()
        segment_badge_html = ""
        if segment_badge:
            tone_class = " medium" if segment_priority == 1 else ""
            segment_badge_html = (
                f"<span class='gabrela-favorite-chip{tone_class}'>{html_lib.escape(segment_badge)}</span>"
            )
        with st.container(border=True):
            cols = st.columns([1.0, 1.0, 3.2, 1.8, 1.0, 1.0, 1.2], gap="small")
            cols[0].markdown(f"**{html_lib.escape(format_kickoff(pick.get('kickoff_time')))}**")
            cols[1].markdown(
                f"<span class='gabrela-market-chip'>MBS {html_lib.escape(str(mbs_value))}</span>",
                unsafe_allow_html=True,
            )
            cols[2].markdown(
                (
                    f"**{html_lib.escape(str(pick.get('home_team', '-')))} - "
                    f"{html_lib.escape(str(pick.get('away_team', '-')))}**  \n"
                    f"<span style='color:#94a3b8;font-size:0.80rem'>{html_lib.escape(str(pick.get('league_name', '-')))}</span>"
                ),
                unsafe_allow_html=True,
            )
            market_html = (
                f"<span class='gabrela-market-chip'>{html_lib.escape(format_pick_label(pick))}</span>"
                f"{segment_badge_html}"
                f"<div class='gabrela-bulletin-meta'>{html_lib.escape(str(pick.get('source', '-')).upper())}</div>"
            )
            cols[3].markdown(market_html, unsafe_allow_html=True)
            cols[4].markdown(
                (
                    f"**{html_lib.escape(number_text(pick.get('recommended_odds')))}**  \n"
                    "<span style='color:#94a3b8;font-size:0.78rem'>oran</span>"
                ),
                unsafe_allow_html=True,
            )
            cols[5].markdown(
                (
                    f"**{html_lib.escape(pct_text(pick.get('model_confidence')))}**  \n"
                    "<span style='color:#94a3b8;font-size:0.78rem'>guven</span>"
                ),
                unsafe_allow_html=True,
            )
            if selection_state_key:
                is_selected = manual_label in selected_labels
                button_label = "Cikar" if is_selected else "Kupona ekle"
                if cols[6].button(
                    button_label,
                    key=f"quick_coupon_{selection_state_key}_{idx}_{pick.get('match_id', idx)}",
                    use_container_width=True,
                ):
                    updated_labels = list(st.session_state.get(selection_state_key, []))
                    if is_selected:
                        updated_labels = [label for label in updated_labels if label != manual_label]
                    else:
                        updated_labels.append(manual_label)
                    st.session_state[selection_state_key] = updated_labels
                    st.rerun()


def render_coupon_signal_panel(
    title: str,
    picks: pd.DataFrame,
    summary: dict,
    min_pick_count: int = 2,
) -> None:
    if picks is None or picks.empty:
        verdict = "PAS GEC"
        bg_color = "#7f1d1d"
        border_color = "#ef4444"
        reason = "Yeterli uygun mac yok."
        avg_edge = None
        avg_confidence = None
        selection_steps_html = ""
        summary_notes_html = ""
        segment_mix_html = ""
    else:
        avg_edge = float(picks["recommended_edge"].mean()) if "recommended_edge" in picks else None
        avg_confidence = (
            float(picks["model_confidence"].mean()) if "model_confidence" in picks else None
        )
        segment_priority_series = (
            pd.to_numeric(picks["segment_priority"], errors="coerce").fillna(0).astype(int)
            if "segment_priority" in picks.columns
            else pd.Series([0] * len(picks), index=picks.index, dtype=int)
        )
        favorite_count = int((segment_priority_series >= 2).sum())
        watch_count = int((segment_priority_series == 1).sum())
        fallback_count = max(int(len(picks)) - favorite_count - watch_count, 0)
        segment_parts = []
        if favorite_count:
            segment_parts.append(
                f"<span class='gabrela-favorite-chip'>Favori Segment: {favorite_count}</span>"
            )
        if watch_count:
            segment_parts.append(
                f"<span class='gabrela-favorite-chip medium'>Izleme: {watch_count}</span>"
            )
        segment_mix_html = (
            "<div style='margin-top:12px;'>"
            "<div style='font-size:13px; font-weight:700; color:#e2e8f0; margin-bottom:8px;'>Segment ozeti</div>"
            f"<div style='display:flex; gap:8px; flex-wrap:wrap;'>{''.join(segment_parts)}</div>"
            f"<div style='margin-top:8px; font-size:13px; color:#dbeafe;'>"
            f"Bu kupondaki {favorite_count} secim favori segmentten, {watch_count} secim izleme segmentinden geliyor. "
            f"Kalan {fallback_count} secim genel havuzdan secildi."
            "</div>"
            "</div>"
        )
        summary_lines = summary.get("selection_summary_lines") or []
        if summary_lines:
            notes = "".join(
                f"<li>{html_lib.escape(str(line))}</li>"
                for line in summary_lines
            )
            summary_notes_html = (
                "<div style='margin-top:12px;font-size:14px;'>"
                "<b>Bu secimler neden kaldi?</b>"
                f"<ul style='margin:8px 0 0 18px;padding:0;'>{notes}</ul>"
                "</div>"
            )
        else:
            summary_notes_html = ""
        all_iddaa = (
            "source" in picks.columns and picks["source"].fillna("").str.lower().eq("iddaa").all()
        )
        should_play = (
            len(picks) >= min_pick_count
            and avg_edge is not None
            and avg_confidence is not None
            and avg_edge >= 0.04
            and avg_confidence >= 0.45
        )
        if should_play:
            verdict = "OYNA"
            bg_color = "#14532d"
            border_color = "#22c55e"
            reason = "Edge ve confidence seviyesi bugun icin yeterli gorunuyor."
            steps = []
            for idx, (_, pick) in enumerate(picks.iterrows(), start=1):
                match_name = (
                    f"{pick.get('home_team', '-')} - {pick.get('away_team', '-')}"
                )
                step_text = (
                    f"{idx}. {format_kickoff(pick.get('kickoff_time'))} "
                    f"{match_name} ({pick.get('league_name', '-')}) -> {format_pick_label(pick)}"
                )
                steps.append(f"<li>{html_lib.escape(step_text)}</li>")
            steps_header = (
                "iddaa ekraninda sirayla sunlari sec:"
                if all_iddaa
                else "Bahis ekraninda sirayla sunlari sec:"
            )
            selection_steps_html = (
                "<div style='margin-top:12px;font-size:14px;'>"
                f"<b>{steps_header}</b>"
                f"<ol style='margin:8px 0 0 18px;padding:0;'>{''.join(steps)}</ol>"
                "</div>"
            )
        else:
            verdict = "PAS GEC"
            bg_color = "#7f1d1d"
            border_color = "#ef4444"
            reason = "Kupon kalitesi zayif; bugun zorlamamak daha guvenli."
            selection_steps_html = (
                "<div style='margin-top:12px;font-size:14px;'>"
                "Bugun sistemin verdigi kuponu oynamak yerine pas gecmek daha guvenli."
                "</div>"
            )

    panel_html = textwrap.dedent(
        f"""
    <div style="
        background:{bg_color};
        border:2px solid {border_color};
        border-radius:16px;
        padding:18px 20px;
        margin:10px 0 16px 0;
        color:white;
    ">
        <div style="font-size:14px; opacity:0.9;">{title}</div>
        <div style="font-size:34px; font-weight:800; margin-top:6px;">{verdict}</div>
        <div style="font-size:15px; margin-top:8px;">{reason}</div>
        <div style="font-size:14px; margin-top:10px; opacity:0.95;">
            Mac sayisi: {summary.get('bet_count', 0)} |
            Toplam oran: {number_text(summary.get('total_odds'))} |
            Duz stake: {number_text(summary.get('flat_total_stake'))}
        </div>
        {segment_mix_html}
        {summary_notes_html}
        {selection_steps_html}
    </div>
    """
    ).strip()
    st.markdown(panel_html, unsafe_allow_html=True)
    if avg_edge is not None and avg_confidence is not None:
        c1, c2 = st.columns(2)
        c1.metric("Ort. edge", pct_text(avg_edge))
        c2.metric("Ort. confidence", pct_text(avg_confidence))


def render_sidebar_feedback(message_key: str, details_key: str) -> None:
    message = st.session_state.get(message_key)
    if not message:
        return

    level = message.get("level", "info")
    text = message.get("text", "")

    if level == "success":
        st.success(text)
    elif level == "error":
        st.error(text)
    else:
        st.info(text)


def render_nesine_diagnostics_block(details: dict) -> None:
    diagnostics = details.get("diagnostics") if isinstance(details, dict) else None
    if not isinstance(diagnostics, dict):
        return

    st.warning("Nesine tani modu: veri parse edilemedigi icin teshis bilgisi gosteriliyor.")
    top1, top2, top3 = st.columns(3)
    top1.metric("Aday event", diagnostics.get("candidate_event_count", 0))
    top2.metric("Normalize event", diagnostics.get("normalized_event_count", 0))
    top3.metric("HTTP durum", diagnostics.get("status_code", "-"))

    summary = {
        "endpoint_name": diagnostics.get("endpoint_name"),
        "fetched_url": diagnostics.get("fetched_url"),
        "content_type": diagnostics.get("content_type"),
        "top_level_type": diagnostics.get("top_level_type"),
        "top_level_keys": diagnostics.get("top_level_keys"),
        "walk_dict_count": diagnostics.get("walk_dict_count"),
        "walk_list_count": diagnostics.get("walk_list_count"),
    }
    st.json(summary)

    sample_candidates = diagnostics.get("sample_candidates")
    if isinstance(sample_candidates, list) and sample_candidates:
        st.write("Nesine aday node ornekleri")
        render_table(pd.DataFrame(sample_candidates), height=220)

    normalized_preview = diagnostics.get("normalized_preview")
    if isinstance(normalized_preview, list) and normalized_preview:
        st.write("Normalize edilebilen ornekler")
        render_table(pd.DataFrame(normalized_preview), height=220)

    response_preview = diagnostics.get("response_preview")
    if response_preview:
        with st.expander("Ham cevap onizlemesi", expanded=False):
            st.code(str(response_preview))


def render_dashboard_kpi_row(cards: list[dict[str, str]]) -> None:
    if not cards:
        return

    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        accent_html = (
            f"<div class='gabrela-kpi-sub'><span class='gabrela-kpi-accent'>{html_lib.escape(str(card.get('accent', '')))}</span></div>"
            if card.get("accent")
            else ""
        )
        subtitle_html = (
            f"<div class='gabrela-kpi-sub'>{html_lib.escape(str(card.get('subtitle') or card.get('sub') or ''))}</div>"
            if (card.get("subtitle") or card.get("sub"))
            else ""
        )
        col.markdown(
            textwrap.dedent(
                f"""
            <div class="gabrela-kpi-card">
                <div class="gabrela-kpi-head">
                    <div class="gabrela-kpi-label">{html_lib.escape(str(card.get("label", "")))}</div>
                    <div class="gabrela-kpi-icon">{html_lib.escape(str(card.get("icon", "*")))}</div>
                </div>
                <div class="gabrela-kpi-value">{html_lib.escape(str(card.get("value", "-")))}</div>
                {accent_html}
                {subtitle_html}
            </div>
            """
            ).strip(),
            unsafe_allow_html=True,
        )


def render_panel_card(title: str, subtitle: str = "", body_html: str = "") -> None:
    subtitle_html = (
        f"<div class='gabrela-panel-subtitle'>{html_lib.escape(subtitle)}</div>" if subtitle else ""
    )
    st.markdown(
        textwrap.dedent(
            f"""
        <div class="gabrela-panel-card">
            <div class="gabrela-panel-title">{html_lib.escape(title)}</div>
            {subtitle_html}
            {body_html}
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, badge: str = "*") -> None:
    st.markdown(
        textwrap.dedent(
            f"""
        <div class="gabrela-section-title">
            <span class="gabrela-section-title-badge">{html_lib.escape(badge)}</span>
            <span>{html_lib.escape(title)}</span>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_operation_console() -> None:
    data_feedback = st.session_state.get("sidebar_data_feedback")
    data_details = st.session_state.get("sidebar_data_details")
    model_feedback = st.session_state.get("sidebar_model_feedback")
    model_details = st.session_state.get("sidebar_model_details")

    if not any([data_feedback, data_details, model_feedback, model_details]):
        return

    st.subheader("Son islem ozeti")

    if data_feedback:
        level = data_feedback.get("level", "info")
        text = data_feedback.get("text", "")
        if level == "success":
            st.success(f"Veri islemi: {text}")
        elif level == "error":
            st.error(f"Veri islemi: {text}")
        else:
            st.info(f"Veri islemi: {text}")

        if data_details is not None:
            with st.expander("Veri islemi detaylari", expanded=False):
                if isinstance(data_details, pd.DataFrame):
                    render_table(data_details, height=220)
                elif isinstance(data_details, (dict, list)):
                    st.json(data_details)
                else:
                    st.code(str(data_details))
            if isinstance(data_details, dict):
                render_nesine_diagnostics_block(data_details)

    if model_feedback:
        level = model_feedback.get("level", "info")
        text = model_feedback.get("text", "")
        if level == "success":
            st.success(f"Model islemi: {text}")
        elif level == "error":
            st.error(f"Model islemi: {text}")
        else:
            st.info(f"Model islemi: {text}")

        if model_details is not None:
            with st.expander("Model islemi detaylari", expanded=False):
                if isinstance(model_details, pd.DataFrame):
                    render_table(model_details, height=220)
                elif isinstance(model_details, (dict, list)):
                    st.json(model_details)
                else:
                    st.code(str(model_details))


@contextmanager
def activity_status(kind: str, text: str):
    st.session_state["runtime_activity"] = {
        "state": "running",
        "kind": kind,
        "text": text,
        "started_at": datetime.now(APP_TIMEZONE).isoformat(),
    }
    with st.status(f"{kind}: {text}", expanded=False) as status_box:
        try:
            yield status_box
        except Exception:
            status_box.update(label=f"{kind}: hata olustu", state="error")
            st.session_state["runtime_activity"] = {
                "state": "error",
                "kind": kind,
                "text": text,
                "ended_at": datetime.now(APP_TIMEZONE).isoformat(),
            }
            raise
        else:
            status_box.update(label=f"{kind}: tamamlandi", state="complete")
            st.session_state["runtime_activity"] = {
                "state": "complete",
                "kind": kind,
                "text": text,
                "ended_at": datetime.now(APP_TIMEZONE).isoformat(),
            }


def render_system_health_report(report: dict | None) -> None:
    if not isinstance(report, dict):
        st.warning("Sistem saglik raporu henuz hazir degil.")
        return

    status = report.get("status", "caution")
    summary = report.get("summary", "")
    reasons = report.get("reasons") or []
    model_report = report.get("model_report") or {}
    data_quality = report.get("data_quality") or {}

    if status == "pass":
        st.success(f"Sistem sagligi: GECERLI | {summary}")
    elif status == "fail":
        st.error(f"Sistem sagligi: RISKLI | {summary}")
    else:
        st.warning(f"Sistem sagligi: DIKKAT | {summary}")

    st.markdown("**Model metrikleri**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Egitim satiri", metric_text(model_report.get("rows_total"), "int"))
    m2.metric("Accuracy", metric_text(model_report.get("accuracy"), "pct"))
    m3.metric("Log loss", metric_text(model_report.get("log_loss"), "float"))
    m4.metric("CV log loss", metric_text(model_report.get("cv_log_loss_mean"), "float"))
    m5.metric("Brier", metric_text(model_report.get("brier_score"), "float"))

    st.markdown("**Veri kalitesi**")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Toplam mac", metric_text(data_quality.get("total_matches"), "int"))
    q2.metric("Tamamlanmis", metric_text(data_quality.get("completed_matches"), "int"))
    q3.metric("Programda", metric_text(data_quality.get("scheduled_matches"), "int"))
    q4.metric("Canli", metric_text(data_quality.get("live_matches"), "int"))
    q5.metric("Supheli 1X2", metric_text(data_quality.get("invalid_1x2_odds_count"), "int"))

    source_summary = data_quality.get("source_summary") or {}
    if isinstance(source_summary, dict) and source_summary:
        st.markdown("**Kaynak dagilimi**")
        source_rows = pd.DataFrame(
            [
                {"Kaynak": str(source_name).upper(), "Mac sayisi": int(count)}
                for source_name, count in source_summary.items()
            ]
        ).sort_values("Mac sayisi", ascending=False)
        render_table(source_rows, height=min(240, 60 + 36 * len(source_rows)))

    if reasons:
        st.write("Risk notlari")
        for reason in reasons:
            st.write(f"- {reason}")


def main() -> None:
    st.set_page_config(
        page_title=BRAND_TITLE,
        page_icon=":soccer:",
        layout="wide",
    )
    create_tables()
    inject_app_shell_css()

    render_brand_header()

    with st.sidebar:
        render_sidebar_brand_shell()
        render_sidebar_flow()
        st.session_state.setdefault("sidebar_data_feedback", None)
        st.session_state.setdefault("sidebar_data_details", None)
        st.session_state.setdefault("sidebar_model_feedback", None)
        st.session_state.setdefault("sidebar_model_details", None)
        st.session_state.setdefault("system_health_report", None)
        st.session_state.setdefault("runtime_activity", None)
        st.session_state.setdefault("manual_coupon_selected_labels", [])
        with st.expander("[01] Kontrol Merkezi", expanded=True):
            st.markdown(
                "<div class='gabrela-sidebar-mini-note'>Gunluk kupon davranisini ve kasa ayarlarini buradan yonet.</div>",
                unsafe_allow_html=True,
            )
            model_type = st.selectbox(
                "Model tipi",
                options=["hist_gradient_boosting", "random_forest", "logreg"],
                format_func=lambda value: {
                    "logreg": "Lojistik Regresyon",
                    "random_forest": "Random Forest",
                    "hist_gradient_boosting": "Hist Gradient Boosting",
                }[value],
            )
            edge_threshold = st.slider(
                "Minimum edge esigi",
                min_value=0.01,
                max_value=0.15,
                value=0.02,
                step=0.01,
            )
            bankroll_amount = st.number_input(
                "Bankroll",
                min_value=100.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0,
            )
            flat_stake_amount = st.number_input(
                "Flat stake",
                min_value=5.0,
                max_value=10000.0,
                value=25.0,
                step=5.0,
            )
            coupon_size = st.slider("Kupon secim adedi", min_value=2, max_value=8, value=4)
            coupon_profile = st.selectbox(
                "Kupon modu",
                options=["Temkinli", "Dengeli", "Yuksek Oran"],
                index=0,
            )
            st.caption(
                {
                    "Temkinli": "Daha dusuk oranli, daha guvenli ve ayni ligde yigilmayan kupon secmeye calisir.",
                    "Dengeli": "Guven ile oran arasinda orta yol arar; gunluk kupon icin dengeli varsayilandir.",
                    "Yuksek Oran": "Daha cesur secimler yapar; toplam oran ve lig cesitliligi daha genistir.",
                }[coupon_profile]
            )
            coupon_source_mode = st.selectbox(
                "Kupon kaynagi",
                options=["Sadece IDDAA", "Tum kaynaklar"],
                index=0,
            )
            strong_segments_only = st.toggle(
                "Sadece guclu segmentleri kullan",
                value=False,
                help="Backtestte ROI ve hit rate olarak guclu gorunen lig x market segmentlerine kuponu daraltir.",
            )
            temkinli_fallback_enabled = st.toggle(
                "Temkinli fallback kullan",
                value=True,
                help="Temkinli mod kupon cikaramazsa kontrollu Dengeli fallback dener.",
            )
            segment_min_bets = st.slider(
                "Minimum segment bet sayisi",
                min_value=3,
                max_value=12,
                value=5,
                step=1,
                help="Lig x market gucunu hesaplarken en az kac backtest bahsi olmasi gerektigini belirler.",
            )

        with st.expander("[02] Egitim Ayarlari", expanded=False):
            st.markdown(
                "<div class='gabrela-sidebar-mini-note'>Tarihsel egitim kapsamini ve sezon derinligini buradan belirle.</div>",
                unsafe_allow_html=True,
            )
            default_leagues = default_training_league_codes(profile="broad")
            historical_leagues = st.multiselect(
                "Tarihsel ligler",
                options=sorted(FOOTBALL_DATA_LEAGUES.keys()),
                default=default_leagues,
                format_func=lambda code: f"{code} - {FOOTBALL_DATA_LEAGUES[code]['league_name']}",
            )
            season_count = st.slider("Tarihsel sezon sayisi", min_value=1, max_value=10, value=8)
            historical_seasons = default_historical_season_starts(num_seasons=season_count)

        with st.expander("[03] Veri Islemleri", expanded=True):
            st.markdown(
                f"<div class='gabrela-sidebar-mini-note'>Yerel CSV klasoru (alt klasorler dahil): {html_lib.escape(str(LOCAL_CSV_IMPORT_DIR))}</div>",
                unsafe_allow_html=True,
            )
            with st.form("sidebar_data_form"):
                data_action = st.selectbox(
                    "Veri aksiyonu",
                    options=[
                        ("demo", "Demo veriyi yukle"),
                        ("historical", "Tarihsel sonuclari yukle"),
                        ("download_csv_auto", "CSV paketini otomatik indir"),
                        ("download_import_train_auto", "CSV paketini indir + ice aktar + modeli egit"),
                        ("local_csv_auto", "Yerel CSV klasorunu ice aktar ve modeli egit"),
                        ("iddaa", "iddaa sportsbook verisini cek"),
                        ("cleanup_upcoming", "Sahte scheduled/live kayitlari temizle"),
                        ("cleanup_odds", "Supheli aktif 1X2 oranlarini temizle"),
                    ],
                    format_func=lambda item: item[1],
                )[0]
                data_submitted = st.form_submit_button("Veri islemini calistir", width="stretch")

        if data_submitted:
            try:
                if data_action == "demo":
                    with activity_status("Veri islemi", "Demo veriler olusturuluyor"):
                        result = seed_demo_data(force=False)
                elif data_action == "historical":
                    with activity_status("Veri islemi", "Tarihsel CSV arsivi okunuyor"):
                        result = persist_historical_results_from_football_data(
                            league_codes=historical_leagues,
                            season_starts=historical_seasons,
                        )
                elif data_action == "local_csv_auto":
                    with activity_status("Veri islemi", "Yerel CSV klasoru ice aktariliyor ve model egitiliyor"):
                        import_result = import_historical_results_from_local_csv_folder(
                            str(LOCAL_CSV_IMPORT_DIR)
                        )
                        if import_result.get("ok"):
                            clear_runtime_caches()
                            train_metrics = train_and_save_model(model_type=model_type)
                            st.session_state["system_health_report"] = build_system_health_report()
                            st.session_state["sidebar_model_feedback"] = {
                                "level": "success",
                                "text": "Yerel CSV import sonrasi model otomatik egitildi.",
                            }
                            st.session_state["sidebar_model_details"] = train_metrics
                        result = {
                            "import_result": import_result,
                            "train_metrics": st.session_state.get("sidebar_model_details")
                            if import_result.get("ok")
                            else None,
                            "system_health_report": st.session_state.get("system_health_report")
                            if import_result.get("ok")
                            else None,
                            "ok": import_result.get("ok", False),
                            "source": "local_csv_auto",
                        }
                elif data_action == "download_import_train_auto":
                    with activity_status("Veri islemi", "CSV paketi indiriliyor, ice aktariliyor ve model egitiliyor"):
                        download_result = download_bulk_import_csv_bundle()
                        import_result = import_historical_results_from_local_csv_folder(
                            str(LOCAL_CSV_IMPORT_DIR)
                        )
                        train_metrics = None
                        health_report = None
                        if import_result.get("ok"):
                            clear_runtime_caches()
                            train_metrics = train_and_save_model(model_type=model_type)
                            health_report = build_system_health_report()
                            st.session_state["system_health_report"] = health_report
                            st.session_state["sidebar_model_feedback"] = {
                                "level": "success",
                                "text": "CSV paketi sonrasi model otomatik egitildi.",
                            }
                            st.session_state["sidebar_model_details"] = train_metrics
                        result = {
                            "download_result": download_result,
                            "import_result": import_result,
                            "train_metrics": train_metrics,
                            "system_health_report": health_report,
                            "ok": import_result.get("ok", False),
                            "source": "download_import_train_auto",
                        }
                elif data_action == "download_csv_auto":
                    with activity_status("Veri islemi", "Onerilen CSV paketi indiriliyor"):
                        result = download_bulk_import_csv_bundle()
                elif data_action == "iddaa":
                    with activity_status("Veri islemi", "iddaa endpoint'leri okunuyor"):
                        result = persist_upcoming_from_iddaa(include_highlighted=True)
                elif data_action == "cleanup_upcoming":
                    with activity_status("Veri islemi", "Supheli scheduled/live kayitlari temizleniyor"):
                        result = cleanup_suspicious_upcoming_matches()
                elif data_action == "cleanup_odds":
                    with activity_status("Veri islemi", "Supheli aktif 1X2 oranlari temizleniyor"):
                        session = get_session()
                        try:
                            result = cleanup_invalid_active_1x2_odds_snapshots(session)
                            session.commit()
                        finally:
                            session.close()
                else:
                    raise RuntimeError("Desteklenmeyen veri aksiyonu.")

                if result.get("ok"):
                    clear_runtime_caches()
                    st.session_state["sidebar_data_feedback"] = {
                        "level": "success",
                        "text": "Veri islemi tamamlandi.",
                    }
                else:
                    error_text = result.get("error") or result.get("message") or "Veri islemi hata verdi."
                    st.session_state["sidebar_data_feedback"] = {
                        "level": "error",
                        "text": f"Veri islemi hata verdi: {error_text}",
                    }
                st.session_state["sidebar_data_details"] = result
            except Exception as exc:
                st.session_state["sidebar_data_feedback"] = {
                    "level": "error",
                    "text": f"Veri islemi hatasi: {exc}",
                }
                st.session_state["sidebar_data_details"] = None

        with st.expander("[04] Veri Sonucu", expanded=False):
            render_sidebar_feedback("sidebar_data_feedback", "sidebar_data_details")

        with st.expander("[05] Model Islemleri", expanded=True):
            st.markdown(
                "<div class='gabrela-sidebar-mini-note'>Tahmin, geri test ve karsilastirma akisini bu bloktan calistir.</div>",
                unsafe_allow_html=True,
            )
            with st.form("sidebar_model_form"):
                model_action = st.selectbox(
                    "Model aksiyonu",
                    options=[
                        ("train", "Modeli egit"),
                        ("predict", "Yaklasan maclari tahmin et"),
                        ("backtest", "Calistir'de geri test"),
                        ("flat_kelly", "Flat vs Kelly Karsilastir"),
                        ("explain", "Model aciklamalarini hazirla"),
                        ("compare", "Modelleri karsilastir"),
                    ],
                    format_func=lambda item: item[1],
                )[0]
                model_submitted = st.form_submit_button("Model islemini calistir", width="stretch")

        if model_submitted:
            try:
                if model_action == "train":
                    with activity_status("Model islemi", "Model egitiliyor"):
                        metrics = train_and_save_model(model_type=model_type)
                    clear_runtime_caches()
                    st.session_state["system_health_report"] = build_system_health_report()
                    st.session_state["sidebar_model_feedback"] = {
                        "level": "success",
                        "text": "Model egitildi.",
                    }
                    st.session_state["sidebar_model_details"] = metrics
                elif model_action == "predict":
                    with activity_status("Model islemi", "Tahminler uretiliyor"):
                        predictions = predict_upcoming_matches()
                        live_predictions = predict_live_matches()
                        clear_runtime_caches()
                        system_health_report = build_system_health_report()
                    st.session_state["latest_prediction_output"] = predictions
                    st.session_state["latest_live_prediction_output"] = live_predictions
                    st.session_state["system_health_report"] = system_health_report
                    prediction_warnings = []
                    if getattr(predictions, "attrs", {}).get("persistence_warning"):
                        prediction_warnings.append(
                            f"Yaklasan mac tahminleri DB'ye yazilirken uyari olustu: {predictions.attrs['persistence_warning']}"
                        )
                    if getattr(live_predictions, "attrs", {}).get("persistence_warning"):
                        prediction_warnings.append(
                            f"Canli mac tahminleri DB'ye yazilirken uyari olustu: {live_predictions.attrs['persistence_warning']}"
                        )
                    recommendation_result = build_betting_recommendations(
                        predictions=predictions,
                        bankroll=float(bankroll_amount),
                        flat_stake=float(flat_stake_amount),
                        edge_threshold=float(edge_threshold),
                        coupon_size=int(coupon_size),
                        coupon_profile=coupon_profile,
                        source_filter=(
                            "iddaa" if coupon_source_mode == "Sadece IDDAA" else None
                        ),
                    )
                    if (
                        coupon_source_mode == "Sadece IDDAA"
                        and (
                            not isinstance(recommendation_result.get("recommended_bets"), pd.DataFrame)
                            or recommendation_result["recommended_bets"].empty
                        )
                    ):
                        recommendation_result["fallback_recommendations"] = build_betting_recommendations(
                            predictions=predictions,
                            bankroll=float(bankroll_amount),
                            flat_stake=float(flat_stake_amount),
                            edge_threshold=float(edge_threshold),
                            coupon_size=int(coupon_size),
                            coupon_profile=coupon_profile,
                            source_filter=None,
                        )
                    if not system_health_report.get("coupon_generation_allowed", False):
                        recommendation_result["coupon_picks"] = pd.DataFrame()
                        recommendation_result["coupon_summary"] = {}
                        recommendation_result["blocked_reason"] = system_health_report.get("summary")
                        recommendation_result["coupon_blocked"] = True
                    else:
                        recommendation_result["coupon_blocked"] = False
                    if prediction_warnings:
                        recommendation_result["prediction_warnings"] = prediction_warnings
                    st.session_state["betting_recommendations"] = recommendation_result

                    live_recommendation_result = build_betting_recommendations(
                        predictions=live_predictions,
                        bankroll=float(bankroll_amount),
                        flat_stake=float(flat_stake_amount),
                        edge_threshold=float(edge_threshold),
                        coupon_size=int(coupon_size),
                        coupon_profile=coupon_profile,
                    )
                    if not system_health_report.get("coupon_generation_allowed", False):
                        live_recommendation_result["recommended_bets"] = pd.DataFrame()
                        live_recommendation_result["coupon_picks"] = pd.DataFrame()
                        live_recommendation_result["coupon_summary"] = {}
                        live_recommendation_result["blocked_reason"] = system_health_report.get(
                            "summary"
                        )
                    st.session_state["live_betting_recommendations"] = live_recommendation_result
                    st.session_state["sidebar_model_feedback"] = {
                        "level": "success",
                        "text": f"{len(predictions)} yaklasan mac icin tahmin uretildi.",
                    }
                    st.session_state["sidebar_model_details"] = {
                        "upcoming_predictions": len(predictions),
                        "live_predictions": len(live_predictions),
                        "coupon_generation_allowed": system_health_report.get(
                            "coupon_generation_allowed", False
                        ),
                    }
                elif model_action == "backtest":
                    with activity_status("Model islemi", "Walk-forward backtest calisiyor"):
                        backtest_result = run_backtest(
                            model_type=model_type,
                            edge_threshold=float(edge_threshold),
                            initial_bankroll=float(bankroll_amount),
                            flat_stake=float(flat_stake_amount),
                        )
                        model_backtest_comparison = compare_model_backtests(
                            edge_threshold=float(edge_threshold),
                            initial_bankroll=float(bankroll_amount),
                            flat_stake=float(flat_stake_amount),
                        )
                    st.session_state["backtest_result"] = backtest_result
                    st.session_state["model_backtest_comparison"] = model_backtest_comparison
                    st.session_state["sidebar_model_feedback"] = {
                        "level": "success",
                        "text": "Backtest tamamlandi.",
                    }
                    st.session_state["sidebar_model_details"] = backtest_result.get("summary")
                elif model_action == "flat_kelly":
                    with activity_status("Model islemi", "Backtest strateji karsilastirmasi calisiyor"):
                        strategy_result = compare_backtest_strategies(
                            model_type=model_type,
                            edge_threshold=float(edge_threshold),
                            initial_bankroll=float(bankroll_amount),
                        )
                    st.session_state["backtest_strategy_comparison"] = strategy_result
                    st.session_state["sidebar_model_feedback"] = {
                        "level": "success",
                        "text": "Strateji karsilastirmasi tamamlandi.",
                    }
                    st.session_state["sidebar_model_details"] = strategy_result.get("comparison")
                elif model_action == "explain":
                    with activity_status("Model islemi", "Model aciklamalari hazirlaniyor"):
                        st.session_state["explanations"] = compute_model_explanations()
                    st.session_state["sidebar_model_feedback"] = {
                        "level": "success",
                        "text": "Model aciklamalari hazirlandi.",
                    }
                    st.session_state["sidebar_model_details"] = {
                        "method": st.session_state["explanations"].get("method"),
                        "model_name": st.session_state["explanations"].get("model_name"),
                    }
                else:
                    with activity_status("Model islemi", "Model karsilastirmasi calisiyor"):
                        st.session_state["model_comparison"] = train_and_compare_models()
                    st.session_state["sidebar_model_feedback"] = {
                        "level": "success",
                        "text": "Model karsilastirmasi hazirlandi.",
                    }
                    st.session_state["sidebar_model_details"] = st.session_state[
                        "model_comparison"
                    ].get("comparison")
            except Exception as exc:
                st.session_state["sidebar_model_feedback"] = {
                    "level": "error",
                    "text": f"Model islemi hatasi: {exc}",
                }
                st.session_state["sidebar_model_details"] = None

        with st.expander("[06] Model Sonucu", expanded=False):
            render_sidebar_feedback("sidebar_model_feedback", "sidebar_model_details")

        with st.expander("[07] Durum Ozeti", expanded=False):
            st.caption(f"Model dosyasi: {MODEL_PATH}")
            st.caption(
                f"Panel zamani: {datetime.now(APP_TIMEZONE).strftime('%d.%m.%Y %H:%M')}"
            )


    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Genel Durum",
            "Mac Merkezi",
            "Tahminler",
            "Geri Test",
            "Model Laboratuvari",
            "Yonetici",
            "Kuponlarim",
        ]
    )

    with tab1:
        if st.session_state.get("system_health_report") is None:
            st.session_state["system_health_report"] = build_system_health_report()

        report = st.session_state.get("system_health_report") or {}
        render_operation_console()

        match_counts = load_match_counts()
        df_upcoming_preview = load_match_center_preview(status_filter=["scheduled", "live"], limit=120)
        df_predictions = load_latest_predictions()
        df_source_summary = load_source_summary()
        saved_coupons_df = load_saved_coupons_dataframe()
        coupon_perf = load_saved_coupon_performance()

        model_report = report.get("model_report") or {}
        accuracy = safe_float(model_report.get("accuracy"))
        if accuracy is None:
            model_score_value = "Hazirlaniyor"
            model_score_sub = "Metrikler tamamlandiginda burada gorunecek."
        else:
            model_score_raw = max(0.0, min(9.9, 4.0 + accuracy * 8.0))
            model_score_value = f"{model_score_raw:.1f}/10"
            if model_score_raw >= 8:
                model_score_sub = "Stabilite: Yuksek"
            elif model_score_raw >= 6.5:
                model_score_sub = "Stabilite: Orta"
            else:
                model_score_sub = "Stabilite: Dikkat"

        render_dashboard_kpi_row(
            [
                {
                    "label": "Tahmin basari orani",
                    "value": pct_text(accuracy),
                    "sub": f"Log loss: {number_text(model_report.get('log_loss'))}",
                    "accent": "Backtest bazli genel gorunum",
                    "icon": "%",
                },
                {
                    "label": "Aktif tahmin sayisi",
                    "value": str(match_counts.get("upcoming", 0)),
                    "sub": f"Kayitli tahmin: {len(df_predictions)}",
                    "accent": "Programdaki canli + yaklasan mac",
                    "icon": "A",
                },
                {
                    "label": "Model guven skoru",
                    "value": model_score_value,
                    "sub": model_score_sub,
                    "accent": f"Saglik durumu: {str(report.get('status', 'hazirlaniyor')).upper()}",
                    "icon": "M",
                },
            ]
        )

        left_col, right_col = st.columns([0.66, 0.34], gap="large")

        with left_col:
            render_section_heading("Kontrol Paneli", "KP")
            render_panel_card(
                "Gelecek Tahminler",
                "Sistemin son urettigi adaylar ve veri akis ozeti.",
            )
            if not df_predictions.empty:
                next_predictions_view = df_predictions.head(8).copy()
                if "kickoff_time" in next_predictions_view.columns:
                    next_predictions_view["kickoff_time"] = format_local_datetime_series(
                        next_predictions_view["kickoff_time"]
                    )
                if "confidence" in next_predictions_view.columns:
                    next_predictions_view["confidence"] = next_predictions_view["confidence"].apply(pct_text)
                render_table(
                    next_predictions_view[
                        [
                            col
                            for col in [
                                "kickoff_time",
                                "league",
                                "home_team",
                                "away_team",
                                "predicted_label",
                                "confidence",
                            ]
                            if col in next_predictions_view.columns
                        ]
                    ],
                    height=310,
                )
            else:
                st.info("Henuz tahmin uretilmedi.")

            render_section_heading("Veri Akis Panosu", "V")
            inline_stats = [
                ("Toplam mac", str(match_counts.get("total", 0))),
                ("Tamamlanmis", str(match_counts.get("completed", 0))),
                ("Yaklasan", str(match_counts.get("upcoming", 0))),
                ("Kayitli tahmin", str(len(df_predictions))),
            ]
            st.markdown(
                "<div class='gabrela-inline-stat-grid'>"
                + "".join(
                    f"<div class='gabrela-inline-stat'><div class='gabrela-inline-stat-label'>{label}</div>"
                    f"<div class='gabrela-inline-stat-value'>{value}</div></div>"
                    for label, value in inline_stats
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Veri kaynagi dagilimi", expanded=False):
                if not df_source_summary.empty:
                    render_table(df_source_summary, height=220)
                else:
                    st.info("Kaynak ozet verisi henuz olusmadi.")
            with st.expander("Yaklasan mac onizlemesi", expanded=False):
                if not df_upcoming_preview.empty:
                    st.caption("Performans icin bu alanda varsayilan olarak son 120 yaklasan/canli kayit gosterilir.")
                    render_table(df_upcoming_preview.head(120), height=260)
                else:
                    st.info("Yaklasan mac onizlemesi yok.")

        with right_col:
            render_section_heading("Kasa ve Risk", "KR")
            render_panel_card(
                "Kasa Yonetimi",
                "Kaydedilen kuponlardan olusan finansal gorunum.",
                (
                    f"<div class='gabrela-kpi-card' style='margin:0;'>"
                    f"<div class='gabrela-kpi-label'>Gerceklesen donus</div>"
                    f"<div class='gabrela-kpi-value'>{number_text(coupon_perf.get('realized_return'))}</div>"
                    f"<div class='gabrela-kpi-accent'>Net kar / zarar: {number_text(coupon_perf.get('total_profit'))}</div>"
                    f"</div>"
                ),
            )

            reasons = report.get("reasons", []) if isinstance(report, dict) else []
            reasons_html = "".join(
                f"<div class='gabrela-note-item'>{html_lib.escape(str(reason))}</div>" for reason in reasons[:4]
            )
            if not reasons_html:
                reasons_html = "<div class='gabrela-note-item'>Aktif risk notu yok. Sistem temiz gorunuyor.</div>"
            render_panel_card(
                "Sistem Iyilestirmeleri",
                "Risk notlari ve bir sonraki teknik odaklar.",
                f"<div class='gabrela-note-list'>{reasons_html}</div>",
            )

            coupon_kpis = [
                {
                    "label": "Toplam kupon",
                    "value": str(coupon_perf.get("total_coupons", len(saved_coupons_df))),
                    "sub": f"Acik: {coupon_perf.get('open_coupons', 0)}",
                    "icon": "K",
                },
                {
                    "label": "Kazanan kupon",
                    "value": str(coupon_perf.get("won_coupons", 0)),
                    "sub": f"Kaybeden: {coupon_perf.get('lost_coupons', 0)}",
                    "icon": "W",
                },
            ]
            render_dashboard_kpi_row(coupon_kpis)

            render_section_heading("Sistem Saglik Raporu", "R")
            render_system_health_report(report)

    with tab2:
        st.subheader("Mac Merkezi")
        load_full_match_center = st.toggle(
            "Tum kayitlari yukle",
            value=False,
            key="match_center_full_toggle",
            help="Kapali konumda performans icin son 500 kayit gosterilir.",
        )
        status_options = {
            "Tum Durumlar": None,
            "Programdaki": ["scheduled"],
            "Canli": ["live"],
            "Sonuclanan": ["completed"],
            "Programda + Canli": ["scheduled", "live"],
        }
        selected_status_label = st.selectbox(
            "Durum filtresi",
            list(status_options.keys()),
            index=4,
            key="match_center_status_filter",
        )
        selected_status_filter = status_options[selected_status_label]
        df_all = (
            load_matches_dataframe(status_filter=selected_status_filter)
            if load_full_match_center
            else load_match_center_preview(status_filter=selected_status_filter, limit=500)
        )
        if not df_all.empty:
            st.caption(
                "Performans icin varsayilan olarak son 500 kayit gosterilir. Tum kayitlari yukle aciksa tablo daha yavas olabilir."
            )
            league_options = ["Tumu"] + sorted(df_all["league"].dropna().unique().tolist())
            selected_league = st.selectbox("Lig filtresi", league_options)

            view = df_all.copy()
            if selected_league != "Tumu":
                view = view[view["league"] == selected_league]
            st.caption(
                "Varsayilan gorunum iddaa bulteni gibi hizli okunur satirlari gosterir. Ham tabloyu istersen alttan acabilirsin."
            )
            render_match_center_board(view, limit=120 if not load_full_match_center else 200)
            with st.expander("Ham tabloyu goster", expanded=False):
                render_table(view.head(500), height=420)
        else:
            st.info("Veritabaninda mac yok. Once demo veriyi yukle.")

    with tab3:
        st.subheader("Model tahminleri")
        latest_prediction_output = st.session_state.get("latest_prediction_output")
        latest_live_prediction_output = st.session_state.get("latest_live_prediction_output")
        recommendation_result = st.session_state.get("betting_recommendations")
        live_recommendation_result = st.session_state.get("live_betting_recommendations")
        df_predictions = load_latest_predictions()

        subtab1, subtab2, subtab3 = st.tabs(["Bugun Oyna", "Canli Oneriler", "Gecmis Tahminler"])

        with subtab1:
            if isinstance(latest_prediction_output, pd.DataFrame) and not latest_prediction_output.empty:
                latest_prediction_view = latest_prediction_output.copy()
                latest_prediction_view["kickoff_time"] = pd.to_datetime(
                    latest_prediction_view["kickoff_time"],
                    errors="coerce",
                    utc=True,
                )
                latest_prediction_view["local_match_date"] = latest_prediction_view["kickoff_time"].apply(
                    local_calendar_date
                )
                latest_prediction_view["local_kickoff"] = latest_prediction_view["kickoff_time"].apply(
                    local_timestamp
                )
                today_cut = datetime.now(APP_TIMEZONE).date()
                now_local = datetime.now(APP_TIMEZONE)
                today_prediction_view = latest_prediction_view[
                    (latest_prediction_view["local_match_date"] == today_cut)
                    & (latest_prediction_view["local_kickoff"] >= now_local)
                ].copy()

                with st.expander("Anlik tahmin ciktilari", expanded=False):
                    st.caption("Performans icin bu tabloda ilk 200 kayit gosterilir.")
                    render_table(latest_prediction_view.head(200), height=300)
                ready_count = 0
                if "betting_ready" in today_prediction_view.columns:
                    ready_count = int(today_prediction_view["betting_ready"].fillna(False).sum())
                unknown_league_count = 0
                if "league_name" in today_prediction_view.columns:
                    unknown_league_count = int(
                        today_prediction_view["league_name"].fillna("Unknown League").eq("Unknown League").sum()
                    )
                no_odds_count = 0
                if "has_supported_market_odds" in today_prediction_view.columns:
                    no_odds_count = int(
                        (~today_prediction_view["has_supported_market_odds"].fillna(False)).sum()
                    )
                elif "has_valid_1x2_odds" in today_prediction_view.columns:
                    no_odds_count = int((~today_prediction_view["has_valid_1x2_odds"].fillna(False)).sum())
                total_today_prediction_count = len(today_prediction_view)
                qc1, qc2, qc3, qc4 = st.columns(4)
                qc1.metric("Bugun ham tahmin", total_today_prediction_count)
                qc2.metric("Bugun bahis-hazir", ready_count)
                qc3.metric("Bugun lig bilgisi eksik", unknown_league_count)
                qc4.metric("Bugun desteklenen marketi olmayan", no_odds_count)
            else:
                st.info("Anlik tahmin ciktilari henuz yok. Kenar cubugundan tahmin uret.")

            if recommendation_result:
                blocked_reason = recommendation_result.get("blocked_reason")
                coupon_blocked = bool(recommendation_result.get("coupon_blocked"))
                prediction_warnings = recommendation_result.get("prediction_warnings") or []
                fallback_recommendations = recommendation_result.get("fallback_recommendations") or {}
                match_audit = recommendation_result.get("match_audit")
                for warning_text in prediction_warnings:
                    st.warning(warning_text)
                if blocked_reason:
                    st.error(
                        "Kupon guvenlik nedeniyle bloke edildi. "
                        f"Neden: {blocked_reason}"
                    )
                    st.info(
                        "Bu blok kupon olusturmayi durdurur; tahmin listesini gizlememeli. "
                            "Asagida kupona uygun gorunen maclari ve reddedilen mac nedenlerini yine gorebilirsin."
                    )

                recommended_bets = recommendation_result.get("recommended_bets")
                if isinstance(recommended_bets, pd.DataFrame) and not recommended_bets.empty:
                    total_prediction_count = (
                        len(latest_prediction_output)
                        if isinstance(latest_prediction_output, pd.DataFrame)
                        else 0
                    )
                    eligible_count = len(recommended_bets)
                    rejected_count = (
                        int((~match_audit["is_recommended"]).sum())
                        if isinstance(match_audit, pd.DataFrame) and "is_recommended" in match_audit
                        else 0
                    )
                    recommended_bets = recommended_bets.copy()
                    recommended_bets["kickoff_time"] = pd.to_datetime(
                        recommended_bets["kickoff_time"],
                        errors="coerce",
                        utc=True,
                    )
                    recommended_bets["local_match_date"] = recommended_bets["kickoff_time"].apply(
                        local_calendar_date
                    )
                    recommended_bets["local_kickoff"] = recommended_bets["kickoff_time"].apply(
                        local_timestamp
                    )
                    today_cut = datetime.now(APP_TIMEZONE).date()
                    now_local = datetime.now(APP_TIMEZONE)
                    today_bets_all = recommended_bets[
                        (recommended_bets["local_match_date"] == today_cut)
                        & (recommended_bets["local_kickoff"] >= now_local)
                    ].copy()
                    future_bets_all = recommended_bets[
                        recommended_bets["local_match_date"] > today_cut
                    ].copy()

                    if coupon_source_mode == "Tum kaynaklar":
                        today_bets = today_bets_all
                        st.info(
                            "Bugun kupon tum kaynaklardan uretiliyor. iddaa ekraninda kupon yaparken karttaki takim, saat ve lig bilgisini birlikte kontrol et."
                        )
                    else:
                        today_bets = today_bets_all[
                            today_bets_all["source"].fillna("").str.lower() == "iddaa"
                        ].copy()
                        if not today_bets.empty:
                            st.success(
                                "Bugunku kupon sadece iddaa kaynagi bulunan maclardan olusturuldu."
                            )
                        else:
                            st.warning(
                                "Bugun iddaa kaynagindan kupona uygun mac yok. Yanlis kaynaklardan kupon onermemek icin bugun kuponu bos birakiyorum."
                            )

                    today_bets = today_bets.sort_values(
                        ["kickoff_time", "recommended_edge"],
                        ascending=[True, False],
                    ).copy()
                    future_bets = future_bets_all.sort_values(
                        ["kickoff_time", "recommended_edge"],
                        ascending=[True, False],
                    ).copy()
                    segment_lookup = build_segment_strength_lookup(
                        st.session_state.get("backtest_result", {}).get("bets")
                        if st.session_state.get("backtest_result")
                        else pd.DataFrame(),
                        min_bets=int(segment_min_bets),
                    )
                    today_bets, today_segment_meta = apply_segment_strength_filter(
                        today_bets,
                        segment_lookup,
                        target_coupon_size=int(coupon_size),
                        strong_only=bool(strong_segments_only),
                    )
                    future_bets, future_segment_meta = apply_segment_strength_filter(
                        future_bets,
                        segment_lookup,
                        target_coupon_size=int(coupon_size),
                        strong_only=bool(strong_segments_only),
                    )
                    today_bets = annotate_segment_badges(today_bets, segment_lookup)
                    future_bets = annotate_segment_badges(future_bets, segment_lookup)
                    today_eligible_count = len(today_bets)
                    future_eligible_count = len(future_bets)
                    today_favorite_count = int(
                        (today_bets.get("segment_priority", pd.Series(dtype=int)).fillna(0).astype(int) >= 2).sum()
                    ) if isinstance(today_bets, pd.DataFrame) and not today_bets.empty else 0
                    future_favorite_count = int(
                        (future_bets.get("segment_priority", pd.Series(dtype=int)).fillna(0).astype(int) >= 2).sum()
                    ) if isinstance(future_bets, pd.DataFrame) and not future_bets.empty else 0

                    today_coupon, today_summary, today_fallback_meta = build_coupon_with_smart_fallback(
                        recommended_bets=today_bets,
                        coupon_size=int(coupon_size),
                        bankroll=float(bankroll_amount),
                        coupon_profile=coupon_profile,
                        fallback_enabled=bool(temkinli_fallback_enabled),
                    )
                    today_profile_count = int(today_summary.get("bet_count", 0))

                    future_coupon = pd.DataFrame()
                    future_summary: Dict[str, Any] = {}
                    future_profile_count = 0
                    future_fallback_meta: Dict[str, Any] = {}
                    if not future_bets.empty:
                        future_coupon, future_summary, future_fallback_meta = build_coupon_with_smart_fallback(
                            recommended_bets=future_bets,
                            coupon_size=int(coupon_size),
                            bankroll=float(bankroll_amount),
                            coupon_profile=coupon_profile,
                            fallback_enabled=bool(temkinli_fallback_enabled),
                        )
                        future_profile_count = int(future_summary.get("bet_count", 0))

                    sm1, sm2, sm3, sm4 = st.columns(4)
                    sm1.metric("Toplam ham tahmin", total_prediction_count)
                    sm2.metric("Toplam filtreyi gecen", eligible_count)
                    sm3.metric("Bugun filtreyi gecen", today_eligible_count)
                    sm4.metric("Bugun kupona giren", today_profile_count)

                    sm5, sm6, sm7 = st.columns(3)
                    sm5.metric("Gelecek gun filtreyi gecen", future_eligible_count)
                    sm6.metric("Gelecek gun kupona giren", future_profile_count)
                    sm7.metric("Reddedilen mac", rejected_count)
                    st.caption(
                        "Filtreyi gecen sayisi, tek tek bakildiginda onerilebilir adaylari gosterir. "
                        "Kupona giren sayisi ise secili kupon modu, oran tavani ve lig/kalite sinirlarindan sonra gercekten karta girebilen maclari gosterir."
                    )
                    st.caption(
                        f"Favori segment rozeti, backtestte pozitif ROI + dusuk drawdown + yeterli sample veren lig x market kombinasyonlarini isaretler. "
                        f"Bugun rozetli aday: {today_favorite_count} | Gelecek gun rozetli aday: {future_favorite_count}."
                    )
                    if today_segment_meta.get("enabled"):
                        if today_segment_meta.get("strong_only"):
                            st.caption(
                                f"Kullanici filtresi aktif: sadece guclu segmentler kullaniliyor. "
                                f"Minimum bet sayisi: {int(segment_min_bets)} | Guclu havuz: {today_segment_meta.get('strong_pool', 0)}."
                            )
                        elif today_segment_meta.get("used_min_priority") == 2:
                            st.caption(
                                f"Otomatik segment filtresi aktif: Bugun kuponu, backtestte guclu gorunen lig x market segmentlerine agirlik verdi. "
                                f"Guclu segment havuzu: {today_segment_meta.get('strong_pool', 0)}."
                            )
                        elif today_segment_meta.get("used_min_priority") == 1:
                            st.caption(
                                f"Otomatik segment filtresi aktif: Guclu segment sayisi yetmedigi icin orta segmentler de dahil edildi. "
                                f"Guclu: {today_segment_meta.get('strong_pool', 0)} | Orta+Guclu: {today_segment_meta.get('medium_pool', 0)}."
                            )
                        else:
                            st.caption(
                                "Otomatik segment filtresi devrede ama bugun yeterli sayida guclu/orta segment olmadigi icin aday havuzu tamamen daraltilmadi."
                            )
                        if int(today_segment_meta.get("negative_roi_removed_count", 0)) > 0:
                            st.caption(
                                f"Negatif ROI verdigi gorulen {int(today_segment_meta.get('negative_roi_removed_count', 0))} lig x market segmenti otomatik olarak dislandi."
                            )
                        if int(today_segment_meta.get("high_drawdown_removed_count", 0)) > 0:
                            st.caption(
                                f"Yuksek drawdown verdigi gorulen {int(today_segment_meta.get('high_drawdown_removed_count', 0))} lig x market segmenti otomatik olarak dislandi."
                            )

                    st.info(
                        "Garanti kupon yok. Buradaki secimler modelin istatistiksel onerileridir; iddaa'da oynarken kartlardaki takim adi, lig, saat ve oran kaynagini birlikte kontrol et."
                    )

                    picks_col, coupon_col = st.columns([0.62, 0.38], gap="large")

                    with coupon_col:
                        render_section_heading("Karar Paneli", "KP")
                        render_panel_card(
                            "Gunun Kuponu",
                            "Secili moda gore bugun icin kalan en uygun kombinasyon bu alanda toplanir.",
                        )
                        if coupon_blocked:
                            st.warning(
                                "Otomatik kupon olusturma gecici olarak kapali. "
                                "Ama soldaki listede algoritmanin kupona uygun buldugu maclari gorebilirsin."
                            )
                        elif not today_bets.empty:
                            tek_coupon, tek_summary = today_coupon, today_summary
                            render_coupon_signal_panel(
                                title="Bugun Kupon Karari",
                                picks=tek_coupon,
                                summary=tek_summary,
                                min_pick_count=2,
                            )
                            if today_fallback_meta.get("used"):
                                st.info(
                                    f"{today_fallback_meta.get('reason')} Kupon profili gecici olarak "
                                    f"{today_fallback_meta.get('to_profile')} seviyesine cekildi."
                                )
                            if len(tek_coupon) < 2:
                                st.info(
                                    f"Bugun filtreyi gecen {today_eligible_count} mac olsa da, {coupon_profile} modunun oran/lig/kalite kurallari sonrasinda kupona girebilen mac sayisi {len(tek_coupon)} oldu."
                                )
                            tk1, tk2 = st.columns(2)
                            tk1.metric("Mac sayisi", tek_summary.get("bet_count", 0))
                            tk2.metric("Toplam oran", number_text(tek_summary.get("total_odds")))
                            tk3, tk4 = st.columns(2)
                            tk3.metric("Toplam stake", number_text(tek_summary.get("flat_total_stake")))
                            tk4.metric("Tahmini kar", number_text(tek_summary.get("flat_expected_profit")))
                            st.caption(
                                f"Kupon modu: {tek_summary.get('coupon_profile', coupon_profile)} | "
                                f"Toplam oran tavani: {number_text(tek_summary.get('max_total_odds_cap'))}"
                            )
                            render_simple_pick_cards(tek_coupon)
                            save_today_clicked = st.button(
                                "Bu kuponu Kuponlarima kaydet",
                                key="save_today_coupon_button",
                                width="stretch",
                            )
                            if save_today_clicked:
                                ok, message = save_coupon_to_library(
                                    picks=tek_coupon,
                                    summary=tek_summary,
                                    segment="Bugun",
                                    source_mode=coupon_source_mode,
                                )
                                if ok:
                                    st.success(message)
                                else:
                                    st.error(message)
                            st.info(
                                "En kolay kullanim: bu kartta yazan secim sayisi kadar maci al, her mac icin yazan 'Oyna' secimini iddaa'da aynen sec."
                            )
                        else:
                            st.warning(
                                "Bugun icin kupona uygun mac bulunamadi. Bu panel sadece bugun oynanabilecek maclar icin karar verir."
                            )
                            if not future_bets.empty:
                                show_future_coupon = st.toggle(
                                    "Bugun disi uygun kuponu goster",
                                    value=False,
                                    key="show_future_coupon_toggle",
                                )
                                if show_future_coupon:
                                    next_coupon, next_summary = future_coupon, future_summary
                                    render_coupon_signal_panel(
                                        title="Sonraki Uygun Kupon",
                                        picks=next_coupon,
                                        summary=next_summary,
                                        min_pick_count=2,
                                    )
                                    if future_fallback_meta.get("used"):
                                        st.info(
                                            f"{future_fallback_meta.get('reason')} Sonraki kupon gecici olarak "
                                            f"{future_fallback_meta.get('to_profile')} seviyesine cekildi."
                                        )
                                    render_simple_pick_cards(next_coupon)
                                    save_future_clicked = st.button(
                                        "Sonraki kuponu Kuponlarima kaydet",
                                        key="save_future_coupon_button",
                                        width="stretch",
                                    )
                                    if save_future_clicked:
                                        ok, message = save_coupon_to_library(
                                            picks=next_coupon,
                                            summary=next_summary,
                                            segment="Sonraki",
                                            source_mode=coupon_source_mode,
                                        )
                                        if ok:
                                            st.success(message)
                                        else:
                                            st.error(message)
                                else:
                                    st.caption(
                                        f"Bugun disinda kupona uygun {len(future_bets)} mac bulundu. Gormek istersen anahtari ac."
                                    )

                    with picks_col:
                        render_section_heading("Tahmin Merkezi", "TM")
                        render_panel_card(
                            "Bugun Kupona Uygun Bulunan Tahminler",
                            "Bu liste sadece bugun oynanabilecek ve algoritmanin filtrelerinden gecen maclari gosterir.",
                        )
                        st.caption(
                            "Kupon bloke olsa bile gormeye devam edebilirsin. Istersen alttaki manuel alanla kendi kuponunu olusturabilirsin."
                        )
                        if not today_bets.empty:
                            render_compact_pick_board(
                                today_bets,
                                limit=24,
                                selection_state_key="manual_coupon_selected_labels",
                            )
                        else:
                            st.info("Bugun icin filtreleri gecen mac yok.")

                        render_panel_card(
                            "Kendi Kuponunu Olustur",
                            "Listeden istedigin maclari sec, iddaa'da yaptigin kuponun toplam oran ve tutarini gir, sonucu Kuponlarima kaydet.",
                        )
                        with st.expander("Manuel kupon secimi ve kayit", expanded=not today_bets.empty):
                            st.caption(
                                "Buradan sistemin bugun icin uygun buldugu maclar arasindan kendi kuponunu secip "
                                "iddaa'da oynadigin haliyle Kuponlarima kaydedebilirsin."
                            )
                            manual_options, manual_source_df = build_manual_coupon_options(today_bets)
                            if not manual_options:
                                st.info("Bugun manuel kupon olusturmak icin secilebilir mac yok.")
                            else:
                                selected_from_rows = [
                                    label
                                    for label in st.session_state.get("manual_coupon_selected_labels", [])
                                    if label in manual_options
                                ]
                                if not selected_from_rows:
                                    selected_from_rows = list(manual_options.keys())[: min(3, len(manual_options))]
                                selected_manual_labels = st.multiselect(
                                    "Kupona eklemek istedigin maclari sec",
                                    options=list(manual_options.keys()),
                                    default=selected_from_rows,
                                    key="manual_coupon_pick_select",
                                )
                                st.session_state["manual_coupon_selected_labels"] = list(selected_manual_labels)
                                manual_coupon_df = pd.DataFrame(
                                    [manual_options[label] for label in selected_manual_labels]
                                )
                                if not manual_coupon_df.empty:
                                    manual_coupon_df["actual_odds"] = manual_coupon_df["recommended_odds"]
                                    suggested_total_odds = float(
                                        np.prod(
                                            manual_coupon_df["actual_odds"].fillna(
                                                manual_coupon_df["recommended_odds"]
                                            ).astype(float)
                                        )
                                    )
                                    suggested_total_stake = float(flat_stake_amount) * len(manual_coupon_df)

                                    mc1, mc2 = st.columns(2)
                                    manual_total_odds = mc1.number_input(
                                        "Iddaa toplam oran",
                                        min_value=1.0,
                                        value=round(suggested_total_odds, 2),
                                        step=0.01,
                                        key="manual_coupon_total_odds",
                                    )
                                    manual_total_stake = mc2.number_input(
                                        "Kupon tutari",
                                        min_value=1.0,
                                        value=round(suggested_total_stake, 2),
                                        step=1.0,
                                        key="manual_coupon_total_stake",
                                    )
                                    manual_notes = st.text_input(
                                        "Not",
                                        value="",
                                        placeholder="Orn: iddaa kuponum / favori tercih / canli destek",
                                        key="manual_coupon_notes",
                                    )

                                    preview_summary = {
                                        "bet_count": len(manual_coupon_df),
                                        "total_odds": round(float(manual_total_odds), 2),
                                        "flat_total_stake": round(float(manual_total_stake), 2),
                                        "flat_expected_profit": round(
                                            float(manual_total_stake) * float(manual_total_odds)
                                            - float(manual_total_stake),
                                            2,
                                        ),
                                    }
                                    st.caption("Manuel kupon onizlemesi")
                                    render_simple_pick_cards(manual_coupon_df)
                                    save_manual_clicked = st.button(
                                        "Secili kuponu Kuponlarima kaydet",
                                        key="save_manual_coupon_button",
                                        width="content",
                                    )
                                    if save_manual_clicked:
                                        ok, message = save_coupon_to_library(
                                            picks=manual_coupon_df,
                                            summary=preview_summary,
                                            segment="Manuel",
                                            source_mode=coupon_source_mode,
                                            actual_total_odds=float(manual_total_odds),
                                            actual_total_stake=float(manual_total_stake),
                                            is_manual=True,
                                            notes=manual_notes or None,
                                        )
                                        if ok:
                                            st.success(message)
                                        else:
                                            st.error(message)
                                else:
                                    st.info("Kaydetmek icin en az bir mac sec.")

                    with st.expander("Detayli tablolar", expanded=False):
                        st.download_button(
                            "Bugunku stake onerilerini CSV indir",
                            data=today_bets.to_csv(index=False).encode("utf-8"),
                            file_name="bugunku_stake_onerileri.csv",
                            mime="text/csv",
                        )
                        if not future_bets.empty and st.session_state.get("show_future_coupon_toggle", False):
                            st.caption("Bugunden sonraki uygun maclar")
                            render_table(future_bets.head(200), height=260)

                    explanation_candidates = today_bets.copy()
                    if not explanation_candidates.empty:
                        st.subheader("Secilen mac neden bu yone gidiyor?")
                        explanation_labels = {
                            f"{format_kickoff(row['kickoff_time'])} | {row['home_team']} - {row['away_team']}"
                            : row
                            for _, row in explanation_candidates.head(30).iterrows()
                        }
                        selected_match_label = st.selectbox(
                            "Mac karti",
                            options=list(explanation_labels.keys()),
                            key="match_explanation_select",
                        )
                        selected_match_row = pd.Series(explanation_labels[selected_match_label])
                        explanation_bundle = st.session_state.get("explanations") or {}
                        global_importance = explanation_bundle.get("global_importance")
                        driver_view = build_match_driver_view(
                            selected_match_row,
                            global_importance=global_importance
                            if isinstance(global_importance, pd.DataFrame)
                            else None,
                            limit=6,
                        )

                        e1, e2, e3, e4 = st.columns(4)
                        e1.metric("Tahmin", format_pick_label(selected_match_row))
                        e2.metric("Confidence", pct_text(selected_match_row.get("model_confidence")))
                        e3.metric("Edge", pct_text(selected_match_row.get("recommended_edge")))
                        e4.metric("Oran", number_text(selected_match_row.get("recommended_odds")))

                        st.caption(
                            "Bu kart, secilen macin temel suruculerini gosterir. "
                            "SHAP/global onem verisi hazirsa oncelik onu kullanir; degilse ham feature sinyallerini gosterir."
                        )
                        if not driver_view.empty:
                            render_table(driver_view, height=260)
                    elif not future_bets.empty and st.session_state.get("show_future_coupon_toggle", False):
                        with st.expander("Bugun disi mac aciklamasini goster", expanded=False):
                            st.subheader("Sonraki uygun mac neden bu yone gidiyor?")
                            explanation_labels = {
                                f"{format_kickoff(row['kickoff_time'])} | {row['home_team']} - {row['away_team']}"
                                : row
                                for _, row in future_bets.head(30).iterrows()
                            }
                            selected_match_label = st.selectbox(
                                "Mac karti",
                                options=list(explanation_labels.keys()),
                                key="future_match_explanation_select",
                            )
                            selected_match_row = pd.Series(explanation_labels[selected_match_label])
                            explanation_bundle = st.session_state.get("explanations") or {}
                            global_importance = explanation_bundle.get("global_importance")
                            driver_view = build_match_driver_view(
                                selected_match_row,
                                global_importance=global_importance
                                if isinstance(global_importance, pd.DataFrame)
                                else None,
                                limit=6,
                            )

                            e1, e2, e3, e4 = st.columns(4)
                            e1.metric("Tahmin", format_pick_label(selected_match_row))
                            e2.metric("Confidence", pct_text(selected_match_row.get("model_confidence")))
                            e3.metric("Edge", pct_text(selected_match_row.get("recommended_edge")))
                            e4.metric("Oran", number_text(selected_match_row.get("recommended_odds")))

                            if not driver_view.empty:
                                render_table(driver_view, height=260)
                else:
                    st.info("Bugun icin uygun kupon cikmadi.")
                    fallback_bets = fallback_recommendations.get("recommended_bets")
                    if isinstance(fallback_bets, pd.DataFrame) and not fallback_bets.empty:
                        st.warning(
                            "Sadece IDDAA modunda uygun mac bulunamadi. "
                            "Ama tum kaynaklardan alternatif uygun tahminler bulundu."
                        )
                        fallback_view = fallback_bets.copy().sort_values(
                            ["kickoff_time", "recommended_edge"],
                            ascending=[True, False],
                        )
                        st.subheader("Alternatif Uygun Tahminler")
                        render_table(fallback_view.head(200), height=320)

                if isinstance(match_audit, pd.DataFrame) and not match_audit.empty:
                    rejected_today = match_audit.copy()
                    rejected_today["kickoff_time"] = pd.to_datetime(
                        rejected_today["kickoff_time"], errors="coerce"
                    )
                    rejected_today["local_match_date"] = rejected_today["kickoff_time"].apply(
                        local_calendar_date
                    )
                    today_cut = datetime.now(APP_TIMEZONE).date()
                    rejected_today = rejected_today[
                        rejected_today["local_match_date"] == today_cut
                    ].copy()
                    rejected_today = rejected_today[~rejected_today["is_recommended"]].copy()

                    with st.expander("Bu mac neden onerilmedi?", expanded=False):
                        if rejected_today.empty:
                            st.info("Bugun icin reddedilen mac aciklamasi yok.")
                            reason_summary = (
                                match_audit[~match_audit["is_recommended"]]["reason"]
                                .value_counts()
                                .rename_axis("reason")
                                .reset_index(name="count")
                            )
                            if not reason_summary.empty:
                                st.caption("Tum yaklasan maclar icin red ozeti")
                                render_table(reason_summary, height=220)
                        else:
                            reject_view = rejected_today[
                                [
                                    "kickoff_time",
                                    "league_name",
                                    "home_team",
                                    "away_team",
                                    "source",
                                    "reason",
                                    "best_edge",
                                    "confidence",
                                ]
                            ].copy()
                            render_table(reject_view.head(200), height=320)
            else:
                st.info("Bugun Oyna kutulari icin once tahmin uret.")

        with subtab2:
            st.subheader("Canli Mac Tahminleri")
            if live_recommendation_result:
                live_bets = live_recommendation_result.get("recommended_bets")
                if isinstance(live_bets, pd.DataFrame) and not live_bets.empty:
                    live_segment_lookup = build_segment_strength_lookup(
                        st.session_state.get("backtest_result", {}).get("bets")
                        if st.session_state.get("backtest_result")
                        else pd.DataFrame(),
                        min_bets=int(segment_min_bets),
                    )
                    live_bets = annotate_segment_badges(live_bets, live_segment_lookup)
                    st.warning(
                        "Canli maclar burada gosteriliyor ama bu alan henuz gercek dakika/skor-temelli canli model degil. "
                        "Sadece pre-match modelin canli maclara uygulanmis daha temkinli bir versiyonu."
                    )
                    playable_only = st.toggle(
                        "Sadece oynanabilir marketleri goster",
                        value=True,
                        key="live_playable_only_top",
                    )
                    live_view = live_bets.copy()
                    if (
                        not playable_only
                        and isinstance(latest_live_prediction_output, pd.DataFrame)
                        and not latest_live_prediction_output.empty
                    ):
                        live_view = latest_live_prediction_output.copy()

                    live_coupon, live_summary, live_fallback_meta = build_coupon_with_smart_fallback(
                        recommended_bets=live_bets,
                        coupon_size=min(int(coupon_size), len(live_bets)),
                        bankroll=float(bankroll_amount),
                        sort_by="recommended_edge",
                        coupon_profile=coupon_profile,
                        fallback_enabled=bool(temkinli_fallback_enabled),
                    )
                    lc1, lc2, lc3 = st.columns(3)
                    lc1.metric("Canli mac sayisi", len(live_view))
                    lc2.metric("Canli kupon oran", number_text(live_summary.get("total_odds")))
                    lc3.metric("Canli duz stake", number_text(live_summary.get("flat_total_stake")))
                    st.caption(
                        f"Canli kupon secim adedi: {min(int(coupon_size), len(live_bets))}. "
                        "Istersen soldaki kaydirdan bu adedi degistirebilirsin."
                    )
                    st.caption(
                        f"Kupon modu: {live_summary.get('coupon_profile', coupon_profile)} | "
                        f"Bu mod icin toplam oran tavani: {number_text(live_summary.get('max_total_odds_cap'))}"
                    )
                    if live_fallback_meta.get("used"):
                        st.info(
                            f"{live_fallback_meta.get('reason')} Canli kupon profili gecici olarak {live_fallback_meta.get('to_profile')} seviyesine cekildi."
                        )
                    live_market_summary = build_market_mix_summary(live_bets)
                    if not live_market_summary.empty:
                        st.caption("Canli onerilerde market dagilimi")
                        render_table(live_market_summary, height=180)
                    render_simple_pick_cards(live_coupon)
                    save_live_clicked = st.button(
                        "Canli kuponu Kuponlarima kaydet",
                        key="save_live_coupon_button",
                        width="content",
                    )
                    if save_live_clicked:
                        ok, message = save_coupon_to_library(
                            picks=live_coupon,
                            summary=live_summary,
                            segment="Canli",
                            source_mode=coupon_source_mode,
                        )
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)

                    with st.expander("Canli mac detay tablosu", expanded=False):
                        render_table(live_view.head(200), height=320)
                else:
                    st.info("Su an tahmine uygun canli mac yok.")
            else:
                st.info("Canli mac tahminleri icin once tahmin uret.")

            if isinstance(latest_live_prediction_output, pd.DataFrame) and not latest_live_prediction_output.empty:
                with st.expander("Ham canli tahmin ciktilari", expanded=False):
                    render_table(latest_live_prediction_output.head(200), height=300)

        with subtab3:
            st.subheader("Kayitli tahmin gecmisi")
            if not df_predictions.empty:
                render_table(df_predictions.head(200), height=340)
                csv_bytes = df_predictions.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Tahmin gecmisini CSV indir",
                    data=csv_bytes,
                    file_name="predictions_history.csv",
                    mime="text/csv",
                )
            else:
                st.info("Henuz kayitli tahmin yok. Once modeli egitip tahmin uret.")

    with tab4:
        st.subheader("Walk-forward backtest")
        backtest_result = st.session_state.get("backtest_result")
        summary = backtest_result.get("summary") if backtest_result else None
        bets = backtest_result.get("bets") if backtest_result else None
        bankroll_curve = backtest_result.get("bankroll_curve") if backtest_result else None
        drawdown_curve = backtest_result.get("drawdown_curve") if backtest_result else None
        weekly_performance = backtest_result.get("weekly_performance") if backtest_result else None
        monthly_performance = backtest_result.get("monthly_performance") if backtest_result else None
        strategy_comparison = st.session_state.get("backtest_strategy_comparison")
        model_backtest_comparison = st.session_state.get("model_backtest_comparison")

        if summary:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Bahis adedi", summary["bets_placed"])
            c2.metric("ROI", pct_text(summary["roi"]))
            c3.metric("Ending bankroll", number_text(summary["ending_bankroll"]))
            c4.metric("Max drawdown", pct_text(summary["max_drawdown"]))
            c5.metric("Hit rate", pct_text(summary["hit_rate"]))
            market_performance = build_backtest_market_performance(bets)
            market_league_performance = build_backtest_market_league_performance(bets, min_bets=3)
            segment_strength = build_segment_strength_lookup(
                bets,
                min_bets=int(segment_min_bets),
            )

            overview_tab, period_tab, bets_tab, market_performance_tab = st.tabs(
                ["Ozet", "Periyot Analizi", "Bahis Kayitlari", "Market Performansi"]
            )

            with overview_tab:
                overview_left, overview_right = st.columns(2)
                with overview_left:
                    st.write("Bankroll egri")
                    if isinstance(bankroll_curve, pd.DataFrame) and not bankroll_curve.empty:
                        render_line_chart_dark(bankroll_curve.set_index("kickoff_time")["bankroll"], height=280)
                    else:
                        st.info("Bankroll egrisi olusmadi.")
                with overview_right:
                    st.write("Drawdown egri")
                    if isinstance(drawdown_curve, pd.DataFrame) and not drawdown_curve.empty:
                        drawdown_plot = drawdown_curve.copy()
                        drawdown_plot["drawdown"] = -drawdown_plot["drawdown"]
                        render_line_chart_dark(drawdown_plot.set_index("kickoff_time")["drawdown"], height=280, color="#ffd23f")

                        worst_row = drawdown_curve.sort_values("drawdown", ascending=False).iloc[0]
                        st.caption(
                            "En sert drawdown: "
                            f"{pct_text(worst_row['drawdown'])} - "
                            f"{pd.to_datetime(worst_row['kickoff_time']).strftime('%Y-%m-%d')}"
                        )
                    else:
                        st.info("Drawdown egrisi olusmadi.")
                with st.expander("Backtest ozeti", expanded=False):
                    st.json(summary)

            with period_tab:
                period_left, period_right = st.columns(2)
                with period_left:
                    if isinstance(weekly_performance, pd.DataFrame) and not weekly_performance.empty:
                        wc1, wc2, wc3 = st.columns(3)
                        wc1.metric("Haftalik donem", len(weekly_performance))
                        wc2.metric("Pozitif hafta", int((weekly_performance["pnl"] > 0).sum()))
                        wc3.metric("En iyi hafta ROI", pct_text(weekly_performance["roi"].max()))
                    render_period_performance("Haftalik performans", weekly_performance)
                with period_right:
                    if isinstance(monthly_performance, pd.DataFrame) and not monthly_performance.empty:
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Aylik donem", len(monthly_performance))
                        mc2.metric("Pozitif ay", int((monthly_performance["pnl"] > 0).sum()))
                        mc3.metric("En iyi ay ROI", pct_text(monthly_performance["roi"].max()))
                    render_period_performance("Aylik performans", monthly_performance)

            with bets_tab:
                if isinstance(bets, pd.DataFrame) and not bets.empty:
                    render_table(bets, height=360)
                    st.download_button(
                        "Backtest bahislerini CSV indir",
                        data=bets.to_csv(index=False).encode("utf-8"),
                        file_name="backtest_bets.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Edge esigine uyan bahis olusmadi.")

            with market_performance_tab:
                if not market_performance.empty:
                    mp1, mp2, mp3 = st.columns(3)
                    mp1.metric("Calisan market", int(len(market_performance)))
                    mp2.metric(
                        "En iyi market ROI",
                        pct_text(market_performance["roi"].max()),
                    )
                    mp3.metric(
                        "En iyi market hit rate",
                        pct_text(market_performance["hit_rate"].max()),
                    )
                    st.caption(
                        "Bu tablo, backtestte gercekten bahis acilan marketleri ayri ayri olcer. "
                        "Boylece hangi marketin daha verimli oldugunu gorebiliriz."
                    )
                    render_table(market_performance, height=260)
                    if not segment_strength.empty:
                        strong_segments = segment_strength[
                            segment_strength["segment_priority"] >= 2
                        ].copy()
                        if not strong_segments.empty:
                            st.caption(
                                f"Otomatik segment filtresinin guclu saydigi kombinasyonlar. Minimum bet sayisi: {int(segment_min_bets)}."
                            )
                            render_table(strong_segments.head(20), height=220)
                    if not market_league_performance.empty:
                        st.caption(
                            "Lig x market kirilimi. Burada her marketin hangi liglerde daha iyi veya zayif "
                            "calistigini gorebiliriz."
                        )
                        render_table(market_league_performance, height=300)
                else:
                    st.info("Market performansi icin olusan bahis kaydi yok.")
        else:
            st.info("Backtest sonuclari henuz yok. Kenar cubugundan 'Backtest calistir' butonunu kullan.")

        st.divider()
        st.subheader("Eski vs yeni model backtest")
        if model_backtest_comparison and isinstance(model_backtest_comparison.get("comparison"), pd.DataFrame):
            st.caption(
                "Burada eski temel lojistik model ile yeni varsayilan Hist Gradient Boosting ayni tarihsel akista karsilastirilir."
            )
            render_table(model_backtest_comparison["comparison"], height=220)
        else:
            st.info("Bu kiyasi gormek icin once 'Backtest calistir' kullan.")

        st.divider()
        st.subheader("Flat vs Kelly karsilastirmasi")
        if strategy_comparison:
            comparison_df = strategy_comparison.get("comparison")
            strategy_bankroll_curve = strategy_comparison.get("bankroll_curve")
            strategy_drawdown_curve = strategy_comparison.get("drawdown_curve")
            strategy_weekly = strategy_comparison.get("weekly_performance")
            strategy_monthly = strategy_comparison.get("monthly_performance")

            if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
                render_table(comparison_df, height=220)

            strat_tab1, strat_tab2 = st.tabs(["Egriler", "Donem Karsilastirmasi"])
            with strat_tab1:
                if isinstance(strategy_bankroll_curve, pd.DataFrame) and not strategy_bankroll_curve.empty:
                    curve_view = strategy_bankroll_curve.pivot_table(
                        index="kickoff_time",
                        columns="strategy",
                        values="bankroll",
                        aggfunc="last",
                    ).sort_index()
                    st.write("Bankroll karsilastirmasi")
                    render_line_chart_dark(curve_view, height=280)

                if isinstance(strategy_drawdown_curve, pd.DataFrame) and not strategy_drawdown_curve.empty:
                    drawdown_view = strategy_drawdown_curve.copy()
                    drawdown_view["drawdown"] = -drawdown_view["drawdown"]
                    drawdown_view = drawdown_view.pivot_table(
                        index="kickoff_time",
                        columns="strategy",
                        values="drawdown",
                        aggfunc="last",
                    ).sort_index()
                    st.write("Drawdown karsilastirmasi")
                    render_line_chart_dark(drawdown_view, height=280)

            with strat_tab2:
                left, right = st.columns(2)
                with left:
                    if isinstance(strategy_weekly, pd.DataFrame) and not strategy_weekly.empty:
                        st.write("Haftalik strateji ROI")
                        weekly_roi = strategy_weekly.pivot_table(
                            index="period_start",
                            columns="strategy",
                            values="roi",
                            aggfunc="last",
                        ).sort_index()
                        render_line_chart_dark(weekly_roi, height=240)
                        render_table(strategy_weekly, height=260)
                with right:
                    if isinstance(strategy_monthly, pd.DataFrame) and not strategy_monthly.empty:
                        st.write("Aylik strateji ROI")
                        monthly_roi = strategy_monthly.pivot_table(
                            index="period_start",
                            columns="strategy",
                            values="roi",
                            aggfunc="last",
                        ).sort_index()
                        render_line_chart_dark(monthly_roi, height=240)
                        render_table(strategy_monthly, height=260)
        else:
            st.info("Flat ve Kelly karsilastirmasi icin kenar cubugundan ilgili butonu kullan.")

    with tab5:
        st.subheader("Model aciklanabilirligi")
        explanations = st.session_state.get("explanations")
        model_comparison = st.session_state.get("model_comparison")
        backtest_result = st.session_state.get("backtest_result")

        if explanations:
            st.write(f"Model: {explanations['model_name']}")
            st.write(f"Yontem: {explanations['method']}")

            global_importance = explanations.get("global_importance")
            if isinstance(global_importance, pd.DataFrame) and not global_importance.empty:
                st.write("Global ozellik onemi")
                render_bar_chart_dark(global_importance.set_index("feature")["importance"], height=280, color="#ffd23f")
                render_table(global_importance, height=280)

            sample_explanation = explanations.get("sample_explanation")
            if isinstance(sample_explanation, pd.DataFrame) and not sample_explanation.empty:
                st.write("Ornek mac bazli aciklama matrisi")
                render_table(sample_explanation, height=280)
        else:
            st.info(
                "Model aciklamalari henuz hazir degil. Kenar cubugundan 'Model aciklamalarini hazirla' butonunu kullan."
            )

        st.divider()
        st.subheader("Hata analizi")
        if backtest_result and isinstance(backtest_result.get("predictions"), pd.DataFrame):
            error_views = build_error_analysis_views(backtest_result.get("predictions"))
            league_summary = error_views.get("league_summary")
            confidence_summary = error_views.get("confidence_summary")
            label_summary = error_views.get("label_summary")
            worst_rows = error_views.get("worst_rows")

            err_tab1, err_tab2, err_tab3 = st.tabs(
                ["Lig bazli hata", "Confidence bazli hata", "En pahali yanilgilar"]
            )
            with err_tab1:
                if isinstance(league_summary, pd.DataFrame) and not league_summary.empty:
                    render_table(league_summary, height=280)
                else:
                    st.info("Lig bazli hata analizi icin yeterli veri yok.")

            with err_tab2:
                left, right = st.columns(2)
                with left:
                    if isinstance(confidence_summary, pd.DataFrame) and not confidence_summary.empty:
                        render_table(confidence_summary, height=240)
                    else:
                        st.info("Confidence band analizi icin veri yok.")
                with right:
                    if isinstance(label_summary, pd.DataFrame) and not label_summary.empty:
                        render_table(label_summary, height=240)
                    else:
                        st.info("Tahmin yapi analizi icin veri yok.")

            with err_tab3:
                if isinstance(worst_rows, pd.DataFrame) and not worst_rows.empty:
                    mistake_view = worst_rows[
                        [
                            "kickoff_time",
                            "league_name",
                            "home_team",
                            "away_team",
                            "predicted_label",
                            "target_label",
                            "confidence",
                            "prob_home",
                            "prob_draw",
                            "prob_away",
                        ]
                    ].copy()
                    render_table(mistake_view, height=320)
                else:
                    st.info("Hata satiri yok.")
        else:
            st.info(
                "Hata analizi icin once backtest calistir. Bu ekran modelin en cok nerede yanildigini gosterecek."
            )

        st.divider()
        st.subheader("Market bazli analiz")
        market_frames = []
        latest_recs = st.session_state.get("betting_recommendations", {}).get("recommended_bets")
        live_recs = st.session_state.get("live_betting_recommendations", {}).get("recommended_bets")
        if isinstance(latest_recs, pd.DataFrame) and not latest_recs.empty:
            tmp = latest_recs.copy()
            tmp["segment"] = "Yaklasan"
            market_frames.append(tmp)
        if isinstance(live_recs, pd.DataFrame) and not live_recs.empty:
            tmp = live_recs.copy()
            tmp["segment"] = "Canli"
            market_frames.append(tmp)

        if market_frames:
            market_frame = pd.concat(market_frames, ignore_index=True)
            market_summary = build_market_mix_summary(market_frame)
            segment_summary = build_market_segment_summary(market_frame)
            market_backtest = build_backtest_market_performance(
                backtest_result.get("bets") if backtest_result else pd.DataFrame()
            )
            market_league_backtest = build_backtest_market_league_performance(
                backtest_result.get("bets") if backtest_result else pd.DataFrame(),
                min_bets=int(segment_min_bets),
            )
            segment_strength = build_segment_strength_lookup(
                backtest_result.get("bets") if backtest_result else pd.DataFrame(),
                min_bets=int(segment_min_bets),
            )
            market_tab1, market_tab2 = st.tabs(
                ["Market dagilimi", "Market performansi"]
            )
            with market_tab1:
                if not market_summary.empty:
                    render_table(market_summary, height=220)
                if not segment_summary.empty:
                    st.caption("Yaklasan ve canli akista market kirilimi")
                    render_table(segment_summary, height=240)
            with market_tab2:
                if not market_backtest.empty:
                    st.caption(
                        "Burada 1X2, Alt/Ust, KG, Cifte Sans ve Handikap 1:0 marketlerinin "
                        "gercek backtest sonuclari birlikte gorunur."
                    )
                    render_table(market_backtest, height=220)
                    if not segment_strength.empty:
                        strong_segments = segment_strength[
                            segment_strength["segment_priority"] >= 2
                        ].copy()
                        if not strong_segments.empty:
                            st.caption(
                                f"Karar motorunun agirlik verdigi guclu segmentler. Minimum bet sayisi: {int(segment_min_bets)}."
                            )
                            render_table(strong_segments.head(20), height=220)
                    if not market_league_backtest.empty:
                        st.caption(
                            "Lig bazli kirilim. Ornegin Alt/Ust hangi liglerde guclu, Cifte Sans nerede zayif, "
                            "buradan okuyabiliriz."
                        )
                        render_table(market_league_backtest, height=300)
                else:
                    st.info("Market performansi icin once bahis acilan bir backtest sonucu gerekli.")
        else:
            st.info("Market bazli analiz icin once tahmin uret.")

        st.divider()
        st.subheader("Model karsilastirmasi")
        if model_comparison:
            comparison_df = model_comparison.get("comparison")
            if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
                render_table(comparison_df, height=220)
        else:
            st.info("Model karsilastirmasi icin kenar cubugundan 'Modelleri karsilastir' butonunu kullan.")

    with tab6:
        st.subheader("Takim alias admin")
        alias_df = load_team_alias_dataframe()
        team_options = load_team_options()

        if not alias_df.empty:
            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("Toplam alias", len(alias_df))
            ac2.metric(
                "Manual alias",
                int((alias_df["source"].fillna("") == "manual").sum()),
            )
            ac3.metric("Etkilenen takim", int(alias_df["team_id"].nunique()))

        if team_options:
            team_map = {f"{item['team_name']} (#{item['team_id']})": item["team_id"] for item in team_options}
            with st.form("alias_add_form", clear_on_submit=True):
                selected_team_label = st.selectbox("Takim sec", options=list(team_map.keys()))
                new_alias = st.text_input("Yeni alias")
                if new_alias.strip():
                    st.caption(f"Normalize anahtar: `{normalize_team_key(new_alias)}`")
                add_alias_submitted = st.form_submit_button("Alias ekle", width="content")

            if add_alias_submitted:
                if new_alias.strip():
                    session = get_session()
                    try:
                        create_manual_team_alias(
                            session=session,
                            team_id=team_map[selected_team_label],
                            alias=new_alias,
                            source="manual",
                        )
                        session.commit()
                        st.success("Alias eklendi.")
                    except Exception as exc:
                        session.rollback()
                        st.error(f"Alias eklenemedi: {exc}")
                    finally:
                        session.close()
                else:
                    st.warning("Alias alani bos olamaz.")

        st.divider()
        st.subheader("Alias duzenle / sil")
        if not alias_df.empty and team_options:
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            alias_search = filter_col1.text_input("Alias ara", key="alias_search")
            team_filter = filter_col2.selectbox(
                "Takim filtre",
                options=["Tumu"] + sorted(alias_df["team_name"].dropna().astype(str).unique().tolist()),
                key="alias_team_filter",
            )
            source_filter = filter_col3.selectbox(
                "Kaynak filtre",
                options=["Tumu"] + sorted(alias_df["source"].fillna("bos").astype(str).unique().tolist()),
                key="alias_source_filter",
            )

            filtered_alias_df = alias_df.copy()
            if alias_search.strip():
                search_value = alias_search.strip().lower()
                filtered_alias_df = filtered_alias_df[
                    filtered_alias_df["alias"].astype(str).str.lower().str.contains(search_value)
                    | filtered_alias_df["team_name"].astype(str).str.lower().str.contains(search_value)
                ]
            if team_filter != "Tumu":
                filtered_alias_df = filtered_alias_df[
                    filtered_alias_df["team_name"].astype(str) == team_filter
                ]
            if source_filter != "Tumu":
                source_value = "" if source_filter == "bos" else source_filter
                filtered_alias_df = filtered_alias_df[
                    filtered_alias_df["source"].fillna("").astype(str) == source_value
                ]

            if filtered_alias_df.empty:
                st.info("Filtreye uyan alias kaydi yok.")
            else:
                alias_options = {
                    f"{row['alias']} -> {row['team_name']} (#{row['alias_id']})": row
                    for _, row in filtered_alias_df.iterrows()
                }
                selected_alias_label = st.selectbox(
                    "Duzenlenecek alias",
                    options=list(alias_options.keys()),
                )
                selected_alias_row = alias_options[selected_alias_label]
                team_labels = list(team_map.keys())
                current_team_id = int(selected_alias_row["team_id"])
                current_team_index = next(
                    (index for index, label in enumerate(team_labels) if team_map[label] == current_team_id),
                    0,
                )
                with st.form("alias_edit_form"):
                    edited_alias = st.text_input(
                        "Alias metni",
                        value=str(selected_alias_row["alias"]),
                    )
                    st.caption(f"Yeni normalize anahtar: `{normalize_team_key(edited_alias)}`")
                    edited_source = st.text_input(
                        "Alias source",
                        value="" if pd.isna(selected_alias_row["source"]) else str(selected_alias_row["source"]),
                    )
                    edited_team_label = st.selectbox(
                        "Hedef takim",
                        options=team_labels,
                        index=current_team_index,
                    )
                    delete_confirm = st.checkbox(
                        "Silmeden once onayla",
                        value=False,
                    )
                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        update_submitted = st.form_submit_button("Alias guncelle", width="stretch")
                    with action_col2:
                        delete_submitted = st.form_submit_button("Alias sil", width="stretch")

                if update_submitted:
                    session = get_session()
                    try:
                        update_team_alias(
                            session=session,
                            alias_id=int(selected_alias_row["alias_id"]),
                            alias=edited_alias,
                            team_id=int(team_map[edited_team_label]),
                            source=edited_source,
                        )
                        session.commit()
                        st.success("Alias guncellendi.")
                    except Exception as exc:
                        session.rollback()
                        st.error(f"Alias guncellenemedi: {exc}")
                    finally:
                        session.close()

                if delete_submitted:
                    if not delete_confirm:
                        st.warning("Silme icin once onay kutusunu isaretle.")
                    else:
                        session = get_session()
                        try:
                            delete_team_alias(session=session, alias_id=int(selected_alias_row["alias_id"]))
                            session.commit()
                            st.success("Alias silindi.")
                        except Exception as exc:
                            session.rollback()
                            st.error(f"Alias silinemedi: {exc}")
                        finally:
                            session.close()

        if not alias_df.empty:
            with st.expander("Tum alias tablosu", expanded=False):
                render_table(alias_df, height=320)
        else:
            st.info("Henuz alias kaydi yok.")

    with tab7:
        render_section_heading("Kuponlarim", "KP")
        render_panel_card(
            "Kupon Kutuphanesi",
            "Sistem kuponlarini ve senin elle sectigin kuponlari burada takip ediyoruz. Amacimiz hem butceyi hem de performans verisini tek yerde toplamak.",
        )

        saved_coupons_df = load_saved_coupons_dataframe()
        performance_summary = load_saved_coupon_performance()
        preference_summary = load_saved_coupon_preferences()
        if saved_coupons_df.empty:
            st.info("Henuz kaydedilmis kupon yok. Bugun, sonraki veya canli kupon kartindan kaydetmeye baslayabiliriz.")
        else:
            render_dashboard_kpi_row(
                [
                    {
                        "label": "Toplam kupon",
                        "value": str(performance_summary.get("total_coupons", len(saved_coupons_df))),
                        "sub": f"Acik: {performance_summary.get('open_coupons', 0)}",
                        "accent": "Takip edilen toplam kupon",
                    },
                    {
                        "label": "Net kar / zarar",
                        "value": number_text(performance_summary.get("total_profit")),
                        "sub": f"ROI: {pct_text(performance_summary.get('roi'))}",
                        "accent": f"Toplam stake: {number_text(performance_summary.get('total_staked'))}",
                    },
                    {
                        "label": "Kazanan kupon",
                        "value": str(performance_summary.get("won_coupons", 0)),
                        "sub": f"Kaybeden: {performance_summary.get('lost_coupons', 0)}",
                        "accent": "Kupon kutuphanesi genel ozeti",
                    },
                ]
            )

            library_col, detail_col = st.columns([0.44, 0.56], gap="large")

            with library_col:
                render_section_heading("Kayitli Kuponlar", "LK")
                render_panel_card(
                    "Kupon Listesi",
                    "Asagidaki liste sistem ve manuel kayitlarin tamamini bir arada gosterir.",
                )
                render_table(saved_coupons_df, height=320)

            coupon_options = {
                f"#{int(row['coupon_id'])} | {row['name']} | {str(row['status']).upper()}": int(row["coupon_id"])
                for _, row in saved_coupons_df.iterrows()
            }
            selected_coupon_label = detail_col.selectbox(
                "Incelenecek kupon",
                options=list(coupon_options.keys()),
                key="saved_coupon_select",
            )
            selected_coupon_id = coupon_options[selected_coupon_label]
            coupon_header, coupon_items_df = load_saved_coupon_items(selected_coupon_id)

            with detail_col:
                render_section_heading("Secili Kupon", "SK")
                if coupon_header:
                    render_dashboard_kpi_row(
                        [
                            {
                                "label": "Durum",
                                "value": str(coupon_header.get("status", "-")).upper(),
                                "sub": "Manuel guncellenebilir",
                            },
                            {
                                "label": "Oynanan oran",
                                "value": number_text(
                                    coupon_header.get("actual_total_odds") or coupon_header.get("total_odds")
                                ),
                                "sub": f"Stake: {number_text(coupon_header.get('actual_total_stake') or coupon_header.get('flat_total_stake'))}",
                            },
                            {
                                "label": "Gerceklesen kar",
                                "value": number_text(coupon_header.get("settled_profit")),
                                "sub": f"Beklenen: {number_text(coupon_header.get('expected_profit'))}",
                            },
                        ]
                    )

                    action_cols = st.columns(4)
                    with action_cols[0]:
                        mark_open = st.button("Acik yap", key=f"coupon_open_{selected_coupon_id}", width="stretch")
                    with action_cols[1]:
                        mark_won = st.button("Kazandi yap", key=f"coupon_won_{selected_coupon_id}", width="stretch")
                    with action_cols[2]:
                        mark_lost = st.button("Kaybetti yap", key=f"coupon_lost_{selected_coupon_id}", width="stretch")
                    with action_cols[3]:
                        settle_auto = st.button("Sonucu otomatik isle", key=f"coupon_auto_{selected_coupon_id}", width="stretch")

                    if mark_open:
                        ok, message = update_coupon_status_in_library(selected_coupon_id, "open")
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)
                    if mark_won:
                        ok, message = update_coupon_status_in_library(selected_coupon_id, "won")
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)
                    if mark_lost:
                        ok, message = update_coupon_status_in_library(selected_coupon_id, "lost")
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)
                    if settle_auto:
                        ok, message = settle_coupon_in_library(selected_coupon_id)
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)

                    summary_html = (
                        f"<div class='gabrela-note-list'>"
                        f"<div class='gabrela-note-item'>Tip: {'MANUEL' if coupon_header.get('is_manual') else 'SISTEM'}</div>"
                        f"<div class='gabrela-note-item'>Gerceklesen iade: {number_text(coupon_header.get('realized_return'))}</div>"
                        f"<div class='gabrela-note-item'>Not: {html_lib.escape(str(coupon_header.get('notes') or '-'))}</div>"
                        f"</div>"
                    )
                    render_panel_card("Secili Kupon", "Bu panel secili kuponun ozetini ve aksiyonlarini tutar.", summary_html)

                    if not coupon_items_df.empty:
                        coupon_card_df = coupon_items_df.copy()
                        render_simple_pick_cards(coupon_card_df)
                        with st.expander("Kupon detay tablosu", expanded=False):
                            render_table(coupon_items_df, height=280)

            render_section_heading("Kisisel Odak Ozeti", "KO")
            pref1, pref2, pref3 = st.columns(3)
            with pref1:
                st.caption("En cok oynadigin ligler")
                top_leagues_df = pd.DataFrame(preference_summary.get("top_leagues", []))
                if top_leagues_df.empty:
                    st.info("Henuz yeterli kupon verisi yok.")
                else:
                    render_table(top_leagues_df, height=220)
            with pref2:
                st.caption("En cok oynadigin marketler")
                top_markets_df = pd.DataFrame(preference_summary.get("top_markets", []))
                if top_markets_df.empty:
                    st.info("Henuz yeterli kupon verisi yok.")
                else:
                    render_table(top_markets_df, height=220)
            with pref3:
                st.caption("En cok secilen takimlar")
                top_teams_df = pd.DataFrame(preference_summary.get("top_teams", []))
                if top_teams_df.empty:
                    st.info("Henuz yeterli kupon verisi yok.")
                else:
                    render_table(top_teams_df, height=220)

