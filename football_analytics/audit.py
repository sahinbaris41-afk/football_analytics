import json
from typing import Any, Dict

from sqlalchemy import func

from football_analytics.db import Match, OddsSnapshot, Prediction, get_session
from football_analytics.features import valid_1x2_odds
from football_analytics.train import REPORT_PATH


def load_model_report() -> Dict[str, Any] | None:
    if not REPORT_PATH.exists():
        return None
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def evaluate_model_report(report: Dict[str, Any] | None) -> Dict[str, Any]:
    if not report:
        return {
            "status": "fail",
            "coupon_generation_allowed": False,
            "reasons": ["Model raporu yok. Once modeli saglikli sekilde egitmek gerekiyor."],
            "summary": "Model raporu bulunamadi.",
        }

    reasons = []
    status = "pass"

    rows_total = int(report.get("rows_total", 0) or 0)
    accuracy = float(report.get("accuracy", 0.0) or 0.0)
    log_loss = float(report.get("log_loss", 99.0) or 99.0)
    cv_log_loss_mean = float(report.get("cv_log_loss_mean", 99.0) or 99.0)
    brier_score = float(report.get("brier_score", 99.0) or 99.0)

    if rows_total < 250:
        status = "fail"
        reasons.append(f"Egitim verisi cok az ({rows_total} mac).")
    if accuracy < 0.48:
        status = "fail"
        reasons.append(f"Dogruluk dusuk ({accuracy:.2%}).")
    if cv_log_loss_mean > 1.4:
        status = "fail"
        reasons.append(f"CV log-loss zayif ({cv_log_loss_mean:.3f}).")
    if brier_score > 0.7:
        status = "fail"
        reasons.append(f"Brier skoru zayif ({brier_score:.3f}).")
    if status != "fail" and log_loss > 1.2:
        status = "caution"
        reasons.append(f"Log-loss yuksek ({log_loss:.3f}).")

    if not reasons:
        reasons.append("Model metrikleri kabul edilebilir aralikta.")

    return {
        "status": status,
        "coupon_generation_allowed": status == "pass",
        "reasons": reasons,
        "summary": " | ".join(reasons),
    }


def build_data_quality_summary() -> Dict[str, Any]:
    session = get_session()
    try:
        total_matches = int(session.query(func.count(Match.id)).scalar() or 0)
        completed_matches = int(
            session.query(func.count(Match.id)).filter(Match.status == "completed").scalar() or 0
        )
        scheduled_matches = int(
            session.query(func.count(Match.id)).filter(Match.status == "scheduled").scalar() or 0
        )
        live_matches = int(
            session.query(func.count(Match.id)).filter(Match.status == "live").scalar() or 0
        )
        prediction_count = int(session.query(func.count(Prediction.id)).scalar() or 0)

        odds_rows = (
            session.query(
                OddsSnapshot.home_win_odds,
                OddsSnapshot.draw_odds,
                OddsSnapshot.away_win_odds,
            )
            .filter(OddsSnapshot.market == "1X2")
            .all()
        )
        invalid_1x2_count = 0
        for home_win_odds, draw_odds, away_win_odds in odds_rows:
            if not valid_1x2_odds(home_win_odds, draw_odds, away_win_odds):
                invalid_1x2_count += 1

        active_odds_rows = (
            session.query(
                OddsSnapshot.home_win_odds,
                OddsSnapshot.draw_odds,
                OddsSnapshot.away_win_odds,
            )
            .join(Match, Match.id == OddsSnapshot.match_id)
            .filter(
                OddsSnapshot.market == "1X2",
                Match.status.in_(["scheduled", "live"]),
            )
            .all()
        )
        invalid_active_1x2_count = 0
        for home_win_odds, draw_odds, away_win_odds in active_odds_rows:
            if not valid_1x2_odds(home_win_odds, draw_odds, away_win_odds):
                invalid_active_1x2_count += 1

        source_rows = session.query(Match.source, func.count(Match.id)).group_by(Match.source).all()
        source_summary = {str(source or "unknown"): int(count) for source, count in source_rows}

        return {
            "total_matches": total_matches,
            "completed_matches": completed_matches,
            "scheduled_matches": scheduled_matches,
            "live_matches": live_matches,
            "prediction_count": prediction_count,
            "odds_snapshot_count": len(odds_rows),
            "invalid_1x2_odds_count": invalid_1x2_count,
            "invalid_active_1x2_odds_count": invalid_active_1x2_count,
            "source_summary": source_summary,
        }
    finally:
        session.close()


def build_system_health_report() -> Dict[str, Any]:
    model_report = load_model_report()
    model_health = evaluate_model_report(model_report)
    data_quality = build_data_quality_summary()

    reasons = list(model_health["reasons"])
    status = model_health["status"]

    invalid_active_1x2_count = int(data_quality.get("invalid_active_1x2_odds_count", 0) or 0)
    invalid_1x2_count = int(data_quality.get("invalid_1x2_odds_count", 0) or 0)

    if invalid_active_1x2_count > 0:
        reasons.append(
            f"Aktif maclarda supheli 1X2 oran kaydi var ({invalid_active_1x2_count} adet)."
        )
        if status == "pass":
            status = "caution"
    elif invalid_1x2_count > 0:
        reasons.append(
            f"Arsivde supheli 1X2 oran kaydi var ({invalid_1x2_count} adet)."
        )
        if status == "pass":
            status = "caution"

    return {
        "status": status,
        "coupon_generation_allowed": model_health["coupon_generation_allowed"],
        "summary": " | ".join(reasons),
        "reasons": reasons,
        "model_report": model_report,
        "model_health": model_health,
        "data_quality": data_quality,
    }
