from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from football_analytics.audit import build_system_health_report
from football_analytics.data_ingest import (
    default_historical_season_starts,
    default_training_league_codes,
    persist_historical_results_from_footballcsv_cache,
    persist_historical_results_from_football_data,
)
from football_analytics.db import Match, create_tables, get_session
from football_analytics.train import train_and_save_model


def main() -> None:
    create_tables()
    print("Genis tarihsel veri backfill basliyor...")

    historical_result = persist_historical_results_from_football_data(
        league_codes=default_training_league_codes(profile="broad"),
        season_starts=default_historical_season_starts(num_seasons=8),
    )
    print(f"football-data sonucu: {historical_result}")

    mirror_result = persist_historical_results_from_footballcsv_cache(
        league_codes=default_training_league_codes(profile="broad"),
        season_starts=default_historical_season_starts(num_seasons=8),
    )
    print(f"footballcsv mirror sonucu: {mirror_result}")

    session = get_session()
    try:
        completed_matches = session.query(Match).filter(Match.status == "completed").count()
        print(f"Tamamlanmis mac sayisi: {completed_matches}")
    finally:
        session.close()

    metrics = train_and_save_model(model_type="hist_gradient_boosting")
    print(f"Model metrikleri: {metrics}")

    health = build_system_health_report()
    print(f"Sistem saglik raporu: {health}")


if __name__ == "__main__":
    main()
