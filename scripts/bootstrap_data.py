from football_analytics.data_ingest import (
    default_training_league_codes,
    default_historical_season_starts,
    persist_historical_results_from_football_data,
    persist_upcoming_from_iddaa,
)
from football_analytics.db import Match, create_tables, get_session


def main() -> None:
    create_tables()
    print("Veritabani hazirlaniyor...")

    try:
        historical_result = persist_historical_results_from_football_data(
            league_codes=default_training_league_codes(profile="broad"),
            season_starts=default_historical_season_starts(num_seasons=8),
        )
        print(f"tarihsel ingest sonucu: {historical_result}")

        result = persist_upcoming_from_iddaa(include_highlighted=True)
        print(f"iddaa ingest sonucu: {result}")
    except Exception as exc:
        print(f"bootstrap ingest hatasi: {exc}")

    session = get_session()
    try:
        total_matches = session.query(Match).count()
        scheduled_matches = session.query(Match).filter(Match.status.in_(["scheduled", "live"])).count()
        completed_matches = session.query(Match).filter(Match.status == "completed").count()
        print(
            "Veritabani ozeti:",
            {
                "toplam_mac": total_matches,
                "yaklasan_mac": scheduled_matches,
                "tamamlanmis_mac": completed_matches,
            },
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
