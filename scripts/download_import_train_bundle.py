from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from football_analytics.audit import build_system_health_report
from football_analytics.config import LOCAL_CSV_IMPORT_DIR
from football_analytics.data_ingest import (
    download_bulk_import_csv_bundle,
    import_historical_results_from_local_csv_folder,
)
from football_analytics.train import train_and_save_model


def main() -> None:
    print(f"CSV hedef klasoru: {LOCAL_CSV_IMPORT_DIR}")

    download_result = download_bulk_import_csv_bundle()
    print(f"CSV indirme sonucu: {download_result}")

    import_result = import_historical_results_from_local_csv_folder(str(LOCAL_CSV_IMPORT_DIR))
    print(f"CSV ice aktarma sonucu: {import_result}")

    if import_result.get("ok"):
        metrics = train_and_save_model(model_type="hist_gradient_boosting")
        print(f"Model metrikleri: {metrics}")
        health = build_system_health_report()
        print(f"Sistem saglik raporu: {health}")


if __name__ == "__main__":
    main()
