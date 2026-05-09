from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from football_analytics.audit import build_system_health_report
from football_analytics.config import LOCAL_CSV_IMPORT_DIR
from football_analytics.data_ingest import import_historical_results_from_local_csv_folder
from football_analytics.train import train_and_save_model


def main() -> None:
    target_folder = LOCAL_CSV_IMPORT_DIR
    if len(sys.argv) > 1:
        target_folder = Path(sys.argv[1]).expanduser().resolve()

    print(f"Tarama klasoru: {target_folder}")
    print("Alt klasorler dahil tum CSV dosyalari taranacak...")

    import_result = import_historical_results_from_local_csv_folder(str(target_folder))
    print(f"CSV ice aktarma sonucu: {import_result}")

    if not import_result.get("ok"):
        raise SystemExit(1)

    train_metrics = train_and_save_model(model_type="hist_gradient_boosting")
    print(f"Model metrikleri: {train_metrics}")

    health_report = build_system_health_report()
    print(f"Sistem saglik raporu: {health_report}")


if __name__ == "__main__":
    main()
