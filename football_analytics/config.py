import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    def load_dotenv(*_args, **_kwargs):
        return False

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOCAL_CSV_IMPORT_DIR = DATA_DIR / "bulk_import_csv"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOCAL_CSV_IMPORT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/football_analytics.db")

IDDAA_VERIFY_SSL = os.getenv("IDDAA_VERIFY_SSL", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

STREAMLIT_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8502"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
