import os
from pathlib import Path

DATA_DIR = Path(os.getenv("PIPELINE_DATA_DIR", "/data"))
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

SEASONS = [int(s) for s in os.getenv("INGEST_SEASONS", "2022,2023").split(",") if s.strip()]
POSITIONS = [p.strip().upper() for p in os.getenv("INGEST_POSITIONS", "WR,RB,TE").split(",")]

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:9000")
ACTIVE_MODEL_VERSION = os.getenv("ACTIVE_MODEL_VERSION", "xgboost_v1")
PREDICT_BATCH_SIZE = int(os.getenv("PREDICT_BATCH_SIZE", "200"))


def database_url() -> str:
    return (
        f"postgresql+psycopg://{os.getenv('POSTGRES_USER', 'fantasyiq')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'change_me_locally')}@"
        f"{os.getenv('POSTGRES_HOST', 'postgres')}:{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'fantasyiq')}"
    )


# PPR scoring. Kept here, not in SQL, so the scoring rules are versioned with the code
# and a league-format change is a one-line diff plus a backfill.
SCORING = {
    "receptions": 1.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "fumbles_lost": -2.0,
}
