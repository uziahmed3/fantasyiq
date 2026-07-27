"""Airflow DAG - the production-scale alternative to scheduler.py.

Worth using once you want per-stage retries, backfills over historical weeks
(`airflow dags backfill`), and a UI showing which stage failed. Each task is the same
importable function the CLI uses, so there is no duplicated pipeline logic.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable

DEFAULT_ARGS = {
    "owner": "fantasyiq",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
}


@dag(
    dag_id="fantasyiq_weekly",
    description="Ingest NFL weekly stats, refresh features, regenerate projections",
    schedule="0 9 * * 2",  # Tuesdays 09:00 UTC
    start_date=pendulum.datetime(2023, 9, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["etl", "ml", "fantasyiq"],
)
def fantasyiq_weekly():
    @task
    def extract() -> dict:
        import ingest

        return ingest.run([int(Variable.get("fantasyiq_season", default_var="2023"))])

    @task
    def transform(_upstream: dict) -> dict:
        import clean

        return clean.run()

    @task
    def load_to_postgres(_upstream: dict) -> int:
        import pandas as pd

        import load
        from config import CLEAN_DIR

        engine = load.get_engine()
        weekly = pd.read_parquet(CLEAN_DIR / "weekly.parquet")
        rosters = pd.read_parquet(CLEAN_DIR / "rosters.parquet")
        id_map = load.upsert_players(engine, weekly, rosters)
        return load.upsert_stats(engine, weekly, id_map)

    @task
    def score_week(_rows: int) -> int:
        from sqlalchemy import text

        import features
        import load

        season = int(Variable.get("fantasyiq_season", default_var="2023"))
        engine = load.get_engine()
        with engine.connect() as conn:
            week = int(
                conn.execute(
                    text("SELECT COALESCE(MAX(week),0)+1 FROM player_stats WHERE season=:s"),
                    {"s": season},
                ).scalar_one()
            )
        frame = features.build_upcoming(engine, season, week)
        return load.insert_predictions(engine, features.score_upcoming(frame, season, week))

    @task
    def invalidate_cache(_written: int) -> int:
        import run_weekly

        return run_weekly._bust_prediction_cache()

    invalidate_cache(score_week(load_to_postgres(transform(extract()))))


dag_instance = fantasyiq_weekly()
