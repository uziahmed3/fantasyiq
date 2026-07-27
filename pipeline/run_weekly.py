"""Weekly orchestrator.

    ingest -> clean -> load -> features -> batch score -> write predictions -> bust cache

Each stage is importable and independently runnable, so a failure at "load" is retried
from "load" rather than re-downloading the whole season. Exit code is non-zero on
failure so the scheduler (or Airflow, or an ECS scheduled task) can alert on it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
from sqlalchemy import text

import clean
import features as features_mod
import ingest
import load
from config import CLEAN_DIR, SEASONS
from logging_setup import log


def _next_week(engine, season: int) -> int:
    with engine.connect() as conn:
        latest = conn.execute(
            text("SELECT COALESCE(MAX(week), 0) FROM player_stats WHERE season = :s"),
            {"s": season},
        ).scalar_one()
    return int(latest) + 1


def _bust_prediction_cache() -> int:
    """The pipeline just changed the underlying data, so cached projections are stale.

    Cache invalidation lives with the writer, not the reader - the reader has no way to
    know a batch job ran.
    """
    try:
        import redis

        client = redis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"), socket_timeout=2, decode_responses=True
        )
        deleted = 0
        for pattern in ("pred:v1:*", "rank:v1:*"):
            for key in client.scan_iter(match=pattern, count=500):
                deleted += client.delete(key)
        log.info("cache_busted", keys=deleted)
        return deleted
    except Exception as exc:  # noqa: BLE001
        log.warning("cache_bust_skipped", error=str(exc))
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FantasyIQ weekly ETL + scoring run")
    parser.add_argument("--season", type=int, default=max(SEASONS))
    parser.add_argument("--week", type=int, default=None, help="Week to project (default: next)")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse the raw parquet on disk")
    parser.add_argument(
        "--source",
        choices=["auto", "manual", "network"],
        default="auto",
        help="Where ingest gets its data: auto (default), manual (./data/manual), or network",
    )
    parser.add_argument("--skip-score", action="store_true", help="ETL only, no predictions")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    stats: dict[str, int] = {}

    try:
        if args.skip_ingest:
            log.info("stage_skipped", stage="ingest")
        else:
            log.info("stage_start", stage="ingest")
            stats |= ingest.run([args.season], source=args.source)

        log.info("stage_start", stage="clean")
        stats |= clean.run()

        log.info("stage_start", stage="load")
        engine = load.get_engine()
        weekly = pd.read_parquet(CLEAN_DIR / "weekly.parquet")
        rosters = pd.read_parquet(CLEAN_DIR / "rosters.parquet")
        id_map = load.upsert_players(engine, weekly, rosters)
        stats["stats_rows"] = load.upsert_stats(engine, weekly, id_map)

        if not args.skip_score:
            week = args.week or _next_week(engine, args.season)
            log.info("stage_start", stage="features", season=args.season, week=week)
            frame = features_mod.build_upcoming(engine, args.season, week)
            rows = features_mod.score_upcoming(frame, args.season, week)
            stats["predictions"] = load.insert_predictions(engine, rows)
            _bust_prediction_cache()

        log.info("run_complete", duration_s=round(time.perf_counter() - started, 2), **stats)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.error("run_failed", error=str(exc), error_type=type(exc).__name__, exc_info=exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
