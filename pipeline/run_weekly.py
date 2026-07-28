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
import context as context_mod
import features as features_mod
import ingest
import load
from config import CLEAN_DIR, SEASONS
from logging_setup import log


def _resolve_season(engine, requested: int | None) -> int:
    """Which season to project.

    An explicit --season always wins. Otherwise use the newest season that actually has
    rows, not the newest season in INGEST_SEASONS - those differ whenever the configured
    range runs ahead of the data (a season that has not started, or a demo seed of an
    earlier year), and silently projecting an empty season yields zero predictions with
    no error.
    """
    if requested is not None:
        return requested
    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(season) FROM player_stats")).scalar()
    if latest is None:
        log.warning("no_stats_rows", fallback=max(SEASONS))
        return max(SEASONS)
    if int(latest) != max(SEASONS):
        log.info("season_resolved_from_data", configured=max(SEASONS), using=int(latest))
    return int(latest)


# The regular season is 18 weeks; nflverse data continues into the playoffs (weeks
# 19-22). Projecting "the week after the last one on file" therefore produced week 23,
# which does not exist - the real run wrote 530 predictions for a phantom week while the
# dashboard's week 1-18 selectors showed nothing.
REGULAR_SEASON_WEEKS = 18


def _next_week(engine, season: int) -> int:
    """The next week worth projecting, clamped to the regular season.

    Playoff weeks are in the data but are not what anyone sets a lineup for, and once the
    season is over there is no "next week" at all - in that case fall back to projecting
    week 1 of the following season, which is what the preseason model is for.
    """
    with engine.connect() as conn:
        latest = conn.execute(
            text("""
                SELECT COALESCE(MAX(week), 0) FROM player_stats
                WHERE season = :s AND week <= :cap
            """),
            {"s": season, "cap": REGULAR_SEASON_WEEKS},
        ).scalar_one()
    nxt = int(latest) + 1
    if nxt > REGULAR_SEASON_WEEKS:
        log.info(
            "regular_season_complete",
            season=season,
            hint="projecting the final week; use --week to override",
        )
        return REGULAR_SEASON_WEEKS
    return nxt


def _bust_prediction_cache() -> int:
    """The pipeline just changed the underlying data, so cached projections are stale.

    Cache invalidation lives with the writer, not the reader - the reader has no way to
    know a batch job ran.
    """
    url = os.getenv("REDIS_URL", "redis://redis:6379/0").strip()
    if not url or url.startswith("memory://"):
        # Local no-Docker mode: the API's cache lives in the API process, so a separate
        # pipeline process cannot reach it. Entries age out via TTL instead. Not a
        # problem worth solving - with a shared Redis (Docker/AWS) this branch is dead.
        log.info("cache_bust_skipped", reason="in-process cache; entries expire by TTL")
        return 0
    try:
        import redis

        client = redis.from_url(url, socket_timeout=2, decode_responses=True)
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
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to project (default: the newest season present in the database)",
    )
    parser.add_argument("--week", type=int, default=None, help="Week to project (default: next)")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse the raw parquet on disk")
    parser.add_argument(
        "--source",
        choices=["auto", "manual", "network"],
        default="auto",
        help="Where ingest gets its data: auto (default), manual (./data/manual), or network",
    )
    parser.add_argument("--skip-score", action="store_true", help="ETL only, no predictions")
    parser.add_argument(
        "--skip-context",
        action="store_true",
        help="Do not rebuild preseason context (skips the optional snap/depth downloads)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip the whole ETL and just regenerate projections from what is already "
        "in the database (used after a demo seed, or to re-score with a new model)",
    )
    args = parser.parse_args(argv)

    if args.score_only and args.skip_score:
        parser.error("--score-only and --skip-score are contradictory")

    started = time.perf_counter()
    stats: dict[str, int] = {}

    try:
        engine = load.get_engine()

        if args.score_only:
            log.info("stage_skipped", stage="etl", reason="score-only")
        else:
            if args.skip_ingest:
                log.info("stage_skipped", stage="ingest")
            else:
                log.info("stage_start", stage="ingest")
                stats |= ingest.run([args.season] if args.season else None, source=args.source)

            log.info("stage_start", stage="clean")
            stats |= clean.run()

            log.info("stage_start", stage="load")
            weekly = pd.read_parquet(CLEAN_DIR / "weekly.parquet")
            rosters = pd.read_parquet(CLEAN_DIR / "rosters.parquet")
            id_map = load.upsert_players(engine, weekly, rosters)
            stats["stats_rows"] = load.upsert_stats(engine, weekly, id_map)

        if not args.skip_context:
            # Context is what makes a week-1 or rookie projection possible. Rebuilt every
            # run because depth charts and rosters move during the season.
            target_season = _resolve_season(engine, args.season)
            log.info("stage_start", stage="context", season=target_season)
            try:
                stats["context_rows"] = context_mod.build(engine, target_season)
            except Exception as exc:  # noqa: BLE001 - optional feeds must not fail the run
                log.warning("context_build_failed", error=str(exc), error_type=type(exc).__name__)

        if not args.skip_score:
            season = _resolve_season(engine, args.season)
            week = args.week or _next_week(engine, season)
            log.info("stage_start", stage="features", season=season, week=week)
            frame = features_mod.build_upcoming(engine, season, week)
            if frame.empty:
                log.warning(
                    "no_features",
                    season=season,
                    week=week,
                    hint="no player has a prior week in this season - nothing to project",
                )
            rows = features_mod.score_upcoming(frame, season, week)
            stats["predictions"] = load.insert_predictions(engine, rows)
            _bust_prediction_cache()

        log.info("run_complete", duration_s=round(time.perf_counter() - started, 2), **stats)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.error("run_failed", error=str(exc), error_type=type(exc).__name__, exc_info=exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
