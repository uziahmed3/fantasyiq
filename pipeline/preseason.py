"""Project a whole upcoming season, before a single game is played.

This is the draft-board job. It runs once in the offseason (and again whenever depth
charts move), and answers "who should I take?" rather than "who do I start this week".

    context for season S   <- built from season S-1
            |
    preseason model (batch)
            |
    predictions rows at week = 0

Week 0 is a sentinel meaning "the season as a whole" rather than any particular week. It
reuses the predictions table instead of adding another one, so the audit trail, the
model_version column and the append-only history all work exactly as they do for weekly
projections - and a season projection can be compared against what actually happened
later, same as any other.

The model's target is points per game over weeks 1-4, so a season total is that rate
times the number of games. That extrapolation is stated rather than hidden: it assumes a
healthy 17-game season, which is optimistic for any individual player and is the single
biggest caveat on a season-long number.

    python -m preseason                  # project the next season automatically
    python -m preseason --season 2026
"""

from __future__ import annotations

import argparse
import sys

import httpx
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tenacity import retry, stop_after_attempt, wait_fixed

import context as context_mod
import load
from config import ACTIVE_PRESEASON_MODEL_VERSION, ML_SERVICE_URL, PREDICT_BATCH_SIZE
from logging_setup import log

# Sentinel week meaning "full season", not week zero of anything.
SEASON_WEEK = 0

# Games used to turn a per-game rate into a season total. 17 is the current schedule; a
# player who misses time will fall short, which is why confidence is surfaced alongside.
GAMES_PER_SEASON = 17

# Mirrors ml-service/app/features.PRESEASON_FEATURE_ORDER.
PRESEASON_FEATURES = (
    "prior_points_per_game",
    "prior_last4_points_per_game",
    "prior_targets_per_game",
    "prior_target_share",
    "prior_carries_per_game",
    "prior_carry_share",
    "prior_yards_per_game",
    "prior_games",
    "prior_snap_share",
    "depth_chart_rank",
    "draft_round",
    "draft_pick",
    "years_experience",
    "is_rookie",
    "age",
    "team_pass_attempts_prior",
    "qb_changed",
    "career_weighted_ppg",
    "career_weighted_targets_per_game",
    "career_weighted_carries_per_game",
    "career_weighted_target_share",
    "career_best_ppg",
    "career_seasons",
    "career_games",
    "prior_points_per_target",
    "career_points_per_target",
    "efficiency_delta",
    "qb_quality",
    "team_departed_target_share",
    "team_departed_carry_share",
    "teammate_top_target_share",
    "teammate_top_carry_share",
)

CONTEXT_SQL = text("""
SELECT
    pc.player_id, p.name, p.position, pc.team,
    pc.prior_points_per_game, pc.prior_last4_points_per_game, pc.prior_targets_per_game,
    pc.prior_target_share, pc.prior_carries_per_game, pc.prior_carry_share,
    pc.prior_yards_per_game, pc.prior_games, pc.prior_snap_share,
    pc.depth_chart_rank, pc.draft_round, pc.draft_pick, pc.years_experience,
    pc.is_rookie, pc.age, pc.team_pass_attempts_prior, pc.qb_changed,
    pc.career_weighted_ppg,
    pc.career_weighted_targets_per_game,
    pc.career_weighted_carries_per_game,
    pc.career_weighted_target_share,
    pc.career_best_ppg,
    pc.career_seasons,
    pc.career_games,
    pc.prior_points_per_target,
    pc.career_points_per_target,
    pc.efficiency_delta,
    pc.qb_quality,
    pc.team_departed_target_share,
    pc.team_departed_carry_share,
    pc.teammate_top_target_share,
    pc.teammate_top_carry_share
FROM player_context pc
JOIN players p ON p.id = pc.player_id
WHERE pc.season = :season
  -- Quarterbacks are ingested so qb_changed can detect a change of starter, but never
  -- projected: fantasy points here come from receiving and rushing only, so a QB number
  -- would silently omit passing and be badly wrong. Better to show nothing than a
  -- confidently incorrect ranking.
  AND p.position IN ('WR', 'RB', 'TE')
ORDER BY pc.prior_points_per_game DESC
""")

INSERT_SQL = text("""
INSERT INTO predictions (player_id, season, week, opponent, prediction, confidence,
                         model_version, created_at)
VALUES (:player_id, :season, :week, :opponent, :prediction, :confidence, :model_version,
        CURRENT_TIMESTAMP)
""")


def next_season(engine: Engine) -> int:
    """The season to project: the one after the latest season that has games.

    Deliberately data-driven rather than calendar-driven. In August the calendar year and
    the season to project agree; in January they do not, and a projection for a season
    that already finished is not useful to anyone.
    """
    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(season) FROM player_stats")).scalar()
    if latest is None:
        raise RuntimeError("no player_stats rows - ingest at least one season first")
    return int(latest) + 1


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def _score(rows: list[dict], model_version: str) -> tuple[list[float], list[float], str]:
    with httpx.Client(timeout=60.0, trust_env=False) as client:
        resp = client.post(
            f"{ML_SERVICE_URL}/predict/preseason/batch",
            json={"items": rows, "model_version": model_version},
        )
        resp.raise_for_status()
        body = resp.json()
    return body["predictions"], body["confidences"], body["model_version"]


def _clean(value):
    """None stays None: the preseason contract treats absence as meaningful."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def run(engine: Engine, season: int, model_version: str | None = None) -> int:
    model_version = model_version or ACTIVE_PRESEASON_MODEL_VERSION

    with engine.connect() as conn:
        df = pd.read_sql(CONTEXT_SQL, conn, params={"season": season})

    if df.empty:
        log.warning(
            "no_context_for_season",
            season=season,
            hint=f"run `python -m context --season {season}` first",
        )
        return 0

    items = [{f: _clean(row[f]) for f in PRESEASON_FEATURES} for _, row in df.iterrows()]

    predictions: list[float] = []
    confidences: list[float] = []
    resolved = model_version
    for i in range(0, len(items), PREDICT_BATCH_SIZE):
        chunk = items[i : i + PREDICT_BATCH_SIZE]
        preds, confs, resolved = _score(chunk, model_version)
        predictions.extend(preds)
        confidences.extend(confs)

    payload = [
        {
            "player_id": int(row["player_id"]),
            "season": season,
            "week": SEASON_WEEK,
            "opponent": row["team"] or "UNK",
            "prediction": round(float(pred), 2),
            "confidence": round(float(conf), 3),
            "model_version": resolved,
        }
        for (_, row), pred, conf in zip(df.iterrows(), predictions, confidences, strict=True)
    ]

    # Replace rather than append: a season projection is a current opinion, and stacking
    # every rerun would make the "latest per player" query pick an arbitrary winner.
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM predictions WHERE season = :s AND week = :w " "AND model_version = :m"
            ),
            {"s": season, "w": SEASON_WEEK, "m": resolved},
        )
        for i in range(0, len(payload), 500):
            conn.execute(INSERT_SQL, payload[i : i + 500])

    rookies = int(df["is_rookie"].fillna(0).astype(int).sum())
    log.info(
        "preseason_projections_written",
        season=season,
        players=len(payload),
        rookies=rookies,
        model_version=resolved,
        games_assumed=GAMES_PER_SEASON,
    )
    return len(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project an entire upcoming season (the draft board)"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to project (default: the season after the latest one with games)",
    )
    parser.add_argument(
        "--skip-context",
        action="store_true",
        help="Assume context for the season is already built",
    )
    parser.add_argument(
        "--no-optional-feeds",
        action="store_true",
        help="Build context without snap counts / depth charts / bios",
    )
    args = parser.parse_args(argv)

    engine = load.get_engine()
    season = args.season or next_season(engine)

    if not args.skip_context:
        log.info("building_context", season=season)
        context_mod.build(engine, season, use_optional_feeds=not args.no_optional_feeds)

    written = run(engine, season)
    if not written:
        print(f"\nNo projections written for {season}.\n")
        return 1
    print(
        f"\nProjected {written} players for the {season} season "
        f"(per-game rate x {GAMES_PER_SEASON} games for the season total).\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
