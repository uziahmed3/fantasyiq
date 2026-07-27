"""Feature engineering + batch scoring for the upcoming week.

This is the step that makes GET /rankings a pure indexed read: every eligible player is
scored once here, in one batched call to the ML service, and the results are written to
the predictions table. Serving 5,000 dashboard loads then costs zero inferences.
"""

from __future__ import annotations

import httpx
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tenacity import retry, stop_after_attempt, wait_fixed

from config import ACTIVE_MODEL_VERSION, ML_SERVICE_URL, PREDICT_BATCH_SIZE
from logging_setup import log

FEATURE_ORDER = (
    "targets_last_3",
    "receptions_last_3",
    "yards_last_3",
    "touchdowns_last_3",
    "fantasy_points_last_3",
    "fantasy_points_last_1",
    "season_avg_points",
    "games_played",
    "opponent_rank",
    "is_home",
)

# Same window semantics as ml-service/train/dataset.py: strictly prior weeks only.
# Portable SQL: no LEAST/GREATEST, no Postgres-only functions, so this runs unchanged
# against SQLite in local no-Docker mode. FILTER and window functions are supported by
# SQLite 3.30+ (shipped with every current Python) and by Postgres.
UPCOMING_FEATURES_SQL = text("""
WITH ranked AS (
    SELECT
        ps.player_id,
        ps.week,
        ps.targets, ps.receptions, ps.yards, ps.touchdowns, ps.fantasy_points,
        ROW_NUMBER() OVER (PARTITION BY ps.player_id ORDER BY ps.week DESC) AS recency
    FROM player_stats ps
    WHERE ps.season = :season AND ps.week < :week
),
rolling AS (
    SELECT
        player_id,
        AVG(targets)        FILTER (WHERE recency <= 3) AS targets_last_3,
        AVG(receptions)     FILTER (WHERE recency <= 3) AS receptions_last_3,
        AVG(yards)          FILTER (WHERE recency <= 3) AS yards_last_3,
        AVG(touchdowns)     FILTER (WHERE recency <= 3) AS touchdowns_last_3,
        AVG(fantasy_points) FILTER (WHERE recency <= 3) AS fantasy_points_last_3,
        MAX(fantasy_points) FILTER (WHERE recency  = 1) AS fantasy_points_last_1,
        AVG(fantasy_points)                             AS season_avg_points,
        COUNT(*)                                        AS games_played
    FROM ranked
    GROUP BY player_id
),
defence AS (
    SELECT
        opponent,
        RANK() OVER (ORDER BY AVG(fantasy_points) DESC) AS opponent_rank
    FROM player_stats
    WHERE season = :season AND week < :week AND opponent IS NOT NULL
    GROUP BY opponent
)
SELECT
    p.id AS player_id, p.name, p.position, p.team,
    COALESCE(r.targets_last_3, 0)        AS targets_last_3,
    COALESCE(r.receptions_last_3, 0)     AS receptions_last_3,
    COALESCE(r.yards_last_3, 0)          AS yards_last_3,
    COALESCE(r.touchdowns_last_3, 0)     AS touchdowns_last_3,
    COALESCE(r.fantasy_points_last_3, 0) AS fantasy_points_last_3,
    COALESCE(r.fantasy_points_last_1, 0) AS fantasy_points_last_1,
    COALESCE(r.season_avg_points, 0)     AS season_avg_points,
    COALESCE(r.games_played, 0)          AS games_played,
    -- clamped to 1..32 in pandas: LEAST/GREATEST do not exist in SQLite
    COALESCE(d.opponent_rank, 16)        AS opponent_rank,
    1 AS is_home
FROM players p
JOIN rolling r ON r.player_id = p.id
LEFT JOIN defence d ON d.opponent = p.team
WHERE r.games_played >= 1
ORDER BY r.season_avg_points DESC
""")


def build_upcoming(engine: Engine, season: int, week: int) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(UPCOMING_FEATURES_SQL, conn, params={"season": season, "week": week})
    for col in FEATURE_ORDER:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["games_played"] = df["games_played"].astype(int)
    # Clamp here rather than in SQL: LEAST/GREATEST do not exist in SQLite, and the
    # ML service rejects opponent_rank outside 1..32 at the contract boundary.
    df["opponent_rank"] = df["opponent_rank"].clip(1, 32).astype(int)
    df["is_home"] = df["is_home"].astype(int)
    log.info("features_built", season=season, week=week, players=len(df))
    return df


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def _score_batch(rows: list[dict], model_version: str) -> tuple[list[float], str]:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{ML_SERVICE_URL}/predict/batch",
            json={"items": rows, "model_version": model_version},
        )
        resp.raise_for_status()
        body = resp.json()
    return body["predictions"], body["model_version"]


def score_upcoming(
    df: pd.DataFrame, season: int, week: int, model_version: str | None = None
) -> list[dict]:
    if df.empty:
        return []
    model_version = model_version or ACTIVE_MODEL_VERSION
    items = df[list(FEATURE_ORDER)].to_dict("records")

    predictions: list[float] = []
    resolved = model_version
    for i in range(0, len(items), PREDICT_BATCH_SIZE):
        batch, resolved = _score_batch(items[i : i + PREDICT_BATCH_SIZE], model_version)
        predictions.extend(batch)

    out = []
    for (_, row), pred in zip(df.iterrows(), predictions, strict=True):
        out.append(
            {
                "player_id": int(row["player_id"]),
                "season": season,
                "week": week,
                "opponent": row.get("team") or "UNK",
                "prediction": round(float(pred), 2),
                # Pipeline-written rows carry no per-row confidence; the API's /predict
                # path is where the calibrated-ish score is attached.
                "confidence": None,
                "model_version": resolved,
            }
        )
    log.info("scored_upcoming", rows=len(out), model_version=resolved)
    return out
