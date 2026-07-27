"""Training-set construction.

The rolling features are computed in SQL with window functions rather than in pandas,
for two reasons: Postgres does the aggregation next to the data instead of shipping every
row over the wire, and the window frame (`ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING`)
makes the no-lookahead guarantee explicit and reviewable rather than an easily-broken
pandas `.shift()`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.features import FEATURE_ORDER, TARGET

ARTIFACT_DIR = Path(os.getenv("MODEL_DIR", "/models"))

FEATURE_SQL = text("""
WITH windowed AS (
    SELECT
        ps.player_id,
        ps.season,
        ps.week,
        p.position,
        ps.opponent,
        ps.is_home,
        ps.fantasy_points AS target,
        AVG(ps.targets)        OVER w AS targets_last_3,
        AVG(ps.receptions)     OVER w AS receptions_last_3,
        AVG(ps.yards)          OVER w AS yards_last_3,
        AVG(ps.touchdowns)     OVER w AS touchdowns_last_3,
        AVG(ps.fantasy_points) OVER w AS fantasy_points_last_3,
        LAG(ps.fantasy_points) OVER (
            PARTITION BY ps.player_id, ps.season ORDER BY ps.week
        ) AS fantasy_points_last_1,
        AVG(ps.fantasy_points) OVER season_to_date AS season_avg_points,
        COUNT(ps.id)           OVER season_to_date AS games_played
    FROM player_stats ps
    JOIN players p ON p.id = ps.player_id
    WHERE (:positions = '' OR p.position = ANY(string_to_array(:positions, ',')))
    WINDOW
        w AS (
            PARTITION BY ps.player_id, ps.season ORDER BY ps.week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ),
        season_to_date AS (
            PARTITION BY ps.player_id, ps.season ORDER BY ps.week
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
)
SELECT * FROM windowed
WHERE games_played >= 1   -- drop each player's first game: no history to learn from
ORDER BY season, week, player_id
""")


def _database_url() -> str:
    return (
        f"postgresql+psycopg://{os.getenv('POSTGRES_USER', 'fantasyiq')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'change_me_locally')}@"
        f"{os.getenv('POSTGRES_HOST', 'postgres')}:{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'fantasyiq')}"
    )


def _add_opponent_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Strength-of-schedule proxy: rank defences by fantasy points allowed, per season.

    Computed on the training frame only (not per-row live data) which does leak a small
    amount of full-season information into the feature. Flagged here because it is the
    kind of thing worth being asked about: the fix is a season-to-date expanding rank,
    at the cost of very noisy early-season values.
    """
    allowed = (
        df.groupby(["season", "opponent"])["target"].mean().rename("pts_allowed").reset_index()
    )
    allowed["opponent_rank"] = (
        allowed.groupby("season")["pts_allowed"].rank(ascending=False, method="first").astype(int)
    )
    out = df.merge(
        allowed[["season", "opponent", "opponent_rank"]], on=["season", "opponent"], how="left"
    )
    out["opponent_rank"] = out["opponent_rank"].fillna(16).clip(1, 32).astype(int)
    return out


def load_from_db(positions: str | None = None) -> pd.DataFrame:
    engine = create_engine(_database_url(), pool_pre_ping=True)
    positions = positions if positions is not None else os.getenv("INGEST_POSITIONS", "WR,RB,TE")
    with engine.connect() as conn:
        df = pd.read_sql(FEATURE_SQL, conn, params={"positions": positions or ""})
    engine.dispose()
    return df


def make_synthetic(n_players: int = 220, weeks: int = 17, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic season so training/CI work with no database.

    Generative story: each player has a latent talent level; weekly volume is talent plus
    noise; points are a noisy function of volume and matchup. Enough structure that a real
    model beats the mean baseline, enough noise that it cannot beat it by much - which is
    also true of real fantasy football.
    """
    rng = np.random.default_rng(seed)
    rows = []
    talent = rng.gamma(shape=3.0, scale=2.2, size=n_players)
    for pid in range(1, n_players + 1):
        t = talent[pid - 1]
        for wk in range(1, weeks + 1):
            targets = max(0.0, rng.normal(t * 1.15, 2.4))
            catch_rate = np.clip(rng.normal(0.63, 0.08), 0.2, 0.95)
            receptions = targets * catch_rate
            yards = max(0.0, receptions * rng.normal(12.0, 2.5))
            tds = rng.poisson(np.clip(t / 18.0, 0.01, 1.2))
            opp_rank = int(rng.integers(1, 33))
            matchup = 1.0 + (16.5 - opp_rank) / 90.0
            pts = max(
                0.0, (yards / 10.0 + receptions * 0.5 + tds * 6.0) * matchup + rng.normal(0, 2.0)
            )
            rows.append(
                {
                    "player_id": pid,
                    "season": 2023,
                    "week": wk,
                    "opponent": f"T{opp_rank:02d}",
                    "is_home": wk % 2 == 0,
                    "targets": targets,
                    "receptions": receptions,
                    "yards": yards,
                    "touchdowns": tds,
                    "fantasy_points": pts,
                }
            )
    raw = pd.DataFrame(rows).sort_values(["player_id", "week"])

    g = raw.groupby(["player_id", "season"])
    out = raw.assign(
        target=raw["fantasy_points"],
        targets_last_3=g["targets"].transform(lambda s: s.shift(1).rolling(3, 1).mean()),
        receptions_last_3=g["receptions"].transform(lambda s: s.shift(1).rolling(3, 1).mean()),
        yards_last_3=g["yards"].transform(lambda s: s.shift(1).rolling(3, 1).mean()),
        touchdowns_last_3=g["touchdowns"].transform(lambda s: s.shift(1).rolling(3, 1).mean()),
        fantasy_points_last_3=g["fantasy_points"].transform(
            lambda s: s.shift(1).rolling(3, 1).mean()
        ),
        fantasy_points_last_1=g["fantasy_points"].transform(lambda s: s.shift(1)),
        season_avg_points=g["fantasy_points"].transform(lambda s: s.shift(1).expanding().mean()),
        games_played=g["fantasy_points"].transform(lambda s: s.shift(1).expanding().count()),
    )
    return out[out["games_played"] >= 1].copy()


def load_dataset(allow_synthetic: bool = True) -> tuple[pd.DataFrame, str]:
    """Postgres if reachable and populated, otherwise synthetic. Returns (df, source)."""
    try:
        df = load_from_db()
        if len(df) >= 500:
            return _finalise(df), "postgres"
        print(f"[dataset] only {len(df)} rows in Postgres - falling back to synthetic")
    except Exception as exc:  # noqa: BLE001
        print(f"[dataset] Postgres unavailable ({type(exc).__name__}: {exc}) - using synthetic")
    if not allow_synthetic:
        raise RuntimeError("no training data available")
    return _finalise(make_synthetic()), "synthetic"


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_opponent_rank(df)
    df["is_home"] = df["is_home"].astype(int)
    df["games_played"] = df["games_played"].fillna(0).astype(int)
    for col in FEATURE_ORDER:
        if col not in df.columns:
            raise KeyError(f"feature '{col}' missing from dataset")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "target" in df.columns:
        # The synthetic frame carries the raw column too; drop it before renaming so the
        # target is unambiguously one column.
        df = df.drop(columns=[TARGET], errors="ignore").rename(columns={"target": TARGET})
    return df.dropna(subset=[TARGET])


def time_split(df: pd.DataFrame, holdout_weeks: int = 4):
    """Chronological split, never random.

    A random split lets week 12 leak into training and week 5 into validation, which
    inflates the score for a task that is inherently forecasting. Splitting on week is
    the only honest evaluation of "can this predict next week".
    """
    cutoff = df["week"].max() - holdout_weeks
    train = df[df["week"] <= cutoff]
    valid = df[df["week"] > cutoff]
    x_cols = list(FEATURE_ORDER)
    return (
        train[x_cols].to_numpy("float32"),
        train[TARGET].to_numpy("float32"),
        valid[x_cols].to_numpy("float32"),
        valid[TARGET].to_numpy("float32"),
        int(cutoff),
    )
