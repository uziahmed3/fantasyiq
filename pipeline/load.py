"""Load: idempotent upserts into Postgres.

Everything here is written so the weekly job can be re-run any number of times without
duplicating or corrupting data - the single most important property of a scheduled ETL,
because it will be re-run (retries, backfills, someone fixing a bug on a Tuesday night).

Mechanism: INSERT ... ON CONFLICT DO UPDATE against the natural keys
(players.external_id, player_stats(player_id, season, week)).

The SQL is deliberately standard rather than Postgres-flavoured (CURRENT_TIMESTAMP, not
NOW(); no LEAST/GREATEST) so the identical statements run against SQLite in local
no-Docker mode. SQLite has supported upsert since 3.24 and window functions since 3.25.
"""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import database_url
from logging_setup import log

UPSERT_PLAYER = text("""
INSERT INTO players (external_id, name, team, position, age, height_inches, weight_lbs,
                     created_at, updated_at)
VALUES (:external_id, :name, :team, :position, :age, :height_inches, :weight_lbs,
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (external_id) DO UPDATE SET
    name          = EXCLUDED.name,
    team          = EXCLUDED.team,
    position      = EXCLUDED.position,
    age           = COALESCE(EXCLUDED.age, players.age),
    height_inches = COALESCE(EXCLUDED.height_inches, players.height_inches),
    weight_lbs    = COALESCE(EXCLUDED.weight_lbs, players.weight_lbs),
    updated_at    = CURRENT_TIMESTAMP
RETURNING id
""")

UPSERT_STATS = text("""
INSERT INTO player_stats (player_id, season, week, opponent, is_home, targets, receptions,
                          yards, touchdowns, fantasy_points, created_at, updated_at)
VALUES (:player_id, :season, :week, :opponent, :is_home, :targets, :receptions,
        :yards, :touchdowns, :fantasy_points, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (player_id, season, week) DO UPDATE SET
    opponent       = EXCLUDED.opponent,
    is_home        = EXCLUDED.is_home,
    targets        = EXCLUDED.targets,
    receptions     = EXCLUDED.receptions,
    yards          = EXCLUDED.yards,
    touchdowns     = EXCLUDED.touchdowns,
    fantasy_points = EXCLUDED.fantasy_points,
    updated_at     = CURRENT_TIMESTAMP
""")

INSERT_PREDICTION = text("""
INSERT INTO predictions (player_id, season, week, opponent, prediction, confidence,
                         model_version, created_at)
VALUES (:player_id, :season, :week, :opponent, :prediction, :confidence, :model_version,
        CURRENT_TIMESTAMP)
""")


def get_engine() -> Engine:
    return create_engine(database_url(), pool_pre_ping=True, future=True)


def _clean_int(value) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def upsert_players(engine: Engine, weekly: pd.DataFrame, rosters: pd.DataFrame) -> dict[str, int]:
    """Returns external_id -> players.id, so the stats load never re-queries per row."""
    roster_lookup = rosters.set_index("player_id").to_dict("index") if len(rosters) else {}
    identities = weekly.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")[
        ["player_id", "name", "team", "position"]
    ]

    id_map: dict[str, int] = {}
    with engine.begin() as conn:
        for row in identities.itertuples(index=False):
            meta = roster_lookup.get(row.player_id, {})
            pid = conn.execute(
                UPSERT_PLAYER,
                {
                    "external_id": str(row.player_id),
                    "name": row.name,
                    "team": row.team,
                    "position": row.position,
                    "age": _clean_int(meta.get("age")),
                    "height_inches": _clean_int(meta.get("height_inches")),
                    "weight_lbs": _clean_int(meta.get("weight_lbs")),
                },
            ).scalar_one()
            id_map[str(row.player_id)] = int(pid)

    log.info("upsert_players", players=len(id_map))
    return id_map


def upsert_stats(engine: Engine, weekly: pd.DataFrame, id_map: dict[str, int]) -> int:
    payload = [
        {
            "player_id": id_map[str(r.player_id)],
            "season": int(r.season),
            "week": int(r.week),
            "opponent": r.opponent,
            "is_home": bool(r.is_home),
            "targets": int(r.targets),
            "receptions": int(r.receptions),
            "yards": float(r.yards),
            "touchdowns": int(r.touchdowns),
            "fantasy_points": float(r.fantasy_points),
        }
        for r in weekly.itertuples(index=False)
        if str(r.player_id) in id_map
    ]
    # One transaction, one round trip per 1000 rows - not one INSERT per row.
    with engine.begin() as conn:
        for i in range(0, len(payload), 1000):
            conn.execute(UPSERT_STATS, payload[i : i + 1000])
    log.info("upsert_stats", rows=len(payload))
    return len(payload)


def insert_predictions(engine: Engine, rows: list[dict]) -> int:
    if not rows:
        return 0
    with engine.begin() as conn:
        for i in range(0, len(rows), 1000):
            conn.execute(INSERT_PREDICTION, rows[i : i + 1000])
    log.info("insert_predictions", rows=len(rows))
    return len(rows)
