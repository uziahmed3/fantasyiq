"""Extract: pull weekly NFL stats and roster data, land them raw as parquet.

Raw data is written to disk before any transformation. That separation means a bad
cleaning rule is re-runnable without re-downloading, and the raw files are the evidence
when a number in the dashboard looks wrong.
"""

from __future__ import annotations

import sys

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from config import POSITIONS, RAW_DIR, SEASONS
from logging_setup import log

WEEKLY_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "season",
    "week",
    "opponent_team",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "rushing_yards",
    "rushing_tds",
    "fantasy_points_ppr",
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch_weekly(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    log.info("fetch_weekly_start", seasons=seasons)
    return nfl.import_weekly_data(seasons, downcast=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch_rosters(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_seasonal_rosters(seasons)


def run(seasons: list[int] | None = None) -> dict[str, int]:
    seasons = seasons or SEASONS
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    weekly = _fetch_weekly(seasons)
    keep = [c for c in WEEKLY_COLUMNS if c in weekly.columns]
    weekly = weekly[keep]
    weekly = weekly[weekly["position"].isin(POSITIONS)]
    weekly_path = RAW_DIR / "weekly.parquet"
    weekly.to_parquet(weekly_path, index=False)

    rosters = _fetch_rosters(seasons)
    roster_cols = [
        c
        for c in ["player_id", "player_name", "position", "team", "age", "height", "weight"]
        if c in rosters.columns
    ]
    rosters = rosters[roster_cols]
    roster_path = RAW_DIR / "rosters.parquet"
    rosters.to_parquet(roster_path, index=False)

    log.info(
        "fetch_weekly_done",
        weekly_rows=len(weekly),
        roster_rows=len(rosters),
        weekly_path=str(weekly_path),
    )
    return {"weekly_rows": len(weekly), "roster_rows": len(rosters)}


if __name__ == "__main__":
    seasons = [int(a) for a in sys.argv[1:]] or None
    run(seasons)
