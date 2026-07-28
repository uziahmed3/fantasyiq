"""Transform: normalise the raw feed into the shape of the relational schema.

Rules applied here, each of which exists because the raw feed violates it somewhere:
  * a player_id + season + week appears at most once (the source occasionally duplicates)
  * counting stats are non-negative integers, yardage is float, nulls become 0
  * fantasy points are recomputed from components under our own scoring config rather
    than trusted from the feed, so the number is reproducible and league-format agnostic
  * rows with no team or no position are dropped (unjoinable)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CLEAN_DIR, RAW_DIR, SCORING
from logging_setup import log

COUNT_COLS = ["targets", "receptions", "receiving_tds", "rushing_tds"]
FLOAT_COLS = ["receiving_yards", "rushing_yards"]


def compute_fantasy_points(df: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for column, weight in SCORING.items():
        if column in df.columns:
            total = total + df[column].fillna(0) * weight
    return total.round(2)


def clean_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    df = weekly.copy()
    before = len(df)

    df = df.dropna(subset=["player_id", "season", "week"])
    df = df[df["position"].notna() & df["recent_team"].notna()]

    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).clip(lower=0).round().astype(int)
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)

    # Deterministic dedupe: keep the row with the most receiving yards (the fuller record).
    sort_col = "receiving_yards" if "receiving_yards" in df.columns else "receptions"
    df = df.sort_values(["player_id", "season", "week", sort_col]).drop_duplicates(
        ["player_id", "season", "week"], keep="last"
    )

    df["fantasy_points"] = compute_fantasy_points(df)
    df["yards"] = df.get("receiving_yards", 0.0) + df.get("rushing_yards", 0.0)
    df["touchdowns"] = df.get("receiving_tds", 0) + df.get("rushing_tds", 0)
    df["opponent"] = df["opponent_team"].fillna("UNK").astype(str).str.upper().str.slice(0, 8)
    # nfl_data_py does not expose home/away on the weekly frame; a schedule join would
    # fill this properly. Defaulted rather than fabricated, and flagged in the README.
    df["is_home"] = True
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    out = df[
        [
            "player_id",
            "player_display_name",
            "position",
            "recent_team",
            "season",
            "week",
            "opponent",
            "is_home",
            "targets",
            "receptions",
            "yards",
            "touchdowns",
            "fantasy_points",
        ]
    ].rename(columns={"player_display_name": "name", "recent_team": "team"})

    log.info("clean_weekly", rows_in=before, rows_out=len(out), dropped=before - len(out))
    return out.reset_index(drop=True)


def clean_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    df = rosters.copy().dropna(subset=["player_id"])
    df = df.drop_duplicates("player_id", keep="last")
    rename = {"player_name": "name", "height": "height_inches", "weight": "weight_lbs"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Draft columns included: they are how a rookie gets projected at all.
    for col in [
        "age",
        "height_inches",
        "weight_lbs",
        "draft_round",
        "draft_pick",
        "rookie_season",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.reset_index(drop=True)


def run() -> dict[str, int]:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    weekly = clean_weekly(pd.read_parquet(RAW_DIR / "weekly.parquet"))
    rosters = clean_rosters(pd.read_parquet(RAW_DIR / "rosters.parquet"))
    weekly.to_parquet(CLEAN_DIR / "weekly.parquet", index=False)
    rosters.to_parquet(CLEAN_DIR / "rosters.parquet", index=False)
    return {"weekly_rows": len(weekly), "roster_rows": len(rosters)}


if __name__ == "__main__":
    run()
