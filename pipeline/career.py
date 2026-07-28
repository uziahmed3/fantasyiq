"""Career history, volume/efficiency separation, and situation.

## Why this module exists

The model used to see one prior season, so one bad year erased a career. Justin Jefferson
went 19.5, 21.5, 20.4, 18.6, then 11.9 - and projected 11.91.

The diagnosis matters more than the number. In his down year his target share was still
0.315, the highest on his team, on 141 targets. His role never shrank. What collapsed was
efficiency: 2.06 points per target became 1.43. And efficiency is the least repeatable
thing in football, while volume is one of the most - measured on 2021-2025, a player's
target rate carries ~0.83 correlation from one season to the next.

So this module does three things:

1. CAREER HISTORY. Every past season is weighted by three factors multiplied together:
     recency      - heavy emphasis on the last three years, decaying after
     games played - a 10-game season is a noisier estimate of a per-game rate, so it
                    counts less. This is the injury adjustment.
     age          - each past season is translated to what that performance is worth at
                    next season's age, via the age curve.

2. VOLUME vs EFFICIENCY. Volume comes from last season, because it is stable. Efficiency
   comes from the career weighted mean, because it is not. `efficiency_delta` records the
   gap, which is the bounce-back signal: a large negative value means the role held and
   the finishing broke.

3. SITUATION. Not just "did the quarterback change" but how good he is; how much target
   and carry share left the roster (someone inherits it); and who is still there competing
   for it.

Everything is computed from data already in the database - no new downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from logging_setup import log

REGULAR_SEASON_WEEKS = 18
FULL_SEASON_GAMES = 17

# Recency weights by "seasons ago" (0 = last season). Front-loaded on the last three, as
# requested, with a long thin tail so a genuine multi-year peak still registers.
RECENCY_WEIGHTS = {0: 0.45, 1: 0.28, 2: 0.15}
TAIL_WEIGHT = 0.06  # each season beyond three years ago

# A season needs at least this many games before its rate is trusted at all.
MIN_GAMES_FOR_RATE = 3

# Age curve. Skill-position production rises into the mid-twenties and declines after;
# the peak and steepness here are deliberately gentle, because the model also gets raw
# age as a feature and can learn a sharper shape from the data if one exists.
PEAK_AGE = 26.5
AGE_DECAY = 0.010


def age_multiplier(age: float | None) -> float:
    """Relative production expected at a given age, peaking at PEAK_AGE."""
    if age is None or not np.isfinite(age):
        return 1.0
    return float(max(0.55, 1.0 - AGE_DECAY * (float(age) - PEAK_AGE) ** 2))


def _season_weight(seasons_ago: int, games: int) -> float:
    """recency x games-played. The injury adjustment lives in the second term."""
    recency = RECENCY_WEIGHTS.get(seasons_ago, TAIL_WEIGHT)
    # Cap at 1.0 so a 17-game season is the reference, not a bonus.
    completeness = min(games / FULL_SEASON_GAMES, 1.0)
    return recency * completeness


HISTORY_SQL = text("""
SELECT
    ps.player_id,
    ps.season,
    COUNT(*)                  AS games,
    AVG(ps.fantasy_points)    AS ppg,
    AVG(ps.targets)           AS targets_per_game,
    AVG(ps.carries)           AS carries_per_game,
    SUM(ps.targets)           AS targets,
    SUM(ps.carries)           AS carries,
    SUM(ps.fantasy_points)    AS points
FROM player_stats ps
WHERE ps.season < :season AND ps.week <= :max_week
GROUP BY ps.player_id, ps.season
HAVING COUNT(*) >= :min_games
""")

TEAM_TOTALS_SQL = text("""
SELECT p.team AS team, ps.season AS season,
       SUM(ps.targets) AS team_targets, SUM(ps.carries) AS team_carries
FROM player_stats ps
JOIN players p ON p.id = ps.player_id
WHERE ps.season < :season AND ps.week <= :max_week AND p.team IS NOT NULL
GROUP BY p.team, ps.season
""")

AGES_SQL = text("SELECT id AS player_id, age, team, position FROM players")


def build_career(engine: Engine, season: int) -> pd.DataFrame:
    """Career aggregates for every player, as of the start of `season`.

    Returns one row per player_id with career_* and efficiency columns.
    """
    params = {
        "season": season,
        "max_week": REGULAR_SEASON_WEEKS,
        "min_games": MIN_GAMES_FOR_RATE,
    }
    with engine.connect() as conn:
        history = pd.read_sql(HISTORY_SQL, conn, params=params)
        bios = pd.read_sql(AGES_SQL, conn)

    if history.empty:
        log.warning("no_career_history", season=season)
        return pd.DataFrame()

    history = history.merge(bios[["player_id", "age"]], on="player_id", how="left")

    # Age in each past season, inferred by counting back from the player's current age.
    # Approximate by a year, which is well inside the resolution of the age curve.
    history["seasons_ago"] = season - history["season"]
    history["age_then"] = history["age"] - history["seasons_ago"]
    next_age = history["age"]

    # Translate each past season to what that production is worth at next season's age.
    then = history["age_then"].map(age_multiplier)
    now = next_age.map(age_multiplier)
    history["age_factor"] = (now / then).clip(0.6, 1.4).fillna(1.0)

    history["weight"] = [
        _season_weight(int(ago), int(g))
        for ago, g in zip(history["seasons_ago"], history["games"], strict=True)
    ]
    history["adj_ppg"] = history["ppg"] * history["age_factor"]

    def weighted(group: pd.DataFrame, column: str) -> float:
        w = group["weight"].to_numpy(dtype=float)
        v = pd.to_numeric(group[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return float(np.average(v, weights=w)) if w.sum() > 0 else 0.0

    rows = []
    for player_id, group in history.groupby("player_id"):
        total_targets = float(group["targets"].sum())
        total_points = float(group["points"].sum())
        rows.append(
            {
                "player_id": int(player_id),
                "career_weighted_ppg": round(weighted(group, "adj_ppg"), 3),
                "career_weighted_targets_per_game": round(weighted(group, "targets_per_game"), 3),
                "career_weighted_carries_per_game": round(weighted(group, "carries_per_game"), 3),
                "career_best_ppg": round(float(group["ppg"].max()), 3),
                "career_seasons": int(len(group)),
                "career_games": int(group["games"].sum()),
                # Career efficiency: total points per total target. Aggregated over the
                # whole career rather than averaged per season, so a big season carries
                # proportionally more weight than a two-game cameo.
                "career_points_per_target": round(
                    total_points / total_targets if total_targets > 0 else 0.0, 4
                ),
            }
        )

    career = pd.DataFrame(rows)

    # Career target share needs team totals per season, so it is computed separately.
    with engine.connect() as conn:
        teams = pd.read_sql(TEAM_TOTALS_SQL, conn, params=params)
    if not teams.empty:
        shares = history.merge(bios[["player_id", "team"]], on="player_id", how="left").merge(
            teams, on=["team", "season"], how="left"
        )
        shares["share"] = (
            (shares["targets"] / shares["team_targets"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
        )
        # Weighted mean without groupby.apply: sum(share x weight) / sum(weight). Avoids
        # a pandas-version-dependent API and is faster on a few thousand rows.
        shares["share_w"] = shares["share"] * shares["weight"]
        grouped = shares.groupby("player_id")[["share_w", "weight"]].sum()
        agg = (
            (grouped["share_w"] / grouped["weight"].replace(0, np.nan))
            .fillna(0.0)
            .round(4)
            .rename("career_weighted_target_share")
            .reset_index()
        )
        career = career.merge(agg, on="player_id", how="left")
    career["career_weighted_target_share"] = career.get(
        "career_weighted_target_share", pd.Series(0.0, index=career.index)
    ).fillna(0.0)

    log.info(
        "career_built",
        season=season,
        players=len(career),
        mean_seasons=round(float(career["career_seasons"].mean()), 2),
    )
    return career


# ------------------------------------------------------------------ efficiency
PRIOR_EFFICIENCY_SQL = text("""
SELECT ps.player_id,
       SUM(ps.fantasy_points) AS points,
       SUM(ps.targets)        AS targets
FROM player_stats ps
WHERE ps.season = :prior_season AND ps.week <= :max_week
GROUP BY ps.player_id
""")


def build_efficiency(engine: Engine, season: int, career: pd.DataFrame) -> pd.DataFrame:
    """Last season's points per target, and the gap against the career rate.

    A large negative gap is the bounce-back signal: the player's usage held but his
    finishing fell off, and finishing is the part that regresses.
    """
    with engine.connect() as conn:
        prior = pd.read_sql(
            PRIOR_EFFICIENCY_SQL,
            conn,
            params={"prior_season": season - 1, "max_week": REGULAR_SEASON_WEEKS},
        )
    if prior.empty:
        return pd.DataFrame(columns=["player_id", "prior_points_per_target", "efficiency_delta"])

    prior["prior_points_per_target"] = (
        prior["points"] / prior["targets"].replace(0, np.nan)
    ).fillna(0.0)

    out = prior[["player_id", "prior_points_per_target"]].copy()
    if not career.empty:
        out = out.merge(
            career[["player_id", "career_points_per_target"]], on="player_id", how="left"
        )
        out["efficiency_delta"] = (
            out["prior_points_per_target"] - out["career_points_per_target"].fillna(0.0)
        ).round(4)
    else:
        out["efficiency_delta"] = 0.0
    out["prior_points_per_target"] = out["prior_points_per_target"].round(4)
    return out[["player_id", "prior_points_per_target", "efficiency_delta"]]


# ------------------------------------------------------------------ situation
# A qb_quality feature lived here and was removed. It rated quarterbacks by their own
# fantasy points, but this project scores receiving and rushing only - so a QB's "points"
# were essentially his scrambles. Minnesota came out at 2.2 and Cincinnati at 0.9, which
# is meaningless. Rating quarterbacks properly needs passing yards and touchdowns
# ingested; until then, team_pass_attempts_prior carries the team-volume signal and
# qb_changed flags the risk. Better a missing feature than a fake one.

PRIOR_ROSTER_SHARES_SQL = text("""
SELECT ps.player_id, p.position, p.team AS current_team,
       SUM(ps.targets) AS targets, SUM(ps.carries) AS carries
FROM player_stats ps
JOIN players p ON p.id = ps.player_id
WHERE ps.season = :prior_season AND ps.week <= :max_week
GROUP BY ps.player_id, p.position, p.team
""")

# Which team each player was on LAST season, from the stats themselves - so a player who
# has since moved is attributed to the team he is leaving.
PRIOR_TEAM_SQL = text("""
SELECT DISTINCT ps.player_id, ps.opponent AS ignored, p.team AS team
FROM player_stats ps JOIN players p ON p.id = ps.player_id
WHERE ps.season = :prior_season
""")


def build_situation(engine: Engine, season: int) -> pd.DataFrame:
    """Per-team situation, expanded to per-player.

    Three signals, all derived from data already stored:

                                    season. A receiver's ceiling is capped by his QB, and
                                    "changed" alone does not say whether that is good news.
      team_departed_target_share  - share that belonged to players no longer on the roster.
                                    Those looks go somewhere, and this is the closest thing
                                    to a forward-looking usage signal available without a
                                    projections feed.
      teammate_top_target_share   - the strongest remaining competitor at the same
                                    position, which is what caps the upside.
    """
    params = {"prior_season": season - 1, "max_week": REGULAR_SEASON_WEEKS}
    with engine.connect() as conn:
        shares = pd.read_sql(PRIOR_ROSTER_SHARES_SQL, conn, params=params)
        roster = pd.read_sql(AGES_SQL, conn)

    if shares.empty:
        return pd.DataFrame()

    # Team totals last season, for share denominators.
    team_totals = (
        shares.groupby("current_team")[["targets", "carries"]]
        .sum()
        .rename(columns={"targets": "team_targets", "carries": "team_carries"})
    )
    shares = shares.join(team_totals, on="current_team")
    shares["target_share"] = (shares["targets"] / shares["team_targets"].replace(0, np.nan)).fillna(
        0.0
    )
    shares["carry_share"] = (shares["carries"] / shares["team_carries"].replace(0, np.nan)).fillna(
        0.0
    )

    # Who is still on the roster? A player whose current team differs from the team he
    # produced for has effectively departed that team's target pool.
    current_team = roster.set_index("player_id")["team"].to_dict()
    shares["still_here"] = [
        current_team.get(int(pid)) == team
        for pid, team in zip(shares["player_id"], shares["current_team"], strict=True)
    ]

    departed = (
        shares[~shares["still_here"]]
        .groupby("current_team")[["target_share", "carry_share"]]
        .sum()
        .rename(
            columns={
                "target_share": "team_departed_target_share",
                "carry_share": "team_departed_carry_share",
            }
        )
        .reset_index()
        .rename(columns={"current_team": "team"})
    )

    # Strongest remaining competitor at the same position, per player.
    staying = shares[shares["still_here"]]
    competitor_rows = []
    for (team, position), group in staying.groupby(["current_team", "position"]):
        ordered = group.sort_values("target_share", ascending=False)
        for row in ordered.itertuples(index=False):
            others = ordered[ordered.player_id != row.player_id]
            competitor_rows.append(
                {
                    "player_id": int(row.player_id),
                    "teammate_top_target_share": round(
                        float(others["target_share"].max()) if len(others) else 0.0, 4
                    ),
                    "teammate_top_carry_share": round(
                        float(others["carry_share"].max()) if len(others) else 0.0, 4
                    ),
                }
            )
        del team, position
    competitors = pd.DataFrame(competitor_rows)

    out = roster[["player_id", "team"]].merge(departed, on="team", how="left")
    if not competitors.empty:
        out = out.merge(competitors, on="player_id", how="left")

    for col in (
        "team_departed_target_share",
        "team_departed_carry_share",
        "teammate_top_target_share",
        "teammate_top_carry_share",
    ):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).round(4)

    log.info(
        "situation_built",
        season=season,
        players=len(out),
    )
    return out.drop(columns=["team"])
