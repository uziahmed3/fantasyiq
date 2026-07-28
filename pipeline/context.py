"""Build player_context: what was knowable before a season started.

The in-season model runs on rolling 3-game form. That does not exist in week 1, and it
never exists for a rookie. This module assembles the other half of the picture:

    prior-season production   <- player_stats already in the database
    target share              <- player targets / team targets (prior season)
    playing time              <- snap_counts release (offense_pct)
    role                      <- depth_charts release (depth order)
    draft capital             <- players.parquet (draft round/pick, rookie season)
    team situation            <- team pass volume, and whether the starting QB changed

Two joins are awkward and worth knowing about:

  * snap_counts has no gsis_id - it keys on pfr_player_id. players.parquet carries both,
    so it acts as the crosswalk.
  * depth_charts changed schema in 2025: the older files have season/week/depth_team,
    the newer ESPN-sourced ones have dt/pos_rank and no season column. Both are handled;
    an unrecognised shape logs a warning and leaves depth rank null rather than guessing.

Everything degrades to null rather than failing. A missing optional source should cost
accuracy, not availability - the model handles nulls, an outage of one feed should not
stop the projections.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

import career as career_mod
import ingest
import load
from logging_setup import log

# The NFL regular season is 18 weeks; nflverse data continues through the playoffs
# (weeks 19-22). Including those inflated prior-season rates and produced "19 games last
# season" on the draft board, which is impossible - and it rewarded players whose teams
# made a deep run rather than players who were good.
REGULAR_SEASON_WEEKS = 18

# Quarterbacks are ingested (qb_changed needs them) but never projected: fantasy points
# here are computed from receiving and rushing only, so a QB's number would omit passing
# entirely and be badly wrong. Excluded at the source rather than hidden in the UI.
PROJECTED_POSITIONS = ("WR", "RB", "TE")

UPSERT_CONTEXT = text("""
INSERT INTO player_context (
    player_id, season, team,
    prior_games, prior_points_per_game, prior_targets_per_game, prior_yards_per_game,
    prior_target_share, prior_carries_per_game, prior_carry_share,
    prior_last4_points_per_game, prior_snap_share,
    depth_chart_rank, draft_round, draft_pick, rookie_season, years_experience,
    is_rookie, age, team_pass_attempts_prior, team_points_prior, qb_changed,
    career_weighted_ppg, career_weighted_targets_per_game, career_weighted_carries_per_game,
    career_weighted_target_share, career_best_ppg, career_seasons, career_games,
    prior_points_per_target, career_points_per_target, efficiency_delta,
    qb_quality, team_departed_target_share, team_departed_carry_share,
    teammate_top_target_share, teammate_top_carry_share,
    created_at, updated_at
) VALUES (
    :player_id, :season, :team,
    :prior_games, :prior_points_per_game, :prior_targets_per_game, :prior_yards_per_game,
:prior_target_share, :prior_carries_per_game, :prior_carry_share,
    :prior_last4_points_per_game, :prior_snap_share,
    :depth_chart_rank, :draft_round, :draft_pick, :rookie_season, :years_experience,
:is_rookie, :age, :team_pass_attempts_prior, :team_points_prior, :qb_changed,
    :career_weighted_ppg, :career_weighted_targets_per_game, :career_weighted_carries_per_game,
    :career_weighted_target_share, :career_best_ppg, :career_seasons, :career_games,
    :prior_points_per_target, :career_points_per_target, :efficiency_delta,
    :qb_quality, :team_departed_target_share, :team_departed_carry_share,
    :teammate_top_target_share, :teammate_top_carry_share,
    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
)
ON CONFLICT (player_id, season) DO UPDATE SET
    team                        = EXCLUDED.team,
    prior_games                 = EXCLUDED.prior_games,
    prior_points_per_game       = EXCLUDED.prior_points_per_game,
    prior_targets_per_game      = EXCLUDED.prior_targets_per_game,
    prior_yards_per_game        = EXCLUDED.prior_yards_per_game,
    prior_target_share          = EXCLUDED.prior_target_share,
    prior_carries_per_game      = EXCLUDED.prior_carries_per_game,
    prior_carry_share           = EXCLUDED.prior_carry_share,
    prior_last4_points_per_game = EXCLUDED.prior_last4_points_per_game,
    prior_snap_share            = COALESCE(EXCLUDED.prior_snap_share,
                                           player_context.prior_snap_share),
    depth_chart_rank            = COALESCE(EXCLUDED.depth_chart_rank,
                                           player_context.depth_chart_rank),
    draft_round                 = COALESCE(EXCLUDED.draft_round, player_context.draft_round),
    draft_pick                  = COALESCE(EXCLUDED.draft_pick, player_context.draft_pick),
    rookie_season               = COALESCE(EXCLUDED.rookie_season, player_context.rookie_season),
    years_experience            = EXCLUDED.years_experience,
    is_rookie                   = EXCLUDED.is_rookie,
    age                         = COALESCE(EXCLUDED.age, player_context.age),
    team_pass_attempts_prior    = EXCLUDED.team_pass_attempts_prior,
    team_points_prior           = EXCLUDED.team_points_prior,
    qb_changed                  = EXCLUDED.qb_changed,
    career_weighted_ppg              = EXCLUDED.career_weighted_ppg,
    career_weighted_targets_per_game = EXCLUDED.career_weighted_targets_per_game,
    career_weighted_carries_per_game = EXCLUDED.career_weighted_carries_per_game,
    career_weighted_target_share     = EXCLUDED.career_weighted_target_share,
    career_best_ppg                  = EXCLUDED.career_best_ppg,
    career_seasons                   = EXCLUDED.career_seasons,
    career_games                     = EXCLUDED.career_games,
    prior_points_per_target          = EXCLUDED.prior_points_per_target,
    career_points_per_target         = EXCLUDED.career_points_per_target,
    efficiency_delta                 = EXCLUDED.efficiency_delta,
    qb_quality                       = EXCLUDED.qb_quality,
    team_departed_target_share       = EXCLUDED.team_departed_target_share,
    team_departed_carry_share        = EXCLUDED.team_departed_carry_share,
    teammate_top_target_share        = EXCLUDED.teammate_top_target_share,
    teammate_top_carry_share         = EXCLUDED.teammate_top_carry_share,
    updated_at                  = CURRENT_TIMESTAMP
""")


# --------------------------------------------------------------------- prior production
PRIOR_SQL = text("""
SELECT
    ps.player_id,
    p.external_id,
    p.position,
    ps.week,
    ps.opponent,
    ps.targets,
    ps.carries,
    ps.yards,
    ps.fantasy_points,
    p.team AS current_team
FROM player_stats ps
JOIN players p ON p.id = ps.player_id
WHERE ps.season = :prior_season AND ps.week <= :max_week
""")

TEAM_SQL = text("""
SELECT p.team AS team, SUM(ps.targets) AS team_targets,
       SUM(ps.carries) AS team_carries,
       SUM(ps.fantasy_points) AS team_points
FROM player_stats ps
JOIN players p ON p.id = ps.player_id
WHERE ps.season = :prior_season AND p.team IS NOT NULL AND ps.week <= :max_week
GROUP BY p.team
""")


def _prior_production(engine: Engine, prior_season: int) -> pd.DataFrame:
    """Per-player prior-season aggregates, including share of team targets."""
    with engine.connect() as conn:
        params = {"prior_season": prior_season, "max_week": REGULAR_SEASON_WEEKS}
        rows = pd.read_sql(PRIOR_SQL, conn, params=params)
        teams = pd.read_sql(TEAM_SQL, conn, params=params)

    if rows.empty:
        return pd.DataFrame()

    grouped = rows.groupby(["player_id", "external_id", "current_team"], dropna=False)
    agg = grouped.agg(
        prior_games=("fantasy_points", "size"),
        prior_points_per_game=("fantasy_points", "mean"),
        prior_targets_per_game=("targets", "mean"),
        prior_carries_per_game=("carries", "mean"),
        prior_yards_per_game=("yards", "mean"),
        prior_targets_total=("targets", "sum"),
        prior_carries_total=("carries", "sum"),
    ).reset_index()

    # Last 4 games: a player whose role changed late in the year is better described by
    # recent form than by a season average that includes weeks he was not playing.
    last4 = (
        rows.sort_values(["player_id", "week"])
        .groupby("player_id")
        .tail(4)
        .groupby("player_id")["fantasy_points"]
        .mean()
        .rename("prior_last4_points_per_game")
        .reset_index()
    )
    agg = agg.merge(last4, on="player_id", how="left")

    agg = agg.merge(teams.rename(columns={"team": "current_team"}), on="current_team", how="left")
    agg["prior_target_share"] = (
        (agg["prior_targets_total"] / agg["team_targets"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    )
    # Carry share is to a running back what target share is to a receiver: it separates a
    # lead back from someone on a team that simply runs a lot.
    agg["prior_carry_share"] = (
        (agg["prior_carries_total"] / agg["team_carries"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    )
    agg["team_points_prior"] = agg["team_points"].fillna(0.0)
    return agg.drop(
        columns=[
            "prior_targets_total",
            "prior_carries_total",
            "team_targets",
            "team_carries",
            "team_points",
        ]
    )


# --------------------------------------------------------------------- team / QB context
QB_SQL = text("""
SELECT p.team AS team, p.external_id AS qb_id, SUM(ps.targets) AS ignored,
       SUM(ps.yards) AS pass_yards, COUNT(*) AS games
FROM player_stats ps
JOIN players p ON p.id = ps.player_id
WHERE ps.season = :season AND p.position = 'QB' AND p.team IS NOT NULL
  AND ps.week <= :max_week
GROUP BY p.team, p.external_id
""")


def _team_situation(engine: Engine, prior_season: int) -> pd.DataFrame:
    """Team pass volume in the prior season, and whether the primary QB changed.

    'Primary QB' = the QB with the most yardage for that team. Crude but it is the right
    shape: a receiver whose quarterback changed is a genuinely different projection, and
    that is knowable before week 1.

    If QBs are not ingested (INGEST_POSITIONS excludes them), qb_changed stays False and
    team pass volume falls back to team target volume - degraded, not broken.
    """
    with engine.connect() as conn:
        prior = pd.read_sql(
            QB_SQL, conn, params={"season": prior_season, "max_week": REGULAR_SEASON_WEEKS}
        )
        earlier = pd.read_sql(
            QB_SQL,
            conn,
            params={"season": prior_season - 1, "max_week": REGULAR_SEASON_WEEKS},
        )

    if prior.empty:
        log.warning(
            "no_qb_data",
            season=prior_season,
            hint="set INGEST_POSITIONS to include QB for the qb_changed feature",
        )
        return pd.DataFrame(columns=["team", "team_pass_attempts_prior", "qb_changed"])

    def primary(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.sort_values("pass_yards", ascending=False)
            .groupby("team", as_index=False)
            .first()[["team", "qb_id"]]
        )

    volume = prior.groupby("team", as_index=False)["pass_yards"].sum()
    volume = volume.rename(columns={"pass_yards": "team_pass_attempts_prior"})

    now, before = primary(prior), primary(earlier).rename(columns={"qb_id": "prev_qb"})
    merged = volume.merge(now, on="team", how="left").merge(before, on="team", how="left")
    # Unknown previous QB (no season-2 data) is not evidence of a change.
    merged["qb_changed"] = merged["prev_qb"].notna() & (merged["qb_id"] != merged["prev_qb"])
    return merged[["team", "team_pass_attempts_prior", "qb_changed"]]


# --------------------------------------------------------------------- optional feeds
REGISTER_ROOKIE = text("""
INSERT INTO players (external_id, name, team, position, age, height_inches, weight_lbs,
                     draft_round, draft_pick, rookie_season, created_at, updated_at)
VALUES (:external_id, :name, :team, :position, :age, :height_inches, :weight_lbs,
        :draft_round, :draft_pick, :rookie_season, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (external_id) DO UPDATE SET
    team          = COALESCE(EXCLUDED.team, players.team),
    position      = EXCLUDED.position,
    draft_round   = COALESCE(EXCLUDED.draft_round, players.draft_round),
    draft_pick    = COALESCE(EXCLUDED.draft_pick, players.draft_pick),
    rookie_season = COALESCE(EXCLUDED.rookie_season, players.rookie_season),
    updated_at    = CURRENT_TIMESTAMP
""")


def _draft_picks(season: int | None = None) -> pd.DataFrame:
    """Draft round and overall pick from the draft_picks release.

    Needed because players.parquet lags: checked against the real file, all 679 players
    with rookie_season 2026 had null draft_year/round/pick, while draft_picks.parquet is
    published within weeks of the draft. For a rookie, draft position is the single most
    informative feature we have, so it is worth the extra fetch.
    """
    try:
        path = ingest._download(ingest.DRAFT_URL, ingest.DOWNLOAD_DIR / "draft_picks.parquet")
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - optional feed
        log.warning("draft_picks_unavailable", error=str(exc))
        return pd.DataFrame()

    if "gsis_id" not in df.columns:
        log.warning("draft_picks_no_gsis_id", columns=list(df.columns)[:12])
        return pd.DataFrame()

    keep = {"gsis_id": "external_id", "round": "draft_round", "pick": "draft_pick"}
    present = {k: v for k, v in keep.items() if k in df.columns}
    if "round" not in df.columns or "pick" not in df.columns:
        log.warning("draft_picks_unexpected_schema", columns=list(df.columns)[:12])
        return pd.DataFrame()

    out = df[list(present) + (["season"] if "season" in df.columns else [])].rename(columns=present)
    if season is not None and "season" in out.columns:
        # Only this class; older picks already live on the players table.
        out = out[pd.to_numeric(out["season"], errors="coerce") == season]
    out = out.dropna(subset=["external_id"]).drop_duplicates("external_id")
    log.info("draft_picks_loaded", season=season, rows=len(out))
    return out.drop(columns=["season"], errors="ignore")


def register_incoming_rookies(engine: Engine, season: int, positions: list[str]) -> int:
    """Insert this year's rookie class, who have no stats and so are not in the database.

    Players only get a row when they appear in weekly stats, which means a draft class is
    invisible until it plays - and a draft board without rookies is missing exactly the
    players people most need help evaluating. The bios feed knows them in the spring:
    name, position, team and draft slot, which is all the preseason model needs.
    """
    bios = _player_bios(full=True)
    if bios.empty or "rookie_season" not in bios.columns:
        log.warning("rookie_registration_skipped", reason="no bios or no rookie_season")
        return 0

    incoming = bios[pd.to_numeric(bios["rookie_season"], errors="coerce") == season].copy()
    if "position" in incoming.columns:
        incoming = incoming[incoming["position"].str.upper().isin([p.upper() for p in positions])]
    if incoming.empty:
        log.info("no_incoming_rookies", season=season)
        return 0

    # Fill draft capital from the draft_picks release, which is current when
    # players.parquet is not.
    picks = _draft_picks(season)
    if not picks.empty:
        incoming = incoming.merge(picks, on="external_id", how="left", suffixes=("", "_dp"))
        for col in ("draft_round", "draft_pick"):
            alt = f"{col}_dp"
            if alt in incoming.columns:
                incoming[col] = incoming[col].combine_first(incoming[alt])
                incoming = incoming.drop(columns=[alt])

    payload = []
    for row in incoming.itertuples(index=False):
        payload.append(
            {
                "external_id": str(row.external_id),
                "name": getattr(row, "name", None) or str(row.external_id),
                "team": getattr(row, "team", None),
                "position": (getattr(row, "position", None) or "WR").upper()[:4],
                # Draft-eligible players are 21-23; a real birth_date would be better but
                # the age curve barely moves across that range.
                "age": 22.0,
                "height_inches": None,
                "weight_lbs": None,
                "draft_round": _clean_int(getattr(row, "draft_round", None)),
                "draft_pick": _clean_int(getattr(row, "draft_pick", None)),
                "rookie_season": season,
            }
        )

    with engine.begin() as conn:
        for i in range(0, len(payload), 500):
            conn.execute(REGISTER_ROOKIE, payload[i : i + 500])

    drafted = sum(1 for p in payload if p["draft_pick"] is not None)
    log.info(
        "rookies_registered",
        season=season,
        players=len(payload),
        with_draft_pick=drafted,
    )
    return len(payload)


def _player_bios(full: bool = False) -> pd.DataFrame:
    """players.parquet: draft capital, rookie season, and the pfr_id crosswalk."""
    try:
        path = ingest._download(ingest.PLAYERS_URL, ingest.DOWNLOAD_DIR / "players.parquet")
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - optional feed
        log.warning("player_bios_unavailable", error=str(exc))
        return pd.DataFrame()

    wanted = {
        "gsis_id": "external_id",
        "pfr_id": "pfr_id",
        "draft_round": "draft_round",
        "draft_pick": "draft_pick",
        "rookie_season": "rookie_season",
        "years_of_experience": "years_of_experience",
    }
    if full:
        # Registering a brand-new player needs identity, not just draft numbers.
        wanted |= {
            "display_name": "name",
            "full_name": "name",
            "position": "position",
            "latest_team": "team",
            "team_abbr": "team",
        }
    present = {}
    for src, dst in wanted.items():
        if src in df.columns and dst not in present.values():
            present[src] = dst
    missing = set(wanted) - set(present)
    if missing:
        log.warning("player_bios_columns_missing", columns=sorted(missing))
    if "gsis_id" not in present:
        return pd.DataFrame()
    return df[list(present)].rename(columns=present).drop_duplicates("external_id")


def _snap_shares(season: int, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Mean offensive snap share for a season, keyed to external_id via pfr_id."""
    if crosswalk.empty or "pfr_id" not in crosswalk.columns:
        log.warning("snap_share_skipped", reason="no pfr_id crosswalk available")
        return pd.DataFrame()
    try:
        dest = ingest.DOWNLOAD_DIR / f"snap_counts_{season}.parquet"
        path = ingest._download_any(ingest.SNAPS_URL_CANDIDATES, season, dest)
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - optional feed
        log.warning("snap_counts_unavailable", season=season, error=str(exc))
        return pd.DataFrame()

    if not {"pfr_player_id", "offense_pct"} <= set(df.columns):
        log.warning("snap_counts_unexpected_schema", columns=list(df.columns)[:12])
        return pd.DataFrame()

    shares = (
        df.groupby("pfr_player_id", as_index=False)["offense_pct"]
        .mean()
        .rename(columns={"pfr_player_id": "pfr_id", "offense_pct": "prior_snap_share"})
    )
    out = shares.merge(crosswalk[["external_id", "pfr_id"]], on="pfr_id", how="inner")
    # Some seasons express the share as a percentage, others as a fraction.
    if not out.empty and out["prior_snap_share"].max() > 1.5:
        out["prior_snap_share"] = out["prior_snap_share"] / 100.0
    log.info("snap_shares_built", season=season, players=len(out))
    return out[["external_id", "prior_snap_share"]]


def _depth_ranks(season: int) -> pd.DataFrame:
    """Depth-chart order for a season. Handles both known schemas.

    2001-2024: season / week / depth_team / gsis_id
    2025+    : dt / pos_rank / gsis_id  (ESPN-sourced, many daily snapshots, no season)
    """
    try:
        dest = ingest.DOWNLOAD_DIR / f"depth_charts_{season}.parquet"
        path = ingest._download_any(ingest.DEPTH_URL_CANDIDATES, season, dest)
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - optional feed
        log.warning("depth_charts_unavailable", season=season, error=str(exc))
        return pd.DataFrame()

    if "gsis_id" not in df.columns:
        log.warning("depth_charts_no_gsis_id", columns=list(df.columns)[:12])
        return pd.DataFrame()

    if "pos_rank" in df.columns and "dt" in df.columns:
        # Newer schema: take the most recent snapshot per player.
        df = df.sort_values("dt").groupby("gsis_id", as_index=False).last()
        rank_col = "pos_rank"
    elif "depth_team" in df.columns:
        # Older schema: earliest week is closest to a preseason depth chart.
        if "week" in df.columns:
            df = df.sort_values("week").groupby("gsis_id", as_index=False).first()
        else:
            df = df.groupby("gsis_id", as_index=False).first()
        rank_col = "depth_team"
    else:
        log.warning("depth_charts_unexpected_schema", columns=list(df.columns)[:12])
        return pd.DataFrame()

    out = df[["gsis_id", rank_col]].rename(
        columns={"gsis_id": "external_id", rank_col: "depth_chart_rank"}
    )
    out["depth_chart_rank"] = pd.to_numeric(out["depth_chart_rank"], errors="coerce").clip(1, 10)
    out = out.dropna(subset=["depth_chart_rank"])
    log.info("depth_ranks_built", season=season, players=len(out), schema=rank_col)
    return out


# --------------------------------------------------------------------- assembly
def _clean_int(value) -> int | None:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_float(value) -> float | None:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build(
    engine: Engine,
    season: int,
    use_optional_feeds: bool = True,
    register_rookies: bool = True,
) -> int:
    """Assemble and upsert player_context for `season`, from season - 1 data."""
    prior_season = season - 1
    log.info("context_build_start", season=season, prior_season=prior_season)

    if use_optional_feeds and register_rookies:
        # Do this first so the roster query below picks the new players up.
        from config import POSITIONS

        register_incoming_rookies(engine, season, POSITIONS)

    production = _prior_production(engine, prior_season)
    situation = _team_situation(engine, prior_season)

    # Multi-season history, the volume/efficiency split, and the situation signals.
    # All derived from the database - no extra downloads.
    career = career_mod.build_career(engine, season)
    efficiency = career_mod.build_efficiency(engine, season, career)
    extra_situation = career_mod.build_situation(engine, season)

    bios = _player_bios() if use_optional_feeds else pd.DataFrame()
    snaps = _snap_shares(prior_season, bios) if use_optional_feeds else pd.DataFrame()
    depth = _depth_ranks(season) if use_optional_feeds else pd.DataFrame()

    # Who plausibly plays in this season?
    #
    # Not "every row in players" - that table accumulates every player we have ever seen,
    # including ones who only appear in later seasons. Including them made 198 of 486 rows
    # look like rookies (no prior games -> assumed rookie) when they simply did not exist
    # yet, which poisoned both the rookie flag and the training set.
    #
    # A player belongs in season S's context if he played in S, or played in S-1, or the
    # bios say S is his rookie season (the genuine rookie case: no stats anywhere yet).
    with engine.connect() as conn:
        roster = pd.read_sql(
            text("""
                SELECT p.id AS player_id, p.external_id, p.team, p.position, p.age,
                       p.draft_round, p.draft_pick, p.rookie_season
                FROM players p
                WHERE EXISTS (
                    SELECT 1 FROM player_stats ps
                    WHERE ps.player_id = p.id AND ps.season IN (:season, :prior_season)
                )
                -- ...or he is entering the league this year. Rookies have no stats
                -- anywhere, so EXISTS alone would drop the entire draft class - which is
                -- exactly the group a draft board most needs.
                OR p.rookie_season = :season
            """),
            conn,
            params={"season": season, "prior_season": prior_season},
        )
    if roster.empty:
        log.warning("context_no_players")
        return 0

    df = roster.merge(
        production.drop(columns=["external_id", "current_team"], errors="ignore"),
        on="player_id",
        how="left",
    )
    if not situation.empty:
        df = df.merge(situation, on="team", how="left")
    for frame in (career, efficiency, extra_situation):
        if not frame.empty:
            df = df.merge(frame, on="player_id", how="left")
    for optional in (snaps, depth):
        if not optional.empty:
            df = df.merge(optional, on="external_id", how="left")
    if not bios.empty:
        df = df.merge(bios.drop(columns=["pfr_id"], errors="ignore"), on="external_id", how="left")

    numeric_defaults = {
        "prior_games": 0,
        "prior_points_per_game": 0.0,
        "prior_targets_per_game": 0.0,
        "prior_yards_per_game": 0.0,
        "prior_target_share": 0.0,
        "prior_carries_per_game": 0.0,
        "prior_carry_share": 0.0,
        "prior_last4_points_per_game": 0.0,
        "team_pass_attempts_prior": 0.0,
        "team_points_prior": 0.0,
        "career_weighted_ppg": 0.0,
        "career_weighted_targets_per_game": 0.0,
        "career_weighted_carries_per_game": 0.0,
        "career_weighted_target_share": 0.0,
        "career_best_ppg": 0.0,
        "career_seasons": 0,
        "career_games": 0,
        "prior_points_per_target": 0.0,
        "career_points_per_target": 0.0,
        "efficiency_delta": 0.0,
        "qb_quality": 0.0,
        "team_departed_target_share": 0.0,
        "team_departed_carry_share": 0.0,
        "teammate_top_target_share": 0.0,
        "teammate_top_carry_share": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    for col in (
        "prior_snap_share",
        "depth_chart_rank",
        "draft_round",
        "draft_pick",
        "rookie_season",
        "years_of_experience",
        "age",
    ):
        if col not in df.columns:
            df[col] = None
    if "qb_changed" not in df.columns:
        df["qb_changed"] = False
    df["qb_changed"] = df["qb_changed"].fillna(False).astype(bool)

    # A rookie is someone whose first season is the one being projected. Falling back to
    # "no prior games" would also flag veterans returning from injury, who are a very
    # different projection problem.
    rookie_from_bio = df["rookie_season"].astype("Float64") == season
    df["is_rookie"] = rookie_from_bio.fillna(df["prior_games"] == 0).astype(bool)

    payload = [
        {
            "player_id": int(r.player_id),
            "season": season,
            "team": r.team,
            "prior_games": int(r.prior_games),
            "prior_points_per_game": float(r.prior_points_per_game),
            "prior_targets_per_game": float(r.prior_targets_per_game),
            "prior_yards_per_game": float(r.prior_yards_per_game),
            "prior_target_share": float(r.prior_target_share),
            "prior_carries_per_game": float(r.prior_carries_per_game),
            "prior_carry_share": float(r.prior_carry_share),
            "prior_last4_points_per_game": float(r.prior_last4_points_per_game),
            "prior_snap_share": _clean_float(getattr(r, "prior_snap_share", None)),
            "depth_chart_rank": _clean_int(getattr(r, "depth_chart_rank", None)),
            "draft_round": _clean_int(getattr(r, "draft_round", None)),
            "draft_pick": _clean_int(getattr(r, "draft_pick", None)),
            "rookie_season": _clean_int(getattr(r, "rookie_season", None)),
            "years_experience": _clean_int(getattr(r, "years_of_experience", None)),
            "is_rookie": bool(r.is_rookie),
            "age": _clean_float(getattr(r, "age", None)),
            "team_pass_attempts_prior": float(r.team_pass_attempts_prior),
            "team_points_prior": float(r.team_points_prior),
            "qb_changed": bool(r.qb_changed),
            "career_weighted_ppg": float(r.career_weighted_ppg),
            "career_weighted_targets_per_game": float(r.career_weighted_targets_per_game),
            "career_weighted_carries_per_game": float(r.career_weighted_carries_per_game),
            "career_weighted_target_share": float(r.career_weighted_target_share),
            "career_best_ppg": float(r.career_best_ppg),
            "career_seasons": int(r.career_seasons),
            "career_games": int(r.career_games),
            "prior_points_per_target": float(r.prior_points_per_target),
            "career_points_per_target": float(r.career_points_per_target),
            "efficiency_delta": float(r.efficiency_delta),
            "qb_quality": float(r.qb_quality),
            "team_departed_target_share": float(r.team_departed_target_share),
            "team_departed_carry_share": float(r.team_departed_carry_share),
            "teammate_top_target_share": float(r.teammate_top_target_share),
            "teammate_top_carry_share": float(r.teammate_top_carry_share),
        }
        for r in df.itertuples(index=False)
    ]

    with engine.begin() as conn:
        for i in range(0, len(payload), 500):
            conn.execute(UPSERT_CONTEXT, payload[i : i + 500])

        # Make this build authoritative for the season. Upserting alone leaves behind rows
        # for players who are no longer relevant - someone who left the league, or (as
        # happened here) rows written by an earlier, buggier version of this code. Without
        # this, a stale row keeps being served and keeps polluting training.
        keep = {p["player_id"] for p in payload}
        existing = {
            r[0]
            for r in conn.execute(
                text("SELECT player_id FROM player_context WHERE season = :s"),
                {"s": season},
            )
        }
        stale = existing - keep
        if stale:
            log.info("context_pruned_stale", season=season, removed=len(stale))
            for i in range(0, len(stale), 500):
                chunk = list(stale)[i : i + 500]
                conn.execute(
                    text(
                        "DELETE FROM player_context WHERE season = :s AND player_id IN "
                        "(" + ",".join(str(int(x)) for x in chunk) + ")"
                    ),
                    {"s": season},
                )

    rookies = sum(1 for p in payload if p["is_rookie"])
    with_snaps = sum(1 for p in payload if p["prior_snap_share"] is not None)
    with_depth = sum(1 for p in payload if p["depth_chart_rank"] is not None)
    log.info(
        "context_build_done",
        season=season,
        rows=len(payload),
        rookies=rookies,
        with_snap_share=with_snaps,
        with_depth_rank=with_depth,
    )
    return len(payload)


def seasons_with_data(engine: Engine) -> list[int]:
    with engine.connect() as conn:
        return [
            int(r[0])
            for r in conn.execute(text("SELECT DISTINCT season FROM player_stats ORDER BY season"))
        ]


def build_all(engine: Engine, use_optional_feeds: bool = True) -> dict[int, int]:
    """Build context for every season we can, including the one being projected.

    Two groups, and missing the second was a real bug:

      * seasons that have games, except the earliest (nothing precedes it) - these are the
        training pairs the preseason model learns from.
      * the NEXT season, which by definition has no games yet. It is the whole point of a
        draft board, and because it never appears in seasons_with_data it was silently
        skipped - so `context --all` left the projection season holding whatever stale
        values an earlier run had written. On real data that meant "21 games last season"
        for players whose team went deep in the playoffs, long after the regular-season
        filter was added.
    """
    seasons = seasons_with_data(engine)
    if len(seasons) < 2:
        log.warning(
            "context_needs_two_seasons",
            seasons=seasons,
            hint="ingest at least two consecutive seasons",
        )
        return {}
    targets = [*seasons[1:], seasons[-1] + 1]
    built = {}
    for season in targets:
        built[season] = build(engine, season, use_optional_feeds=use_optional_feeds)
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build preseason context")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season", type=int, help="Single season being projected")
    group.add_argument(
        "--all",
        action="store_true",
        help="Build for every season that has a prior season - what the preseason model "
        "needs in order to have more than one training pair",
    )
    parser.add_argument(
        "--no-optional-feeds",
        action="store_true",
        help="Skip snap counts, depth charts and player bios (no network beyond the DB)",
    )
    args = parser.parse_args(argv)

    engine = load.get_engine()
    feeds = not args.no_optional_feeds

    if args.all:
        built = build_all(engine, use_optional_feeds=feeds)
        if not built:
            print(
                "\nNothing to build. Context needs at least two seasons of stats "
                "(season S uses season S-1).\n"
            )
            return 1
        for season, rows in sorted(built.items()):
            print(f"  {season}: context for {rows} players")
        print()
        return 0

    rows = build(engine, args.season, use_optional_feeds=feeds)
    print(f"\nBuilt context for {rows} players for the {args.season} season.\n")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
