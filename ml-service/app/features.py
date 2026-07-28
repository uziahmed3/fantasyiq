"""The feature contract.

FEATURE_ORDER is the single source of truth for column ordering. Training scripts build
their design matrix from it and the serving path builds its input row from it, so a
reordered or renamed feature is a loud failure instead of silent training/serving skew.
Bump FEATURE_SCHEMA_VERSION whenever this list changes, and retrain.
"""

from collections.abc import Mapping

import numpy as np

FEATURE_SCHEMA_VERSION = "fs1"

FEATURE_ORDER: tuple[str, ...] = (
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

TARGET = "fantasy_points"

# ---------------------------------------------------------------- preseason contract
# A separate contract for the case the in-season features cannot describe: no games yet
# this season. That is week 1 for everyone, and the whole season for a rookie.
#
# Nothing here requires a single game to have been played. Draft capital and depth-chart
# rank are the only real signal for a rookie; target share and snap share separate
# "productive" from "happened to be on a pass-heavy team"; qb_changed captures the
# biggest knowable swing factor for a receiver.
PRESEASON_SCHEMA_VERSION = "ps5"

PRESEASON_FEATURE_ORDER: tuple[str, ...] = (
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
    # ---- multi-season history ----
    # One prior season let a single down year erase a career: Jefferson went 19.5, 21.5,
    # 20.4, 18.6, 11.9 and projected 11.91. These carry the whole record, weighted by
    # recency, games played and the age curve.
    "career_weighted_ppg",
    "career_weighted_targets_per_game",
    "career_weighted_carries_per_game",
    "career_weighted_target_share",
    "career_best_ppg",
    "career_seasons",
    "career_games",
    # ---- volume vs efficiency ----
    # Volume is stable year to year (~0.83 correlation); efficiency is not. A large
    # negative efficiency_delta means the role held and only the finishing broke.
    "prior_points_per_target",
    "career_points_per_target",
    "efficiency_delta",
    # ---- situation ----
    "team_departed_target_share",
    "team_departed_carry_share",
    "teammate_top_target_share",
    "teammate_top_carry_share",
)

# Sentinels for genuinely-unknown values. Undrafted is not round 0 - it is worse than
# round 7, so it maps to 8. An unknown depth rank maps to 5 (buried, not starting).
PRESEASON_DEFAULTS: dict[str, float] = {
    "prior_points_per_game": 0.0,
    "prior_last4_points_per_game": 0.0,
    "prior_targets_per_game": 0.0,
    "prior_target_share": 0.0,
    "prior_carries_per_game": 0.0,
    "prior_carry_share": 0.0,
    "prior_yards_per_game": 0.0,
    "prior_games": 0.0,
    "prior_snap_share": 0.0,
    "depth_chart_rank": 5.0,
    "draft_round": 8.0,
    "draft_pick": 300.0,
    "years_experience": 0.0,
    "is_rookie": 0.0,
    "age": 25.0,
    "team_pass_attempts_prior": 0.0,
    "qb_changed": 0.0,
    "career_weighted_ppg": 0.0,
    "career_weighted_targets_per_game": 0.0,
    "career_weighted_carries_per_game": 0.0,
    "career_weighted_target_share": 0.0,
    "career_best_ppg": 0.0,
    "career_seasons": 0.0,
    "career_games": 0.0,
    "prior_points_per_target": 0.0,
    "career_points_per_target": 0.0,
    "efficiency_delta": 0.0,
    "team_departed_target_share": 0.0,
    "team_departed_carry_share": 0.0,
    "teammate_top_target_share": 0.0,
    "teammate_top_carry_share": 0.0,
}


# ---------------------------------------------------------- draft capital, decayed
# Draft position is a guess about talent nobody has measured yet. For a rookie it is
# almost the only signal there is. For a player with three seasons behind him it has been
# superseded - we no longer need to guess, we can look.
#
# The model did not know that. Across the training set high picks outproduce low picks,
# so it learned to reward draft position for everybody, and with about 1,300 training
# rows it could not discover the interaction ("use this only when production is missing")
# on its own. The bill came due on Puka Nacua: pick 177, and still being charged for it
# after 44 games at 23.6 points a game. Against Jaxon Smith-Njigba, pick 20, draft
# capital alone swung 2.4 points - more than their entire difference in production.
#
# So the shrink is made explicit instead of hoped for. Draft capital fades toward
# league-average as real games accumulate, and is gone after two full seasons.
DRAFT_EVIDENCE_GAMES = 32.0
# Round 4 of 7 and the middle of the draft: "we know nothing either way".
NEUTRAL_DRAFT_ROUND = 4.0
NEUTRAL_DRAFT_PICK = 120.0


def draft_capital_weight(career_games: float | None) -> float:
    """How much draft position still deserves to count: 1.0 for a rookie, 0.0 by game 32."""
    if career_games is None or not np.isfinite(float(career_games)):
        return 1.0
    return float(max(0.0, 1.0 - float(career_games) / DRAFT_EVIDENCE_GAMES))


def decay_draft_capital(
    draft_round: float, draft_pick: float, career_games: float | None
) -> tuple[float, float]:
    """Blend draft position toward neutral in proportion to games actually played."""
    w = draft_capital_weight(career_games)
    return (
        w * float(draft_round) + (1.0 - w) * NEUTRAL_DRAFT_ROUND,
        w * float(draft_pick) + (1.0 - w) * NEUTRAL_DRAFT_PICK,
    )


class FeatureContractError(ValueError):
    pass


def to_row(features: Mapping[str, float]) -> np.ndarray:
    missing = [f for f in FEATURE_ORDER if f not in features]
    if missing:
        raise FeatureContractError(f"missing features: {missing}")
    unexpected = [k for k in features if k not in FEATURE_ORDER]
    if unexpected:
        raise FeatureContractError(f"unexpected features: {unexpected}")
    return np.asarray([[float(features[f]) for f in FEATURE_ORDER]], dtype=np.float32)


def to_matrix(rows: list[Mapping[str, float]]) -> np.ndarray:
    return np.vstack([to_row(r) for r in rows])


def preseason_row(features: Mapping[str, float | None]) -> np.ndarray:
    """Build a preseason input row, substituting documented defaults for nulls.

    Unlike the in-season contract this tolerates missing values, because they are
    expected: a rookie has no prior production, and the optional feeds (snap counts,
    depth charts) can be unavailable. Silently defaulting is the right call here as long
    as the defaults are explicit and reviewable - see PRESEASON_DEFAULTS.
    """
    unexpected = [k for k in features if k not in PRESEASON_FEATURE_ORDER]
    if unexpected:
        raise FeatureContractError(f"unexpected preseason features: {unexpected}")
    resolved: dict[str, float] = {}
    for name in PRESEASON_FEATURE_ORDER:
        raw = features.get(name)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raw = PRESEASON_DEFAULTS[name]
        resolved[name] = float(raw)

    # Callers pass raw draft position; the model is trained on the decayed version. Doing
    # it here rather than in the pipeline keeps one definition for training and serving,
    # so the two cannot drift.
    resolved["draft_round"], resolved["draft_pick"] = decay_draft_capital(
        resolved["draft_round"], resolved["draft_pick"], resolved["career_games"]
    )
    return np.asarray([[resolved[n] for n in PRESEASON_FEATURE_ORDER]], dtype=np.float32)


def preseason_matrix(rows: list[Mapping[str, float | None]]) -> np.ndarray:
    return np.vstack([preseason_row(r) for r in rows])
