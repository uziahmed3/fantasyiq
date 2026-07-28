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
PRESEASON_SCHEMA_VERSION = "ps1"

PRESEASON_FEATURE_ORDER: tuple[str, ...] = (
    "prior_points_per_game",
    "prior_last4_points_per_game",
    "prior_targets_per_game",
    "prior_target_share",
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
)

# Sentinels for genuinely-unknown values. Undrafted is not round 0 - it is worse than
# round 7, so it maps to 8. An unknown depth rank maps to 5 (buried, not starting).
PRESEASON_DEFAULTS: dict[str, float] = {
    "prior_points_per_game": 0.0,
    "prior_last4_points_per_game": 0.0,
    "prior_targets_per_game": 0.0,
    "prior_target_share": 0.0,
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
}


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
    values = []
    for name in PRESEASON_FEATURE_ORDER:
        raw = features.get(name)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raw = PRESEASON_DEFAULTS[name]
        values.append(float(raw))
    return np.asarray([values], dtype=np.float32)


def preseason_matrix(rows: list[Mapping[str, float | None]]) -> np.ndarray:
    return np.vstack([preseason_row(r) for r in rows])
