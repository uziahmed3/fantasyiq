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
