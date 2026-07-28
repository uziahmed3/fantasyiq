"""Shared fixtures.

The monotonicity and attribution tests need a real gradient-boosted artifact, not a stub:
a monotonic constraint is enforced by XGBoost at split time, so a fake predictor would
pass the test while proving nothing. One small model is trained once per session on
synthetic data and reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from app.features import PRESEASON_FEATURE_ORDER, PRESEASON_SCHEMA_VERSION

VERSION = "preseason_test_v1"


@pytest.fixture(scope="session")
def trained_preseason_model(tmp_path_factory):
    """A constrained XGBoost model trained on deliberately adversarial synthetic data.

    The target rewards production but is built so that a naive fit would ALSO reward a
    good draft position more than it should, and would fall apart at the top of the
    production range - the two failure modes seen in real training. If the constraints
    are working, the monotonicity assertions hold anyway.
    """
    import xgboost as xgb

    from train.train_preseason import monotone_constraints

    rng = np.random.default_rng(11)
    n = 1500
    idx = {name: i for i, name in enumerate(PRESEASON_FEATURE_ORDER)}
    x = np.zeros((n, len(PRESEASON_FEATURE_ORDER)), dtype="float32")

    career = rng.gamma(3.0, 3.0, n).clip(0, 30)
    x[:, idx["career_weighted_ppg"]] = career
    x[:, idx["prior_points_per_game"]] = (career + rng.normal(0, 3, n)).clip(0)
    x[:, idx["career_best_ppg"]] = career + rng.gamma(1.5, 1.5, n)
    x[:, idx["draft_pick"]] = rng.integers(1, 260, n)
    x[:, idx["depth_chart_rank"]] = rng.integers(1, 5, n)
    x[:, idx["career_games"]] = rng.integers(0, 90, n)
    x[:, idx["age"]] = rng.normal(26, 3, n).clip(21, 36)
    x[:, idx["teammate_top_target_share"]] = rng.uniform(0, 0.35, n)

    # Sparse at the top end, which is what lets an unconstrained model misbehave there.
    y = (
        0.7 * career
        - 0.004 * x[:, idx["draft_pick"]]
        - 1.2 * x[:, idx["depth_chart_rank"]]
        + rng.normal(0, 2.0, n)
    ).clip(0)

    model = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        monotone_constraints=monotone_constraints(),
        random_state=3,
        n_jobs=2,
    )
    model.fit(x, y)
    return model


@pytest.fixture(scope="session")
def preseason_registry(trained_preseason_model, tmp_path_factory):
    """A registry pointed at a directory holding the test artifact plus its metadata."""
    from app.registry import ModelRegistry

    model_dir: Path = tmp_path_factory.mktemp("models")
    joblib.dump(trained_preseason_model, model_dir / f"{VERSION}.joblib")
    (model_dir / f"{VERSION}.json").write_text(
        json.dumps(
            {
                "framework": "xgboost",
                "kind": "preseason",
                "feature_order": list(PRESEASON_FEATURE_ORDER),
                "feature_schema_version": PRESEASON_SCHEMA_VERSION,
                "residual_std": 2.9,
            }
        )
    )
    return ModelRegistry(model_dir=model_dir), VERSION
