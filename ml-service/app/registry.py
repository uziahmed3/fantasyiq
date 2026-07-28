"""Model registry: versioned artifacts on disk, loaded lazily and cached in-process.

Layout (MODEL_DIR, an EFS mount in AWS / a docker volume locally):
    /models/xgboost_v1.joblib
    /models/xgboost_v1.json        <- metadata: framework, feature order, metrics
    /models/torch_v1.pt
    /models/torch_v1.json

Rolling back a bad model is `ACTIVE_MODEL_VERSION=xgboost_v1` and a restart - no
rebuild, no retrain, no code change. That is the whole point of versioning artifacts
separately from the service image.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.features import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    PRESEASON_FEATURE_ORDER,
    preseason_row,
    to_row,
)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))

# Served when MODEL_DIR is empty (fresh clone, before `make train`) so that
# `docker compose up` produces a working end-to-end system.
FALLBACK_VERSION = "heuristic_fallback_v0"
PRESEASON_FALLBACK_VERSION = "preseason_heuristic_v0"


class ModelNotFound(LookupError):
    pass


@dataclass
class LoadedModel:
    version: str
    framework: str
    predict_fn: Any
    metadata: dict


def _load_torch(path: Path, metadata: dict) -> Any:
    import torch

    from app.torch_model import FantasyMLP

    model = FantasyMLP(n_features=len(FEATURE_ORDER), hidden=metadata.get("hidden", 64))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    scaler = None
    scaler_path = path.with_suffix(".scaler.joblib")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    def predict(x: np.ndarray) -> np.ndarray:
        if scaler is not None:
            x = scaler.transform(x)
        with torch.no_grad():
            return model(torch.from_numpy(x.astype("float32"))).numpy().ravel()

    return predict


def _load_sklearn(path: Path, metadata: dict) -> Any:
    estimator = joblib.load(path)

    def predict(x: np.ndarray) -> np.ndarray:
        return np.asarray(estimator.predict(x)).ravel()

    return predict


def _preseason_heuristic(x: np.ndarray) -> np.ndarray:
    """Baseline preseason projection, used until preseason_v1 is trained.

    Carry last season's rate forward, weighted toward the last 4 games, then adjust for
    role. For a player with no prior production - every rookie - the estimate comes
    entirely from draft capital and depth-chart rank, which is genuinely the best
    available information in August.
    """
    idx = {name: i for i, name in enumerate(PRESEASON_FEATURE_ORDER)}
    prior_ppg = x[:, idx["prior_points_per_game"]]
    last4 = x[:, idx["prior_last4_points_per_game"]]
    games = x[:, idx["prior_games"]]
    depth = x[:, idx["depth_chart_rank"]]
    draft_pick = x[:, idx["draft_pick"]]
    is_rookie = x[:, idx["is_rookie"]]
    qb_changed = x[:, idx["qb_changed"]]
    snap = x[:, idx["prior_snap_share"]]
    carry_share = x[:, idx["prior_carry_share"]]

    # Veterans: recency-weighted carry-forward, shrunk toward zero when the sample is
    # small (a 2-game season is weak evidence of a rate).
    carry = 0.6 * last4 + 0.4 * prior_ppg
    confidence = np.clip(games / 8.0, 0.0, 1.0)
    # Shrink toward a modest baseline rather than toward zero. A player with 4 good games
    # is weak evidence of a high rate, but it is not evidence that he scores nothing -
    # shrinking to zero would systematically under-project every partial season.
    PRIOR_MEAN = 6.0
    veteran = carry * confidence + PRIOR_MEAN * (1.0 - confidence)

    # Rookies: draft capital is the signal. Pick 1 overall lands around 12 points/game,
    # decaying with pick number; a late pick projects near replacement level.
    rookie = 13.0 * np.exp(-np.clip(draft_pick, 1, 300) / 90.0)

    base = np.where(is_rookie > 0.5, rookie, veteran)

    # Role multipliers: starters see the volume, backups do not.
    depth_factor = np.where(depth <= 1, 1.15, np.where(depth <= 2, 1.0, 0.65))
    snap_factor = 1.0 + 0.25 * (np.clip(snap, 0, 1) - 0.5)
    # A lead back (high carry share) is a different projection from a rotational one.
    # Neutral at 0 so this cannot penalise receivers, who legitimately have no carries.
    carry_factor = 1.0 + 0.30 * np.clip(carry_share, 0, 1)
    role_factor = depth_factor * snap_factor * carry_factor

    # Apply role only to the extent we lack production history. A veteran's prior points
    # per game ALREADY reflects that he was the WR1 on 85% of snaps - multiplying by a
    # role bonus on top double-counts it and inflates the projection. So the adjustment
    # fades out as history accumulates, and carries full weight for a rookie.
    role_weight = 1.0 - confidence
    effective_role = 1.0 + (role_factor - 1.0) * np.where(is_rookie > 0.5, 1.0, role_weight)

    # A new quarterback is a genuine risk regardless of how much history the receiver has.
    qb_factor = np.where(qb_changed > 0.5, 0.93, 1.0)

    return np.clip(base * effective_role * qb_factor, 0.0, None)


def _heuristic(x: np.ndarray) -> np.ndarray:
    """Recency-weighted average of last-1 and last-3 points, nudged by matchup.

    Not a model. It exists so the service is never down for lack of an artifact, and it
    is the honest baseline any real model has to beat.
    """
    last1 = x[:, FEATURE_ORDER.index("fantasy_points_last_1")]
    last3 = x[:, FEATURE_ORDER.index("fantasy_points_last_3")]
    season = x[:, FEATURE_ORDER.index("season_avg_points")]
    opp_rank = x[:, FEATURE_ORDER.index("opponent_rank")]
    base = 0.45 * last3 + 0.3 * last1 + 0.25 * season
    # rank 1 = softest defence (most points allowed) -> small boost
    matchup = 1.0 + (16.5 - opp_rank) / 165.0
    return np.clip(base * matchup, 0.0, None)


class ModelRegistry:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or MODEL_DIR
        self._cache: dict[str, LoadedModel] = {}

    def available(self) -> list[str]:
        if not self.model_dir.exists():
            return [FALLBACK_VERSION, PRESEASON_FALLBACK_VERSION]
        versions = sorted(
            {p.stem for p in self.model_dir.iterdir() if p.suffix in {".joblib", ".pt"}}
            - {p.stem for p in self.model_dir.glob("*.scaler.joblib")}
        )
        return versions + [FALLBACK_VERSION, PRESEASON_FALLBACK_VERSION]

    def get(self, version: str) -> LoadedModel:
        if version in self._cache:
            return self._cache[version]

        if version == PRESEASON_FALLBACK_VERSION:
            model = LoadedModel(
                version=PRESEASON_FALLBACK_VERSION,
                framework="heuristic",
                predict_fn=_preseason_heuristic,
                metadata={
                    "kind": "preseason",
                    "feature_order": list(PRESEASON_FEATURE_ORDER),
                    "residual_std": 5.0,
                    "note": (
                        "No trained preseason artifact; serving the documented "
                        "carry-forward + draft-capital baseline."
                    ),
                },
            )
            self._cache[version] = model
            return model

        if version == FALLBACK_VERSION:
            model = LoadedModel(
                version=FALLBACK_VERSION,
                framework="heuristic",
                predict_fn=_heuristic,
                metadata={
                    "feature_order": list(FEATURE_ORDER),
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "residual_std": 6.5,
                    "note": "No trained artifact found; serving the documented baseline.",
                },
            )
            self._cache[version] = model
            return model

        meta_path = self.model_dir / f"{version}.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Which contract does this artifact belong to? Preseason models are trained on a
        # different feature set, so validating them against the in-season order would
        # reject every one of them. The guard still fires for a genuine mismatch.
        is_preseason = metadata.get("kind") == "preseason"
        expected = PRESEASON_FEATURE_ORDER if is_preseason else FEATURE_ORDER
        trained_order = metadata.get("feature_order")
        if trained_order and tuple(trained_order) != expected:
            raise ModelNotFound(
                f"{version} was trained on a different feature contract "
                f"({metadata.get('feature_schema_version')}, kind="
                f"{metadata.get('kind', 'in_season')}); retrain before serving"
            )

        joblib_path = self.model_dir / f"{version}.joblib"
        torch_path = self.model_dir / f"{version}.pt"
        if joblib_path.exists():
            model = LoadedModel(
                version,
                metadata.get("framework", "sklearn"),
                _load_sklearn(joblib_path, metadata),
                metadata,
            )
        elif torch_path.exists():
            model = LoadedModel(version, "pytorch", _load_torch(torch_path, metadata), metadata)
        else:
            raise ModelNotFound(f"no artifact for version '{version}' in {self.model_dir}")

        self._cache[version] = model
        return model

    def predict(self, version: str, features: dict) -> tuple[float, float, str]:
        model = self.get(version)
        row = to_row(features)
        value = float(model.predict_fn(row)[0])
        return value, self._confidence(value, features, model), model.version

    def predict_preseason(self, version: str, features: dict) -> tuple[float, float, str]:
        """Preseason path: tolerates missing inputs, and reports lower confidence.

        A preseason number is inherently less certain than an in-season one - there is no
        current-season evidence in it at all - and the confidence score says so rather
        than presenting both as equally trustworthy.
        """
        model = self.get(version)
        row = preseason_row(features)
        value = float(model.predict_fn(row)[0])
        return value, self._preseason_confidence(features, model), model.version

    @staticmethod
    def _preseason_confidence(features: dict, model: LoadedModel) -> float:
        """How much do we actually know about this player?

        Driven by evidence available, not by the size of the number: a veteran with a
        full prior season and a known role is a far safer projection than a rookie whose
        only input is where he was drafted.
        """
        residual_std = float(model.metadata.get("residual_std", 5.5))
        prior_games = float(features.get("prior_games") or 0)
        has_role = features.get("depth_chart_rank") is not None
        has_snaps = features.get("prior_snap_share") is not None
        is_rookie = float(features.get("is_rookie") or 0) > 0.5

        history = min(prior_games / 12.0, 1.0)
        role = (0.5 if has_role else 0.0) + (0.5 if has_snaps else 0.0)
        # Ceiling for rookies: draft capital is real signal but it is not evidence of
        # NFL production, and pretending otherwise would be the dishonest choice.
        ceiling = 0.55 if is_rookie else 0.85
        spread = 1.0 - residual_std / (residual_std + 8.0)
        score = 0.45 * history + 0.25 * role + 0.30 * spread
        return round(float(np.clip(score, 0.05, ceiling)), 3)

    @staticmethod
    def _confidence(prediction: float, features: dict, model: LoadedModel) -> float:
        """Heuristic reliability score, NOT a calibrated probability.

        Two things drive it: the model's validation residual spread relative to the size
        of the prediction, and how much history the player actually has. A week-1
        projection off zero games is reported as low confidence rather than pretending
        the point estimate is as trustworthy as a week-10 one.
        """
        residual_std = float(model.metadata.get("residual_std", 6.0))
        signal = abs(prediction) + residual_std
        spread_score = 1.0 - (residual_std / signal) if signal > 0 else 0.0
        games = float(features.get("games_played", 0))
        history_score = min(games / 4.0, 1.0)
        return round(float(np.clip(0.65 * spread_score + 0.35 * history_score, 0.05, 0.95)), 3)


registry = ModelRegistry()
