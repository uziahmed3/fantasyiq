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

from app.features import FEATURE_ORDER, FEATURE_SCHEMA_VERSION, to_row

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))

# Served when MODEL_DIR is empty (fresh clone, before `make train`) so that
# `docker compose up` produces a working end-to-end system.
FALLBACK_VERSION = "heuristic_fallback_v0"


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
            return [FALLBACK_VERSION]
        versions = sorted(
            {p.stem for p in self.model_dir.iterdir() if p.suffix in {".joblib", ".pt"}}
            - {p.stem for p in self.model_dir.glob("*.scaler.joblib")}
        )
        return versions + [FALLBACK_VERSION]

    def get(self, version: str) -> LoadedModel:
        if version in self._cache:
            return self._cache[version]

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

        trained_order = metadata.get("feature_order")
        if trained_order and tuple(trained_order) != FEATURE_ORDER:
            raise ModelNotFound(
                f"{version} was trained on a different feature contract "
                f"({metadata.get('feature_schema_version')}); retrain before serving"
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
