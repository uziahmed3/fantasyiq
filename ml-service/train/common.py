import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.features import FEATURE_ORDER, FEATURE_SCHEMA_VERSION
from train.dataset import ARTIFACT_DIR


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else 0.0,
        "residual_std": round(float(np.std(err)), 4),
        "n_valid": int(len(y_true)),
    }


def write_metadata(version: str, framework: str, scores: dict, extra: dict | None = None) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "framework": framework,
        "feature_order": list(FEATURE_ORDER),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **scores,
        **(extra or {}),
    }
    path = ARTIFACT_DIR / f"{version}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def report(version: str, scores: dict, source: str, cutoff: int) -> None:
    print(f"\n=== {version} ===")
    print(f"data source        : {source}")
    print(f"train weeks        : <= {cutoff}   validation weeks: > {cutoff}")
    for k, v in scores.items():
        print(f"{k:19}: {v}")
