"""Model bake-off. Reads every artifact's metadata and prints a comparison table
against the naive 'last 3 weeks average' rule any fantasy player already uses for free.
"""

import json
import sys

import numpy as np

from app.features import FEATURE_ORDER
from train.dataset import ARTIFACT_DIR, load_dataset, time_split


def naive_scores() -> dict:
    df, _ = load_dataset()
    _, _, x_va, y_va, _ = time_split(df)
    preds = x_va[:, FEATURE_ORDER.index("fantasy_points_last_3")]
    err = preds - y_va
    return {
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "mae": round(float(np.mean(np.abs(err))), 4),
    }


def main() -> int:
    naive = naive_scores()
    rows = [("naive_last3_avg", "none", naive["rmse"], naive["mae"], "-")]

    for meta_path in sorted(ARTIFACT_DIR.glob("*.json")):
        m = json.loads(meta_path.read_text())
        if "rmse" not in m:
            continue
        lift = (naive["rmse"] - m["rmse"]) / naive["rmse"] * 100
        rows.append((m["version"], m.get("framework", "?"), m["rmse"], m["mae"], f"{lift:+.1f}%"))

    if len(rows) == 1:
        print("No trained artifacts found in", ARTIFACT_DIR)
        print("Run: python -m train.train_baseline && python -m train.train_xgboost")
        return 1

    width = max(len(r[0]) for r in rows) + 2
    print(f"\n{'model'.ljust(width)}{'framework':<12}{'RMSE':>9}{'MAE':>9}{'vs naive':>11}")
    print("-" * (width + 41))
    for name, fw, rmse, mae, lift in sorted(rows, key=lambda r: r[2]):
        print(f"{name.ljust(width)}{fw:<12}{rmse:>9.4f}{mae:>9.4f}{lift:>11}")

    best = min(rows, key=lambda r: r[2])
    print(f"\nbest: {best[0]}  (set ACTIVE_MODEL_VERSION={best[0]} to serve it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
