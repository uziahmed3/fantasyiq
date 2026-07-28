"""Model bake-off, split by which question each model answers.

An earlier version of this printed one table containing both model families and declared
a winner by lowest RMSE. That was wrong, and dangerously so: the two families predict
different targets on different holdouts.

    in-season : points in ONE upcoming week, held out by week
    preseason : mean points per game over weeks 1-4, held out by season

Their RMSEs are not comparable - the preseason target is an average of four games, so it
is inherently less variable and scores lower no matter how good either model is. On real
2021-2025 data the old table reported "preseason_v1 3.99 vs xgboost_v1 5.97, best:
preseason_v1, set ACTIVE_MODEL_VERSION=preseason_v1" - and following that advice would
have broken the in-season endpoint outright, because the preseason artifact declares a
different feature contract and the registry correctly refuses to serve it there.

So: two tables, each against its own baseline, and two separate promotion hints.
"""

import json
import sys

import numpy as np

from app.features import FEATURE_ORDER
from train.dataset import ARTIFACT_DIR, load_dataset, time_split


def naive_in_season() -> dict:
    """Last 3 weeks' average - the free heuristic the in-season model must beat."""
    df, _ = load_dataset()
    _, _, x_va, y_va, _ = time_split(df)
    preds = x_va[:, FEATURE_ORDER.index("fantasy_points_last_3")]
    err = preds - y_va
    return {
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "mae": round(float(np.mean(np.abs(err))), 4),
    }


def _load_metadata() -> tuple[list[dict], list[dict]]:
    """Split artifacts into (in-season, preseason) by the `kind` they declare."""
    in_season, preseason = [], []
    for meta_path in sorted(ARTIFACT_DIR.glob("*.json")):
        m = json.loads(meta_path.read_text())
        if "rmse" not in m:
            continue
        (preseason if m.get("kind") == "preseason" else in_season).append(m)
    return in_season, preseason


def _table(title: str, subtitle: str, rows: list[tuple], baseline_label: str) -> str | None:
    """Print one comparison table. Returns the winning version, or None."""
    if not rows:
        return None
    width = max(len(r[0]) for r in rows) + 2
    print(f"\n{title}")
    print(f"  {subtitle}")
    print(f"\n{'model'.ljust(width)}{'framework':<12}{'RMSE':>9}{'MAE':>9}{'vs baseline':>13}")
    print("-" * (width + 43))
    for name, fw, rmse, mae, lift in sorted(rows, key=lambda r: r[2]):
        print(f"{name.ljust(width)}{fw:<12}{rmse:>9.4f}{mae:>9.4f}{lift:>13}")
    print(f"  baseline = {baseline_label}")
    # The baseline row is not a candidate for promotion.
    trained = [r for r in rows if r[4] != "-"]
    return min(trained, key=lambda r: r[2])[0] if trained else None


def main() -> int:
    in_season_meta, preseason_meta = _load_metadata()
    if not in_season_meta and not preseason_meta:
        print("No trained artifacts found in", ARTIFACT_DIR)
        print("Run: python -m train.train_baseline && python -m train.train_xgboost")
        return 1

    # ---------------------------------------------------------------- in-season
    best_in_season = None
    if in_season_meta:
        naive = naive_in_season()
        rows = [("naive_last3_avg", "none", naive["rmse"], naive["mae"], "-")]
        for m in in_season_meta:
            lift = (naive["rmse"] - m["rmse"]) / naive["rmse"] * 100
            rows.append(
                (m["version"], m.get("framework", "?"), m["rmse"], m["mae"], f"{lift:+.1f}%")
            )
        best_in_season = _table(
            "IN-SEASON  - points in the next single week",
            "held out by week; used once a player has games this season",
            rows,
            "last 3 weeks' average",
        )

    # ---------------------------------------------------------------- preseason
    best_preseason = None
    if preseason_meta:
        rows = []
        carry = None
        for m in preseason_meta:
            carry = m.get("naive_carry_forward_rmse")
            lift = m.get("lift_vs_naive_pct")
            rows.append(
                (
                    m["version"],
                    m.get("framework", "?"),
                    m["rmse"],
                    m["mae"],
                    f"{lift:+.1f}%" if lift is not None else "?",
                )
            )
        if carry is not None:
            # Carry-forward has no MAE recorded, so show it with the RMSE only.
            rows.append(("naive_carry_forward", "none", carry, float("nan"), "-"))
        best_preseason = _table(
            "PRESEASON  - mean points per game over weeks 1-4",
            "held out by season; used in week 1 and for every rookie",
            rows,
            "last season's points per game, carried forward",
        )

    print("\n" + "=" * 66)
    print("These two tables are NOT comparable with each other: different targets")
    print("(one week vs a four-week average) and different holdouts.")
    print("=" * 66)
    if best_in_season:
        print(f"  in-season winner : ACTIVE_MODEL_VERSION={best_in_season}")
    if best_preseason:
        print(f"  preseason winner : ACTIVE_PRESEASON_MODEL_VERSION={best_preseason}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
