"""Train the preseason model: project a player before he has played a game this season.

Different problem from the in-season model, so it gets its own dataset, its own feature
contract and its own artifact:

    in-season  : "given his last 3 games, what does he do next week?"
    preseason  : "given last season, his role, and his draft capital, what does he
                  average early this season?"

Target is mean points per game over weeks 1-4 of the projected season. Weeks 1-4 rather
than the full season because that is the window where no in-season form exists yet - past
about week 4 the in-season model has enough history and takes over.

Training rows are (player, season) pairs: context assembled from season S-1 paired with
what actually happened in weeks 1-4 of season S. That means N seasons of ingested data
yields N-1 usable seasons, so ingest several years if you want this model to be any good.

Validation holds out the most recent season entirely - the honest test, since predicting
a season you trained on tells you nothing about next August.
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.features import PRESEASON_DEFAULTS, PRESEASON_FEATURE_ORDER, PRESEASON_SCHEMA_VERSION
from train.common import metrics, write_metadata
from train.dataset import ARTIFACT_DIR, _database_url

VERSION = "preseason_v1"
EARLY_WEEKS = 4

DATASET_SQL = text("""
SELECT
    pc.player_id,
    pc.season,
    p.position,
    pc.prior_points_per_game,
    pc.prior_last4_points_per_game,
    pc.prior_targets_per_game,
    pc.prior_target_share,
    pc.prior_yards_per_game,
    pc.prior_games,
    pc.prior_snap_share,
    pc.depth_chart_rank,
    pc.draft_round,
    pc.draft_pick,
    pc.years_experience,
    pc.is_rookie,
    pc.age,
    pc.team_pass_attempts_prior,
    pc.qb_changed,
    early.actual_ppg,
    early.games_played
FROM player_context pc
JOIN players p ON p.id = pc.player_id
JOIN (
    SELECT player_id, season,
           AVG(fantasy_points) AS actual_ppg,
           COUNT(*)            AS games_played
    FROM player_stats
    WHERE week <= :early_weeks
    GROUP BY player_id, season
) early
  ON early.player_id = pc.player_id AND early.season = pc.season
ORDER BY pc.season, pc.player_id
""")


def load_dataset() -> pd.DataFrame:
    engine = create_engine(_database_url(), pool_pre_ping=True)
    with engine.connect() as conn:
        df = pd.read_sql(DATASET_SQL, conn, params={"early_weeks": EARLY_WEEKS})
    engine.dispose()
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("is_rookie", "qb_changed"):
        out[col] = out[col].fillna(0).astype(int)
    for col in PRESEASON_FEATURE_ORDER:
        if col not in out.columns:
            out[col] = PRESEASON_DEFAULTS[col]
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(PRESEASON_DEFAULTS[col])
    return out.dropna(subset=["actual_ppg"])


def season_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Hold out the most recent season entirely.

    Splitting within a season would let the same season's team context appear on both
    sides, which is not the question being asked - the question is "does this generalise
    to a season we have never seen".
    """
    seasons = sorted(df["season"].unique())
    holdout = seasons[-1]
    train, valid = df[df["season"] != holdout], df[df["season"] == holdout]
    cols = list(PRESEASON_FEATURE_ORDER)
    return (
        train[cols].to_numpy("float32"),
        train["actual_ppg"].to_numpy("float32"),
        valid[cols].to_numpy("float32"),
        valid["actual_ppg"].to_numpy("float32"),
        int(holdout),
    )


def naive_baseline(x_valid: np.ndarray) -> np.ndarray:
    """Last season's points per game, carried forward unchanged.

    This is what a fantasy player does for free, and it is a genuinely hard baseline to
    beat. If the model cannot, it is not earning its complexity.
    """
    return x_valid[:, PRESEASON_FEATURE_ORDER.index("prior_points_per_game")]


def main() -> int:
    df = load_dataset()
    if df.empty:
        print(
            "No training rows. The preseason model needs player_context for a season "
            "AND actual results for weeks 1-4 of that same season.\n"
            "Ingest at least two consecutive seasons, then:\n"
            "  python -m context --season <later season>\n"
        )
        return 1

    df = prepare(df)
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        print(
            f"Only one season of paired data ({seasons}). Training would have no holdout.\n"
            "Ingest more seasons - three or more makes this model meaningful."
        )
        return 1

    x_tr, y_tr, x_va, y_va, holdout = season_split(df)
    if len(x_va) < 20:
        print(f"Holdout season {holdout} has only {len(x_va)} rows; not enough to evaluate.")
        return 1

    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=4,  # shallower than the in-season model: far fewer training rows
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=8,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=40,
        random_state=7,
        n_jobs=4,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)

    scores = metrics(y_va, model.predict(x_va))
    base = metrics(y_va, naive_baseline(x_va))
    lift = (base["rmse"] - scores["rmse"]) / base["rmse"] * 100 if base["rmse"] else 0.0

    importances = dict(
        sorted(
            zip(
                PRESEASON_FEATURE_ORDER,
                [round(float(v), 4) for v in model.feature_importances_],
                strict=True,
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / f"{VERSION}.joblib")
    write_metadata(
        VERSION,
        "xgboost",
        scores,
        {
            "kind": "preseason",
            "feature_order": list(PRESEASON_FEATURE_ORDER),
            "feature_schema_version": PRESEASON_SCHEMA_VERSION,
            "target": f"mean fantasy points, weeks 1-{EARLY_WEEKS}",
            "train_seasons": [int(s) for s in seasons[:-1]],
            "holdout_season": holdout,
            "naive_carry_forward_rmse": base["rmse"],
            "lift_vs_naive_pct": round(lift, 1),
            "feature_importances": importances,
            "rookies_in_holdout": int((df[df["season"] == holdout]["is_rookie"] == 1).sum()),
        },
    )

    print(f"\n=== {VERSION} ===")
    print(f"train seasons      : {seasons[:-1]}")
    print(f"holdout season     : {holdout}   ({len(x_va)} players)")
    print(f"target             : mean points/game, weeks 1-{EARLY_WEEKS}")
    for k, v in scores.items():
        print(f"{k:19}: {v}")
    print(f"{'naive carry-forward':19}: rmse {base['rmse']}")
    print(f"{'lift vs naive':19}: {lift:+.1f}%")
    print("top features       :", list(importances)[:5])

    rookies = df[(df["season"] == holdout) & (df["is_rookie"] == 1)]
    if len(rookies):
        rx = rookies[list(PRESEASON_FEATURE_ORDER)].to_numpy("float32")
        rookie_scores = metrics(rookies["actual_ppg"].to_numpy("float32"), model.predict(rx))
        print(
            f"\nrookies only ({len(rookies)}): rmse {rookie_scores['rmse']}  "
            f"mae {rookie_scores['mae']}"
        )
        print("(rookies are the hard case - no prior production, only draft capital and role)")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MODEL_DIR", str(ARTIFACT_DIR))
    sys.exit(main())
