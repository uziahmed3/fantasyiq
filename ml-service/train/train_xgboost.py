"""XGBoost with early stopping on the chronological holdout."""

import joblib
import xgboost as xgb

from train.common import metrics, report, write_metadata
from train.dataset import ARTIFACT_DIR, load_dataset, time_split

VERSION = "xgboost_v1"

PARAMS = dict(  # noqa: C408 - kwargs form mirrors the XGBRegressor signature
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_lambda=1.5,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=40,
    random_state=7,
    n_jobs=4,
)


def main() -> dict:
    df, source = load_dataset()
    x_tr, y_tr, x_va, y_va, cutoff = time_split(df)

    model = xgb.XGBRegressor(**PARAMS)
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    scores = metrics(y_va, model.predict(x_va))

    from app.features import FEATURE_ORDER

    importances = dict(
        sorted(
            zip(
                FEATURE_ORDER, [round(float(v), 4) for v in model.feature_importances_], strict=True
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
            "params": {k: v for k, v in PARAMS.items() if k != "n_jobs"},
            "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
            "feature_importances": importances,
        },
    )
    report(VERSION, scores, source, cutoff)
    print("top features       :", list(importances)[:4])
    return scores


if __name__ == "__main__":
    main()
