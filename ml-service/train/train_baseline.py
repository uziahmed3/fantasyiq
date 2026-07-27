"""Baseline: ridge regression on the same features.

Every model comparison needs a floor. If XGBoost cannot beat a linear model on these
features, the extra complexity is not earning its deployment cost.
"""

import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train.common import metrics, report, write_metadata
from train.dataset import ARTIFACT_DIR, load_dataset, time_split

VERSION = "baseline_v1"


def main() -> dict:
    df, source = load_dataset()
    x_tr, y_tr, x_va, y_va, cutoff = time_split(df)

    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(x_tr, y_tr)
    scores = metrics(y_va, model.predict(x_va))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / f"{VERSION}.joblib")
    write_metadata(VERSION, "sklearn", scores, {"estimator": "Ridge(alpha=1.0)"})
    report(VERSION, scores, source, cutoff)
    return scores


if __name__ == "__main__":
    main()
