import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.features import (
    FEATURE_ORDER,
    PRESEASON_FEATURE_ORDER,
    FeatureContractError,
    to_matrix,
    to_row,
)
from app.main import app
from app.registry import FALLBACK_VERSION, PRESEASON_FALLBACK_VERSION, ModelRegistry

client = TestClient(app)

GOOD = {
    "targets_last_3": 11.0,
    "receptions_last_3": 8.0,
    "yards_last_3": 105.0,
    "touchdowns_last_3": 0.67,
    "fantasy_points_last_3": 18.2,
    "fantasy_points_last_1": 17.0,
    "season_avg_points": 16.4,
    "games_played": 4,
    "opponent_rank": 20,
    "is_home": 1,
}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_models_endpoint_exposes_both_contracts():
    """Two models, two feature contracts - /models must describe both, because a caller
    building either kind of request needs to know the shape."""
    body = client.get("/models").json()
    assert body["in_season"]["feature_order"] == list(FEATURE_ORDER)
    assert body["preseason"]["feature_order"] == list(PRESEASON_FEATURE_ORDER)
    assert FALLBACK_VERSION in body["available"]
    assert PRESEASON_FALLBACK_VERSION in body["available"]


def test_preseason_tolerates_a_player_we_know_nothing_about():
    """An empty payload is the rookie-with-no-draft-data case; it must still answer."""
    r = client.post("/predict/preseason", json={"features": {}})
    assert r.status_code == 200
    assert r.json()["prediction"] >= 0


def test_preseason_rewards_earlier_draft_picks():
    def project(pick):
        return client.post(
            "/predict/preseason",
            json={
                "features": {"is_rookie": 1, "draft_pick": pick, "depth_chart_rank": 2},
                "model_version": PRESEASON_FALLBACK_VERSION,
            },
        ).json()["prediction"]

    assert project(5) > project(60) > project(250)


def test_preseason_penalises_a_quarterback_change():
    base = {
        "prior_points_per_game": 15.0,
        "prior_last4_points_per_game": 15.0,
        "prior_games": 16,
        "depth_chart_rank": 1,
        "prior_snap_share": 0.85,
        "is_rookie": 0,
    }

    def project(**extra):
        return client.post(
            "/predict/preseason",
            json={"features": {**base, **extra}, "model_version": PRESEASON_FALLBACK_VERSION},
        ).json()["prediction"]

    assert project(qb_changed=1) < project(qb_changed=0)


def test_preseason_confidence_is_lower_for_rookies():
    """A rookie projection rests on draft position, not production. Reporting it as
    equally confident as a veteran's would be the dishonest choice."""
    rookie = client.post(
        "/predict/preseason",
        json={"features": {"is_rookie": 1, "draft_pick": 6}},
    ).json()["confidence"]
    veteran = client.post(
        "/predict/preseason",
        json={
            "features": {
                "is_rookie": 0,
                "prior_games": 16,
                "prior_points_per_game": 15.0,
                "depth_chart_rank": 1,
                "prior_snap_share": 0.8,
            }
        },
    ).json()["confidence"]
    assert rookie < veteran


def test_preseason_batch_matches_single_calls():
    items = [
        {"is_rookie": 1, "draft_pick": 6},
        {"is_rookie": 0, "prior_games": 16, "prior_points_per_game": 12.0},
    ]
    batch = client.post(
        "/predict/preseason/batch",
        json={"items": items, "model_version": PRESEASON_FALLBACK_VERSION},
    ).json()["predictions"]
    singles = [
        client.post(
            "/predict/preseason",
            json={"features": i, "model_version": PRESEASON_FALLBACK_VERSION},
        ).json()["prediction"]
        for i in items
    ]
    assert np.allclose(batch, singles, atol=1e-3)


def test_preseason_rejects_unknown_feature_names():
    from app.features import FeatureContractError, preseason_row

    with pytest.raises(FeatureContractError):
        preseason_row({"not_a_feature": 1.0})


def test_feature_order_row_shape():
    row = to_row(GOOD)
    assert row.shape == (1, len(FEATURE_ORDER))
    assert row[0][FEATURE_ORDER.index("yards_last_3")] == pytest.approx(105.0)


def test_missing_feature_is_rejected():
    bad = {k: v for k, v in GOOD.items() if k != "yards_last_3"}
    with pytest.raises(FeatureContractError):
        to_row(bad)


def test_unexpected_feature_is_rejected():
    with pytest.raises(FeatureContractError):
        to_row({**GOOD, "vibes": 1.0})


def test_predict_returns_sane_projection():
    r = client.post("/predict", json={"features": GOOD, "model_version": FALLBACK_VERSION})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["prediction"] <= 60
    assert 0.05 <= body["confidence"] <= 0.95
    assert body["model_version"] == FALLBACK_VERSION


def test_unknown_version_degrades_to_fallback_not_500():
    r = client.post("/predict", json={"features": GOOD, "model_version": "does_not_exist_v9"})
    assert r.status_code == 200
    assert r.json()["model_version"] == FALLBACK_VERSION


def test_out_of_range_feature_is_422():
    r = client.post("/predict", json={"features": {**GOOD, "opponent_rank": 99}})
    assert r.status_code == 422


def test_batch_matches_single_predictions():
    items = [GOOD, {**GOOD, "fantasy_points_last_3": 4.0, "fantasy_points_last_1": 3.0}]
    batch = client.post(
        "/predict/batch", json={"items": items, "model_version": FALLBACK_VERSION}
    ).json()["predictions"]
    singles = [
        client.post("/predict", json={"features": i, "model_version": FALLBACK_VERSION}).json()[
            "prediction"
        ]
        for i in items
    ]
    assert np.allclose(batch, singles, atol=1e-3)


def test_confidence_rises_with_more_history():
    reg = ModelRegistry()
    _, low, _ = reg.predict(FALLBACK_VERSION, {**GOOD, "games_played": 0})
    _, high, _ = reg.predict(FALLBACK_VERSION, {**GOOD, "games_played": 8})
    assert high > low


def test_better_matchup_raises_projection():
    reg = ModelRegistry()
    soft, _, _ = reg.predict(FALLBACK_VERSION, {**GOOD, "opponent_rank": 1})
    tough, _, _ = reg.predict(FALLBACK_VERSION, {**GOOD, "opponent_rank": 32})
    assert soft > tough


def test_to_matrix_stacks_rows():
    assert to_matrix([GOOD, GOOD]).shape == (2, len(FEATURE_ORDER))
