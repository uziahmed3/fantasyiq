import pytest

from app.repositories.players import PredictionRepository
from app.schemas.prediction import PredictionRequest
from app.services.ml_client import MLServiceError
from app.services.predictions import PlayerNotFound, PredictionService


def test_features_use_only_prior_weeks(db_session, seed, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    # Predicting week 3 must not see weeks 3 or 4.
    fv = svc.build_features(PredictionRequest(player_id=15, week=3, season=2023, opponent="GB"))
    assert fv.fantasy_points_last_3 == pytest.approx((20.4 + 17.0) / 2, abs=1e-3)
    assert fv.fantasy_points_last_1 == 17.0
    assert fv.games_played == 2


def test_features_full_window(db_session, seed, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    fv = svc.build_features(PredictionRequest(player_id=15, week=5, season=2023, opponent="GB"))
    assert fv.fantasy_points_last_3 == pytest.approx((17.0 + 29.9 + 11.2) / 3, abs=1e-3)
    assert fv.targets_last_3 == pytest.approx((12 + 14 + 8) / 3, abs=1e-3)
    assert fv.games_played == 4


def test_features_cold_start_player(db_session, seed, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    fv = svc.build_features(PredictionRequest(player_id=15, week=1, season=2023, opponent="GB"))
    assert fv.fantasy_points_last_3 == 0.0
    assert fv.games_played == 0


def test_prediction_is_persisted(db_session, seed, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    resp = svc.predict(PredictionRequest(player_id=15, week=5, season=2023, opponent="GB"))
    assert resp.source == "model"
    history = PredictionRepository(db_session).history(15)
    assert any(p.week == 5 and p.model_version == resp.model_version for p in history)


def test_unknown_player_raises(db_session, seed, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    with pytest.raises(PlayerNotFound):
        svc.predict(PredictionRequest(player_id=4242, week=5, season=2023, opponent="GB"))


def test_ml_failure_surfaces_as_503(client, seed, stub_ml):
    stub_ml.fail_with = MLServiceError("ML service timed out")
    r = client.post(
        "/api/v1/predict", json={"player_id": 15, "week": 6, "season": 2023, "opponent": "GB"}
    )
    assert r.status_code == 503


def test_predict_endpoint_roundtrip(client, seed):
    r = client.post(
        "/api/v1/predict", json={"player_id": 15, "week": 6, "season": 2023, "opponent": "GB"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["player"] == "Justin Jefferson"
    assert body["prediction"] > 0
    assert body["opponent"] == "GB"


def test_compare_returns_sorted_desc(client, seed):
    r = client.post(
        "/api/v1/compare",
        json=[
            {"player_id": 15, "week": 6, "season": 2023, "opponent": "GB"},
            {"player_id": 16, "week": 6, "season": 2023, "opponent": "SEA"},
        ],
    )
    assert r.status_code == 200
    preds = [row["prediction"] for row in r.json()]
    assert preds == sorted(preds, reverse=True)


def test_compare_rejects_single_player(client, seed):
    r = client.post(
        "/api/v1/compare", json=[{"player_id": 15, "week": 6, "season": 2023, "opponent": "GB"}]
    )
    assert r.status_code == 422
