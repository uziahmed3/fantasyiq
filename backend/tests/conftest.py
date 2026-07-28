"""Tests run against SQLite + fakeredis + a stubbed ML service.

That combination means the whole API surface is testable in CI with zero
infrastructure, while integration tests (tests/test_integration.py) exercise the
real Postgres/Redis containers when they are available.
"""

import os

os.environ.setdefault("DATABASE_URL_OVERRIDE", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

import fakeredis  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core import cache  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Player, PlayerContext, PlayerStats, Prediction  # noqa: E402
from app.schemas.prediction import FeatureVector, PreseasonFeatureVector  # noqa: E402
from app.services import ml_client as ml_client_module  # noqa: E402

TEST_URL = "sqlite+pysqlite:///:memory:"


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine(
        TEST_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture(autouse=True)
def fake_cache():
    cache.set_redis(fakeredis.FakeRedis(decode_responses=True))
    yield
    cache.set_redis(None)


class StubMLClient:
    """Deterministic stand-in for the ML service."""

    def __init__(self) -> None:
        self.calls: list[FeatureVector] = []
        self.preseason_calls: list[PreseasonFeatureVector] = []
        self.fail_with: Exception | None = None

    def predict(self, features: FeatureVector, model_version: str | None = None) -> dict:
        if self.fail_with:
            raise self.fail_with
        self.calls.append(features)
        return {
            "prediction": round(features.fantasy_points_last_3 * 1.05 + 2.0, 3),
            "confidence": 0.81,
            "model_version": model_version or "xgboost_v1",
        }

    def predict_preseason(
        self, features: PreseasonFeatureVector, model_version: str | None = None
    ) -> dict:
        if self.fail_with:
            raise self.fail_with
        self.preseason_calls.append(features)
        prior = features.prior_points_per_game or 0.0
        rookie = bool(features.is_rookie)
        pick = features.draft_pick or 300
        value = 13.0 * (1.0 - min(pick, 300) / 300.0) if rookie else prior * 0.95
        return {
            "prediction": round(value, 3),
            "confidence": 0.35 if rookie else 0.8,
            "model_version": model_version or "preseason_v1",
            "framework": "stub",
            "basis": "rookie - draft capital" if rookie else "prior season",
        }

    def health(self) -> bool:
        return True


@pytest.fixture
def stub_ml(monkeypatch):
    stub = StubMLClient()
    monkeypatch.setattr(ml_client_module, "ml_client", stub)
    # PredictionService resolves the default client at call time via this module attr.
    import app.services.predictions as pred_module

    monkeypatch.setattr(pred_module, "ml_client", stub)
    return stub


@pytest.fixture
def client(db_session, stub_ml):
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def seed(db_session):
    """One WR with four completed weeks, plus a scattering of opponents to rank."""
    jj = Player(
        id=15, name="Justin Jefferson", team="MIN", position="WR", age=25, external_id="00-0036322"
    )
    chase = Player(
        id=16, name="Ja'Marr Chase", team="CIN", position="WR", age=24, external_id="00-0036900"
    )
    db_session.add_all([jj, chase])
    db_session.flush()

    weeks = [
        (1, "TB", 9, 7, 84.0, 1, 20.4),
        (2, "PHI", 12, 8, 110.0, 0, 17.0),
        (3, "LAC", 14, 9, 149.0, 1, 29.9),
        (4, "KC", 8, 5, 62.0, 0, 11.2),
    ]
    for wk, opp, tgt, rec, yds, td, fp in weeks:
        db_session.add(
            PlayerStats(
                player_id=jj.id,
                season=2023,
                week=wk,
                opponent=opp,
                is_home=wk % 2 == 0,
                targets=tgt,
                receptions=rec,
                yards=yds,
                touchdowns=td,
                fantasy_points=fp,
            )
        )
        db_session.add(
            PlayerStats(
                player_id=chase.id,
                season=2023,
                week=wk,
                opponent=opp,
                is_home=wk % 2 == 1,
                targets=tgt - 1,
                receptions=rec - 1,
                yards=yds - 10,
                touchdowns=td,
                fantasy_points=fp - 2,
            )
        )
    db_session.add(
        Prediction(
            player_id=chase.id,
            season=2023,
            week=5,
            opponent="ARI",
            prediction=22.1,
            confidence=0.79,
            model_version="xgboost_v1",
        )
    )
    db_session.add(
        Prediction(
            player_id=jj.id,
            season=2023,
            week=5,
            opponent="GB",
            prediction=19.8,
            confidence=0.83,
            model_version="xgboost_v1",
        )
    )
    db_session.commit()
    return {"jj": jj, "chase": chase}


@pytest.fixture
def rookie(db_session):
    """A first-round rookie: on a roster, zero games, context but no production.

    This is the case the in-season model cannot answer at all.
    """
    player = Player(
        id=99, name="Rook Firstround", team="LV", position="WR", age=22, external_id="00-0099999"
    )
    db_session.add(player)
    db_session.flush()
    db_session.add(
        PlayerContext(
            player_id=player.id,
            season=2023,
            team="LV",
            prior_games=0,
            prior_points_per_game=0.0,
            prior_targets_per_game=0.0,
            prior_yards_per_game=0.0,
            prior_target_share=0.0,
            prior_last4_points_per_game=0.0,
            prior_snap_share=None,
            depth_chart_rank=2,
            draft_round=1,
            draft_pick=6,
            rookie_season=2023,
            years_experience=0,
            is_rookie=True,
            age=22.0,
            team_pass_attempts_prior=3800.0,
            team_points_prior=210.0,
            qb_changed=False,
        )
    )
    db_session.commit()
    return player


@pytest.fixture
def veteran_with_context(db_session, seed):
    """Justin Jefferson also gets a context row, so week-1 routing can be tested."""
    db_session.add(
        PlayerContext(
            player_id=15,
            season=2024,
            team="MIN",
            prior_games=4,
            prior_points_per_game=19.6,
            prior_targets_per_game=10.75,
            prior_yards_per_game=101.25,
            prior_target_share=0.28,
            prior_last4_points_per_game=19.6,
            prior_snap_share=0.88,
            depth_chart_rank=1,
            draft_round=1,
            draft_pick=22,
            rookie_season=2020,
            years_experience=4,
            is_rookie=False,
            age=25.0,
            team_pass_attempts_prior=4100.0,
            team_points_prior=240.0,
            qb_changed=True,
        )
    )
    db_session.commit()
    return 15
