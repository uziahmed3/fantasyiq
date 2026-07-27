"""Prediction orchestration: features -> cache -> model -> persist.

Request flow:
    /predict
       |-- build feature vector from Postgres (rolling windows, prior weeks only)
       |-- Redis GET  ---- hit --> return (source="cache")
       |-- miss --> POST ml-service/predict
       |-- write predictions row (audit log)
       |-- Redis SETEX (TTL)
"""

from sqlalchemy.orm import Session

from app.core.cache import cache_get, cache_set, prediction_key
from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import PREDICTIONS
from app.repositories.players import PlayerRepository, PredictionRepository, StatsRepository
from app.schemas.prediction import FeatureVector, PredictionRequest, PredictionResponse
from app.services.ml_client import MLClient, ml_client


class PlayerNotFound(LookupError):
    pass


class PredictionService:
    def __init__(self, db: Session, client: MLClient | None = None) -> None:
        self.db = db
        self.players = PlayerRepository(db)
        self.stats = StatsRepository(db)
        self.predictions = PredictionRepository(db)
        self.ml = client or ml_client

    def build_features(self, req: PredictionRequest) -> FeatureVector:
        recent = self.stats.last_n_before_week(req.player_id, req.season, req.week, n=3)
        season_avg, games = self.stats.season_aggregate(req.player_id, req.season, req.week)
        opp_rank = self.stats.opponent_defense_rank(req.season, req.opponent, req.week)

        def mean(attr: str) -> float:
            return round(sum(getattr(r, attr) for r in recent) / len(recent), 3) if recent else 0.0

        return FeatureVector(
            targets_last_3=mean("targets"),
            receptions_last_3=mean("receptions"),
            yards_last_3=mean("yards"),
            touchdowns_last_3=mean("touchdowns"),
            fantasy_points_last_3=mean("fantasy_points"),
            fantasy_points_last_1=round(recent[0].fantasy_points, 3) if recent else 0.0,
            season_avg_points=round(season_avg, 3),
            games_played=games,
            opponent_rank=opp_rank,
            is_home=int(req.is_home),
        )

    def predict(self, req: PredictionRequest, use_cache: bool = True) -> PredictionResponse:
        player = self.players.get(req.player_id)
        if player is None:
            raise PlayerNotFound(f"player {req.player_id} not found")

        model_version = settings.active_model_version
        key = prediction_key(req.player_id, req.week, req.opponent, model_version)

        if use_cache and (cached := cache_get(key)) is not None:
            PREDICTIONS.labels(model_version=model_version, source="cache").inc()
            return PredictionResponse(**cached, source="cache")

        features = self.build_features(req)
        result = self.ml.predict(features, model_version)

        response = PredictionResponse(
            player_id=player.id,
            player=player.name,
            season=req.season,
            week=req.week,
            opponent=req.opponent.upper(),
            prediction=round(float(result["prediction"]), 2),
            confidence=result.get("confidence"),
            model_version=result.get("model_version", model_version),
            source="model",
        )

        self.predictions.record(
            player_id=player.id,
            season=req.season,
            week=req.week,
            opponent=req.opponent.upper(),
            prediction=response.prediction,
            confidence=response.confidence,
            model_version=response.model_version,
        )
        cache_set(
            key,
            response.model_dump(exclude={"source"}),
            ttl=settings.prediction_cache_ttl,
        )
        PREDICTIONS.labels(model_version=response.model_version, source="model").inc()
        logger.info(
            "prediction_served",
            player_id=player.id,
            week=req.week,
            prediction=response.prediction,
            model_version=response.model_version,
        )
        return response
