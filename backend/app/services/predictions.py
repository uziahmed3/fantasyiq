"""Prediction orchestration: route -> features -> cache -> model -> persist.

Two models sit behind one endpoint, chosen by what evidence exists:

    games played this season == 0  ->  preseason model
        prior-season production, target share, snap share, depth-chart rank,
        draft capital, team QB situation. The only path that can project a
        rookie, or anyone in week 1.

    games played this season >= 1  ->  in-season model
        rolling 3-game form. More accurate once it has data, and useless before.

The router is deliberately a data-availability check rather than a week-number check:
"week 1" and "has not played" are usually the same thing but not always - a player
returning from injury in week 8 has no rolling form either, and gets the preseason
treatment for the same reason.

Request flow:
    /predict
       |-- count games played this season   (routing decision)
       |-- build the matching feature vector from the database
       |-- cache GET  ---- hit --> return (source="cache")
       |-- miss --> POST ml-service /predict or /predict/preseason
       |-- write predictions row (audit log)
       |-- cache SETEX (TTL)
"""

from sqlalchemy.orm import Session

from app.core.cache import cache_get, cache_set, prediction_key
from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import PREDICTIONS
from app.repositories.players import (
    ContextRepository,
    PlayerRepository,
    PredictionRepository,
    StatsRepository,
)
from app.schemas.prediction import (
    FeatureVector,
    PredictionRequest,
    PredictionResponse,
    PreseasonFeatureVector,
)
from app.services.ml_client import MLClient, ml_client


class PlayerNotFound(LookupError):
    pass


class PredictionService:
    def __init__(self, db: Session, client: MLClient | None = None) -> None:
        self.db = db
        self.players = PlayerRepository(db)
        self.stats = StatsRepository(db)
        self.predictions = PredictionRepository(db)
        self.context = ContextRepository(db)
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

    def build_preseason_features(self, req: PredictionRequest) -> PreseasonFeatureVector | None:
        """Assemble the preseason vector from player_context. None if we have no context."""
        ctx = self.context.get(req.player_id, req.season)
        if ctx is None:
            # Context for this season has not been built; an older one is still useful.
            ctx = self.context.latest_at_or_before(req.player_id, req.season)
        if ctx is None:
            return None
        return PreseasonFeatureVector(
            prior_points_per_game=ctx.prior_points_per_game,
            prior_last4_points_per_game=ctx.prior_last4_points_per_game,
            prior_targets_per_game=ctx.prior_targets_per_game,
            prior_target_share=ctx.prior_target_share,
            prior_carries_per_game=ctx.prior_carries_per_game,
            prior_carry_share=ctx.prior_carry_share,
            prior_yards_per_game=ctx.prior_yards_per_game,
            prior_games=ctx.prior_games,
            prior_snap_share=ctx.prior_snap_share,
            depth_chart_rank=ctx.depth_chart_rank,
            draft_round=ctx.draft_round,
            draft_pick=ctx.draft_pick,
            years_experience=ctx.years_experience,
            is_rookie=int(bool(ctx.is_rookie)),
            age=ctx.age,
            team_pass_attempts_prior=ctx.team_pass_attempts_prior,
            qb_changed=int(bool(ctx.qb_changed)),
        )

    def choose_mode(self, req: PredictionRequest) -> str:
        """'preseason' when the player has no games this season, else 'in_season'."""
        played = self.stats.games_before_week(req.player_id, req.season, req.week)
        return "in_season" if played >= 1 else "preseason"

    def predict(self, req: PredictionRequest, use_cache: bool = True) -> PredictionResponse:
        player = self.players.get(req.player_id)
        if player is None:
            raise PlayerNotFound(f"player {req.player_id} not found")

        mode = self.choose_mode(req)
        if mode == "preseason":
            model_version = settings.active_preseason_model_version
        else:
            model_version = settings.active_model_version

        # Mode is part of the cache key: the two models answer the same question with
        # different evidence, and must never serve each other's cached values.
        key = prediction_key(req.player_id, req.week, f"{req.opponent}:{mode}", model_version)

        if use_cache and (cached := cache_get(key)) is not None:
            PREDICTIONS.labels(model_version=model_version, source="cache").inc()
            return PredictionResponse(**cached, source="cache")

        basis: str | None = None
        if mode == "preseason":
            preseason_features = self.build_preseason_features(req)
            if preseason_features is None:
                # No context row at all - fall back to the in-season path, which will
                # produce a weak zero-history projection rather than nothing.
                logger.warning(
                    "no_preseason_context",
                    player_id=req.player_id,
                    season=req.season,
                    hint="run `python -m context --season <season>` in the pipeline",
                )
                mode = "in_season"
                model_version = settings.active_model_version
                result = self.ml.predict(self.build_features(req), model_version)
            else:
                result = self.ml.predict_preseason(preseason_features, model_version)
                basis = result.get("basis")
        else:
            result = self.ml.predict(self.build_features(req), model_version)

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
            mode=mode,  # type: ignore[arg-type]
            basis=basis,
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
            mode=mode,
            prediction=response.prediction,
            model_version=response.model_version,
        )
        return response
