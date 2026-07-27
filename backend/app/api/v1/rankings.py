from fastapi import APIRouter, Query

from app.api.deps import DbSession
from app.core.cache import cache_get, cache_set, rankings_key
from app.core.config import settings
from app.repositories.players import PredictionRepository
from app.schemas.player import RankingRow, RankingsOut

router = APIRouter(prefix="/rankings", tags=["rankings"])


@router.get("", response_model=RankingsOut, summary="Weekly projected leaderboard")
def rankings(
    db: DbSession,
    week: int = Query(..., ge=1, le=22),
    season: int = Query(2023, ge=1999),
    position: str = Query("WR", min_length=2, max_length=4),
    limit: int = Query(25, ge=1, le=200),
) -> RankingsOut:
    """Reads the predictions table rather than invoking the model.

    The weekly pipeline batch-writes projections for every player, so this endpoint is a
    pure indexed read - which is why it can serve the dashboard's front page.
    """
    model_version = settings.active_model_version
    key = rankings_key(f"{position}:{season}", week, limit)
    if (cached := cache_get(key)) is not None:
        return RankingsOut(**cached)

    rows = PredictionRepository(db).latest_for_week(
        season=season, week=week, model_version=model_version, position=position, limit=limit
    )
    out = RankingsOut(
        season=season,
        week=week,
        position=position.upper(),
        model_version=model_version,
        rankings=[
            RankingRow(
                rank=i,
                player_id=player.id,
                name=player.name,
                team=player.team,
                position=player.position,
                projected_points=round(pred.prediction, 2),
                confidence=pred.confidence,
            )
            for i, (pred, player) in enumerate(rows, start=1)
        ],
    )
    cache_set(key, out.model_dump(), ttl=settings.rankings_cache_ttl)
    return out
