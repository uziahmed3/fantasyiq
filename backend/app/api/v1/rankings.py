from fastapi import APIRouter, Query

from app.api.deps import DbSession
from app.core.cache import cache_get, cache_set, rankings_key
from app.core.config import settings
from app.repositories.players import PredictionRepository, StatsRepository
from app.schemas.player import RankingRow, RankingsOut, SeasonRankingRow, SeasonRankingsOut

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


def _basis(ctx) -> str | None:
    """Plain-language note on what a season projection actually rests on."""
    if ctx is None:
        return None
    if ctx.is_rookie:
        if ctx.draft_pick:
            return f"rookie - drafted #{ctx.draft_pick}"
        return "rookie - undrafted or draft position unknown"
    games = ctx.prior_games or 0
    if games >= 12:
        return f"{games} games last season"
    if games > 0:
        return f"only {games} games last season"
    return "no games last season"


@router.get(
    "/season",
    response_model=SeasonRankingsOut,
    summary="Draft board - projections for a whole upcoming season",
)
def season_rankings(
    db: DbSession,
    season: int | None = Query(None, description="Defaults to the season being projected"),
    position: str = Query("WR", min_length=2, max_length=4),
    limit: int = Query(100, ge=1, le=400),
) -> SeasonRankingsOut:
    """Season-long projections, produced before the season starts.

    Separate from the weekly endpoint because it answers a different question - "who do I
    draft" rather than "who do I start" - and is served from the preseason model, whose
    output is a per-game rate rather than a single week's points.

    Reports `season_started` so a client can switch itself over to weekly projections
    once real games exist, instead of hardcoding a date.
    """
    model_version = settings.active_preseason_model_version
    repo = PredictionRepository(db)
    stats = StatsRepository(db)

    if season is None:
        season = repo.next_unplayed_season()

    key = rankings_key(f"season:{position}:{season}", 0, limit)
    if (cached := cache_get(key)) is not None:
        return SeasonRankingsOut(**cached)

    rows = repo.season_board(season, model_version, position, limit)
    games = settings.games_per_season

    out = SeasonRankingsOut(
        season=season,
        position=position.upper(),
        model_version=model_version,
        games_assumed=games,
        season_started=stats.season_has_games(season),
        rankings=[
            SeasonRankingRow(
                rank=i,
                player_id=player.id,
                name=player.name,
                team=player.team,
                position=player.position,
                projected_points_per_game=round(pred.prediction, 2),
                projected_season_points=round(pred.prediction * games, 1),
                confidence=pred.confidence,
                is_rookie=bool(ctx.is_rookie) if ctx else False,
                basis=_basis(ctx),
            )
            for i, (pred, player, ctx) in enumerate(rows, start=1)
        ],
    )
    cache_set(key, out.model_dump(), ttl=settings.rankings_cache_ttl)
    return out
