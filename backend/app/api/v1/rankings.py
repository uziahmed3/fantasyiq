from fastapi import APIRouter, Query

from app.api.deps import DbSession
from app.core.cache import cache_get, cache_set, rankings_key
from app.core.config import settings
from app.repositories.players import LeaderRepository, PredictionRepository, StatsRepository
from app.schemas.player import (
    RankingRow,
    RankingsOut,
    SeasonLeaderRow,
    SeasonLeadersOut,
    SeasonRankingRow,
    SeasonRankingsOut,
)
from app.services.draft_value import (
    replacement_levels,
    replacement_note,
    value_over_replacement,
)

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


def _position_rank(pools: dict[str, list[float]], position: str, projection: float) -> int:
    """Where this projection sits within its own position pool."""
    values = sorted(pools.get(position, []), reverse=True)
    for i, value in enumerate(values, start=1):
        if value <= projection:
            return i
    return len(values) + 1


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
    position: str = Query("FLEX", min_length=2, max_length=4, description="WR/RB/TE/FLEX"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort: str = Query(
        "value",
        pattern="^(value|points)$",
        description="value = points above a replacement starter at the same position "
        "(the right order for drafting); points = raw projected points per game",
    ),
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

    offset = (page - 1) * per_page
    key = rankings_key(f"season:{position}:{season}:{page}:{sort}", 0, per_page)
    if (cached := cache_get(key)) is not None:
        return SeasonRankingsOut(**cached)

    games = settings.games_per_season
    # Replacement level is a property of the whole position pool, so it is computed from
    # every projection - not from the page being returned.
    pools = repo.position_projections(season, model_version)
    levels = replacement_levels(pools)

    total = repo.season_board_count(season, model_version, position)

    if sort == "value":
        # Ranking by value needs the full set ordered before paging, because the order is
        # not the stored order. The pool is a few hundred rows, so this is cheap.
        everything = repo.season_board(season, model_version, position, limit=2000, offset=0)
        everything.sort(
            key=lambda r: value_over_replacement(r[0].prediction, r[1].position, levels) or -999,
            reverse=True,
        )
        rows = everything[offset : offset + per_page]
    else:
        rows = repo.season_board(season, model_version, position, per_page, offset)

    out = SeasonRankingsOut(
        season=season,
        position=position.upper(),
        model_version=model_version,
        games_assumed=games,
        season_started=stats.season_has_games(season),
        replacement_level=levels,
        replacement_note=replacement_note(levels),
        page=page,
        per_page=per_page,
        total=total,
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
                value_over_replacement=value_over_replacement(
                    pred.prediction, player.position, levels
                ),
                position_rank=_position_rank(pools, player.position, pred.prediction),
            )
            for i, (pred, player, ctx) in enumerate(rows, start=offset + 1)
        ],
    )
    cache_set(key, out.model_dump(), ttl=settings.rankings_cache_ttl)
    return out


@router.get(
    "/last-season",
    response_model=SeasonLeadersOut,
    summary="What players actually did in a completed season",
)
def last_season_leaders(
    db: DbSession,
    season: int | None = Query(None, description="Defaults to the latest completed season"),
    position: str = Query("FLEX", min_length=2, max_length=4, description="WR/RB/TE/FLEX"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
) -> SeasonLeadersOut:
    """Actual results, not projections - the other half of a draft decision.

    Regular season only. Playoff weeks are in the data but fantasy leagues do not score
    them, and including them rewarded players whose teams went deep rather than players
    who were good.
    """
    repo = LeaderRepository(db)
    if season is None:
        season = repo.latest_completed_season() or 0

    offset = (page - 1) * per_page
    total = repo.count(season, position)
    rows = repo.leaders(season, position, per_page, offset)

    return SeasonLeadersOut(
        season=season,
        position=position.upper(),
        scoring="PPR (1 point per reception)",
        regular_season_only=True,
        page=page,
        per_page=per_page,
        total=total,
        leaders=[
            SeasonLeaderRow(
                rank=i,
                player_id=r.player_id,
                name=r.name,
                team=r.team,
                position=r.position,
                games=int(r.games or 0),
                total_points=round(float(r.total_points or 0), 1),
                points_per_game=round(float(r.points_per_game or 0), 2),
                targets=int(r.targets or 0),
                receptions=int(r.receptions or 0),
                yards=round(float(r.yards or 0), 1),
                touchdowns=int(r.touchdowns or 0),
            )
            for i, r in enumerate(rows, start=offset + 1)
        ],
    )
