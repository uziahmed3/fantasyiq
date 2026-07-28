from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import DbSession
from app.core.cache import cache_get, cache_set, rankings_key
from app.core.config import settings
from app.repositories.players import (
    LeaderRepository,
    PlayerRepository,
    PredictionRepository,
    StatsRepository,
)
from app.schemas.player import (
    ProjectionDriver,
    ProjectionWhy,
    RankingRow,
    RankingsOut,
    SeasonLeaderRow,
    SeasonLeadersOut,
    SeasonRankingRow,
    SeasonRankingsOut,
)
from app.schemas.prediction import PredictionRequest
from app.services.draft_value import (
    replacement_levels,
    replacement_note,
    value_over_replacement,
)
from app.services.ml_client import MLServiceError
from app.services.predictions import PredictionService

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
    """Plain-language note on what a season projection actually rests on.

    This used to read "16 games last season", which was true of nearly every player on
    the board and so told a drafter nothing about which of two to take. Worse, it implied
    the projection was a view of last season only - the exact complaint that prompted the
    career features in the first place.

    What it says now is the shape of the evidence: how long the record is, what the
    career rate is, and whether last season agreed with it. A down year and a breakout
    year look completely different here, and neither is hidden behind a game count.

    Derived from player_context alone, so the board stays a single indexed read. The
    per-feature attribution is a click away on /rankings/season/{player_id}/why; it costs
    a model call and nobody should pay for it just to load a page.
    """
    if ctx is None:
        return None
    if ctx.is_rookie:
        if ctx.draft_pick:
            return f"rookie, no NFL record - drafted #{ctx.draft_pick}"
        return "rookie, no NFL record - undrafted"

    seasons = ctx.career_seasons or 0
    career = ctx.career_weighted_ppg or 0.0
    last = ctx.prior_points_per_game or 0.0
    games = ctx.prior_games or 0

    if seasons <= 1:
        head = f"1 season, {games} games"
    else:
        head = f"{seasons} seasons, {career:.1f} career pts/gm"

    # Did last season agree with the career record? That disagreement is the single most
    # decision-relevant fact on the board: it is the difference between buying a decline
    # and buying a discount.
    if career > 0 and games >= 4:
        delta = last - career
        if delta <= -3.0:
            head += f"; last year {last:.1f} was well below it"
        elif delta >= 3.0:
            head += f"; last year {last:.1f} was a step up"
        else:
            head += f"; last year {last:.1f}, in line"
    if 0 < games < 10:
        head += f" (only {games} games)"
    return head


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
    "/season/{player_id}/why",
    response_model=ProjectionWhy,
    summary="Why is this player projected where he is?",
)
def season_projection_why(
    db: DbSession,
    player_id: int,
    season: int | None = Query(None, description="Defaults to the season being projected"),
    top: int = Query(6, ge=1, le=20),
) -> ProjectionWhy:
    """Decompose one player's season projection into the features that produced it.

    Kept off the board itself deliberately. Attribution walks every tree for every
    feature, so folding it into the leaderboard would make every page load pay for
    explanations of twenty players when the user wants one. This is the drill-down.

    Cached on (player, season, model version): the inputs are a fixed context row and a
    fixed artifact, so the answer cannot change until one of them does.
    """
    repo = PredictionRepository(db)
    if season is None:
        season = repo.next_unplayed_season()

    player = PlayerRepository(db).get(player_id)
    if player is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no player with id {player_id}")

    model_version = settings.active_preseason_model_version
    key = rankings_key(f"why:{player_id}:{season}:{model_version}", 0, top)
    if (cached := cache_get(key)) is not None:
        return ProjectionWhy(**cached)

    service = PredictionService(db)
    features = service.build_preseason_features(
        # week is required by the request contract and validated >= 1; the preseason
        # feature builder reads only player_id and season, so any legal week works.
        PredictionRequest(player_id=player_id, season=season, week=1)
    )
    if features is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no preseason context for player {player_id} in {season} - run the context "
            "pipeline for that season first",
        )

    try:
        body = service.ml.explain_preseason(features, model_version, top=top)
    except MLServiceError as exc:
        # The projection itself is fine and already on screen; only the explanation is
        # missing. 503 says "ask again later", which is the truth, rather than 500.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    out = ProjectionWhy(
        player_id=player_id,
        name=player.name,
        position=player.position,
        season=season,
        projected_points_per_game=round(body["prediction"], 2),
        baseline=round(body["baseline"], 2),
        model_version=body["model_version"],
        headline=body["headline"],
        drivers=[
            ProjectionDriver(
                feature=d["feature"],
                label=d["label"],
                display_value=d["display_value"],
                contribution=d["contribution"],
                explanation=d["explanation"],
            )
            for d in body["drivers"]
        ],
        drivers_shown=body["drivers_shown"],
        total_features=body["total_features"],
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
