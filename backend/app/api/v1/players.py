from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import DbSession
from app.repositories.players import PlayerRepository, PredictionRepository, StatsRepository
from app.schemas.player import (
    PaginatedPlayers,
    PlayerDetailOut,
    PlayerStatsOut,
)
from app.schemas.prediction import PredictionRecord

router = APIRouter(prefix="/players", tags=["players"])


@router.get("", response_model=PaginatedPlayers, summary="List / search players")
def list_players(
    db: DbSession,
    position: str | None = Query(None, examples=["WR"]),
    team: str | None = Query(None, examples=["MIN"]),
    name: str | None = Query(None, description="Case-insensitive substring match"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PaginatedPlayers:
    items, total = PlayerRepository(db).search(position, team, name, limit, offset)
    return PaginatedPlayers(total=total, limit=limit, offset=offset, items=items)


@router.get("/{player_id}", response_model=PlayerDetailOut)
def get_player(player_id: int, db: DbSession) -> PlayerDetailOut:
    player = PlayerRepository(db).get(player_id)
    if player is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Player not found")
    return PlayerDetailOut.model_validate(player)


@router.get("/{player_id}/stats", response_model=list[PlayerStatsOut])
def get_player_stats(
    player_id: int,
    db: DbSession,
    season: int | None = Query(None, ge=1999),
    limit: int = Query(20, ge=1, le=100),
) -> list[PlayerStatsOut]:
    if PlayerRepository(db).get(player_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Player not found")
    rows = StatsRepository(db).for_player(player_id, season, limit)
    return [PlayerStatsOut.model_validate(r) for r in rows]


@router.get("/{player_id}/predictions", response_model=list[PredictionRecord])
def get_player_predictions(
    player_id: int, db: DbSession, limit: int = Query(20, ge=1, le=100)
) -> list[PredictionRecord]:
    if PlayerRepository(db).get(player_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Player not found")
    return [
        PredictionRecord.model_validate(p)
        for p in PredictionRepository(db).history(player_id, limit)
    ]
