from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import DbSession
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.ml_client import MLServiceError
from app.services.predictions import PlayerNotFound, PredictionService

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse, summary="Project fantasy points")
def predict(
    req: PredictionRequest,
    db: DbSession,
    refresh: bool = Query(False, description="Bypass the cache and force a fresh inference"),
) -> PredictionResponse:
    try:
        return PredictionService(db).predict(req, use_cache=not refresh)
    except PlayerNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except MLServiceError as exc:
        # 503, not 500: the API is healthy, a dependency is not.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.get("/players/{player_id}/prediction", response_model=PredictionResponse)
def get_prediction(
    player_id: int,
    db: DbSession,
    week: int = Query(..., ge=1, le=22),
    season: int = Query(2023, ge=1999),
    opponent: str = Query("UNK", min_length=2, max_length=8),
    is_home: bool = True,
) -> PredictionResponse:
    """GET convenience wrapper - cacheable by CDN/browser, unlike the POST."""
    req = PredictionRequest(
        player_id=player_id, week=week, season=season, opponent=opponent, is_home=is_home
    )
    return predict(req, db)


@router.post("/compare", response_model=list[PredictionResponse], summary="Compare players")
def compare(
    reqs: list[PredictionRequest],
    db: DbSession,
) -> list[PredictionResponse]:
    if not 2 <= len(reqs) <= 8:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Compare 2-8 players at a time")
    service = PredictionService(db)
    out: list[PredictionResponse] = []
    for r in reqs:
        try:
            out.append(service.predict(r))
        except PlayerNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except MLServiceError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return sorted(out, key=lambda p: p.prediction, reverse=True)
