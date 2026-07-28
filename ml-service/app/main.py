import os

from fastapi import FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator

from app.features import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    PRESEASON_FEATURE_ORDER,
    PRESEASON_SCHEMA_VERSION,
    FeatureContractError,
    preseason_matrix,
)
from app.registry import (
    FALLBACK_VERSION,
    PRESEASON_FALLBACK_VERSION,
    ModelNotFound,
    registry,
)
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    PreseasonBatchRequest,
    PreseasonBatchResponse,
    PreseasonRequest,
    PreseasonResponse,
)

DEFAULT_VERSION = os.getenv("ACTIVE_MODEL_VERSION", "xgboost_v1")
DEFAULT_PRESEASON_VERSION = os.getenv("ACTIVE_PRESEASON_MODEL_VERSION", "preseason_v1")

app = FastAPI(
    title="FantasyIQ ML Service",
    version="1.0.0",
    description=(
        "Stateless inference service. Owns model artifacts and versioning; owns no data. "
        "The backend assembles features from Postgres and calls /predict."
    ),
)
Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


def _resolve(requested: str | None) -> str:
    version = requested or DEFAULT_VERSION
    available = registry.available()
    if version not in available:
        # Degrade rather than 500: an unavailable artifact should not take the API down.
        return FALLBACK_VERSION
    return version


def _resolve_preseason(requested: str | None) -> str:
    version = requested or DEFAULT_PRESEASON_VERSION
    if version not in registry.available():
        return PRESEASON_FALLBACK_VERSION
    return version


def _basis(features: dict) -> str:
    """Plain-language statement of what a preseason number actually rests on.

    Returned to the caller because "18.2 points" from a full prior season and
    "10.6 points" from a draft slot deserve to be read differently, and the UI should
    not have to reverse-engineer which it is holding.
    """
    if float(features.get("is_rookie") or 0) > 0.5:
        pick = features.get("draft_pick")
        if pick:
            return f"rookie - draft capital (pick {int(pick)}) and projected role"
        return "rookie - projected role only, undrafted or draft position unknown"
    games = float(features.get("prior_games") or 0)
    if games >= 12:
        return f"prior season ({int(games)} games) plus role"
    if games > 0:
        return f"partial prior season ({int(games)} games), shrunk toward baseline"
    return "no prior production - role and experience only"


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/models", tags=["ops"], summary="Available model versions")
def models() -> dict:
    return {
        "default": DEFAULT_VERSION,
        "default_preseason": DEFAULT_PRESEASON_VERSION,
        "available": registry.available(),
        "in_season": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_order": list(FEATURE_ORDER),
        },
        "preseason": {
            "feature_schema_version": PRESEASON_SCHEMA_VERSION,
            "feature_order": list(PRESEASON_FEATURE_ORDER),
        },
    }


@app.post("/predict/preseason", response_model=PreseasonResponse, tags=["inference"])
def predict_preseason(req: PreseasonRequest) -> PreseasonResponse:
    """Project a player who has not played yet this season.

    Separate endpoint rather than a flag on /predict because the input contract is
    genuinely different - different features, different target (points per game rather
    than a single week), and nulls are expected rather than an error.
    """
    version = _resolve_preseason(req.model_version)
    features = req.features.model_dump()
    try:
        value, confidence, resolved = registry.predict_preseason(version, features)
    except FeatureContractError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except ModelNotFound as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return PreseasonResponse(
        prediction=round(value, 3),
        confidence=confidence,
        model_version=resolved,
        framework=registry.get(resolved).framework,
        basis=_basis(features),
    )


@app.post("/predict/preseason/batch", response_model=PreseasonBatchResponse, tags=["inference"])
def predict_preseason_batch(req: PreseasonBatchRequest) -> PreseasonBatchResponse:
    version = _resolve_preseason(req.model_version)
    try:
        model = registry.get(version)
        matrix = preseason_matrix([item.model_dump() for item in req.items])
        values = model.predict_fn(matrix)
    except FeatureContractError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except ModelNotFound as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return PreseasonBatchResponse(
        model_version=model.version, predictions=[round(float(v), 3) for v in values]
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    version = _resolve(req.model_version)
    try:
        value, confidence, resolved = registry.predict(version, req.features.model_dump())
    except FeatureContractError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except ModelNotFound as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return PredictResponse(
        prediction=round(value, 3),
        confidence=confidence,
        model_version=resolved,
        framework=registry.get(resolved).framework,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Used by the weekly pipeline to project every player in one call - one HTTP
    round trip and one vectorised forward pass instead of N of each."""
    version = _resolve(req.model_version)
    try:
        model = registry.get(version)
        from app.features import to_matrix

        matrix = to_matrix([item.model_dump() for item in req.items])
        values = model.predict_fn(matrix)
    except FeatureContractError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except ModelNotFound as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return BatchPredictResponse(
        model_version=model.version, predictions=[round(float(v), 3) for v in values]
    )
