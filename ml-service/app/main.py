import os

from fastapi import FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator

from app.features import FEATURE_ORDER, FEATURE_SCHEMA_VERSION, FeatureContractError
from app.registry import FALLBACK_VERSION, ModelNotFound, registry
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

DEFAULT_VERSION = os.getenv("ACTIVE_MODEL_VERSION", "xgboost_v1")

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


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/models", tags=["ops"], summary="Available model versions")
def models() -> dict:
    return {
        "default": DEFAULT_VERSION,
        "available": registry.available(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order": list(FEATURE_ORDER),
    }


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
