import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, logger
from app.db.session import engine
from app.services.ml_client import ml_client

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", environment=settings.environment, model=settings.active_model_version)
    yield
    logger.info("shutdown")


app = FastAPI(
    title=settings.project_name,
    version="1.0.0",
    description=(
        "Fantasy football projections API. Backed by Postgres, a separately deployed "
        "XGBoost/PyTorch inference service, and a Redis cache-aside layer."
    ),
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_error", error=str(exc), path=request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# /metrics for Prometheus scraping
Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/health", tags=["ops"], summary="Liveness probe")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment}


@app.get("/ready", tags=["ops"], summary="Readiness probe (checks dependencies)")
def ready() -> JSONResponse:
    checks = {"database": False, "ml_service": False}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("readiness_db_failed", error=str(exc))
    checks["ml_service"] = ml_client.health()
    ok = all(checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "checks": checks})
