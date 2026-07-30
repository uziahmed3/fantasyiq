import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
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


# ---------------------------------------------------------------- built-in dashboard
# Served straight off disk with no build step, so `run-local.ps1` produces a real UI on
# a machine that has only Python. The React app in frontend/ is the production front
# end; both talk to this same API.
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():

    class NoCacheStatic(StaticFiles):
        """Serve the dashboard with caching disabled.

        The file is edited in place and has no content hash in its name, so a browser
        will happily keep serving yesterday's copy. That actually happened: a fix to the
        season selectors shipped and the browser kept requesting season=2026 from the old
        bundle. Correctness beats a few saved kilobytes on a local dev UI.
        """

        async def get_response(self, path: str, scope):  # type: ignore[override]
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            return response

    app.mount("/app", NoCacheStatic(directory=STATIC_DIR, html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app/")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)


@app.get("/health", tags=["ops"], summary="Liveness probe")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment}


@app.get("/info", tags=["ops"], summary="What this instance is actually running")
def info() -> dict:
    """Handy when the same code runs on SQLite+memory-cache locally and
    Postgres+Redis when deployed - one call says which.

    Also reports which seasons and weeks actually hold data, so the dashboard can
    default its selectors to something populated instead of to today's calendar year.
    """
    from app.core.cache import backend_name

    seasons: list[int] = []
    latest_week: int | None = None
    try:
        with engine.connect() as conn:
            seasons = [
                int(r[0])
                for r in conn.execute(
                    text("SELECT DISTINCT season FROM player_stats ORDER BY season DESC")
                )
                if r[0] is not None
            ]
            if seasons:
                latest_week = conn.execute(
                    text("SELECT MAX(week) FROM player_stats WHERE season = :s"),
                    {"s": seasons[0]},
                ).scalar()
    except Exception as exc:  # noqa: BLE001 - informational endpoint, never fatal
        logger.warning("info_seasons_failed", error=str(exc))

    return {
        "environment": settings.environment,
        "database": "sqlite" if settings.database_url.startswith("sqlite") else "postgresql",
        "cache_backend": backend_name(),
        "active_model_version": settings.active_model_version,
        "ml_service_url": settings.ml_service_url,
        "seasons_with_data": seasons,
        "latest_week": int(latest_week) if latest_week is not None else None,
    }


@app.get("/ready", tags=["ops"], summary="Readiness probe (checks dependencies)")
def ready() -> JSONResponse:
    checks = {"database": False, "ml_service": False}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("readiness_db_failed", error=str(exc))
    try:
        checks["ml_service"] = ml_client.health()
    except Exception as exc:  # noqa: BLE001 - readiness reports, it does not raise
        logger.warning("readiness_ml_failed", error=str(exc))
    ok = all(checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "checks": checks})
