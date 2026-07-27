from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

_connect_args = {}
_kwargs: dict = {"pool_pre_ping": True, "echo": settings.db_echo}
if settings.database_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}
else:
    _kwargs |= {"pool_size": settings.db_pool_size, "max_overflow": settings.db_max_overflow}

engine = create_engine(settings.database_url, connect_args=_connect_args, **_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
