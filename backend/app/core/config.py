"""Central, typed configuration. Nothing in the codebase reads os.environ directly."""

from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # App
    environment: str = "local"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    project_name: str = "FantasyIQ API"

    # Postgres
    postgres_user: str = "fantasyiq"
    postgres_password: str = "change_me_locally"
    postgres_db: str = "fantasyiq"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    # Set directly in tests (e.g. sqlite+pysqlite:///:memory:) to bypass Postgres.
    database_url_override: str | None = None
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    prediction_cache_ttl: int = 3600
    rankings_cache_ttl: int = 300

    # Auth
    jwt_secret_key: str = "dev-only-secret-do-not-use-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ML service
    ml_service_url: str = "http://localhost:9000"
    ml_service_timeout_seconds: float = 5.0
    active_model_version: str = "xgboost_v1"

    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    is_production: bool = Field(default=False, exclude=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    object.__setattr__(s, "is_production", s.environment.lower() in {"prod", "production"})
    return s


settings = get_settings()
