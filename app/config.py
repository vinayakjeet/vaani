from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "local"
    log_level: str = "INFO"
    git_sha: str = "dev"

    database_url: str | None = None
    redis_url: str | None = None

    llm_provider: str = "mock"
    llm_max_retry_attempts: int = 5

    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_headers: str | None = None


def get_settings() -> Settings:
    """Not cached on purpose: reading env vars is cheap and tests need to be able
    to monkeypatch env per-test without fighting a cached singleton."""
    return Settings()
