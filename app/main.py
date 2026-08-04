from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.logging_config import configure_logging
from app.middleware import RequestContextMiddleware
from app.otel_bootstrap import setup_otel
from app.routers import demo, health


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="ai-portfolio-template", version="0.1.0")
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    app.include_router(demo.router)

    setup_otel(app, settings.otel_exporter_otlp_endpoint, settings.otel_exporter_otlp_headers)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
