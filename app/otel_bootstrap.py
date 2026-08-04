from __future__ import annotations

import structlog
from fastapi import FastAPI

logger = structlog.get_logger(__name__)


def setup_otel(app: FastAPI, otlp_endpoint: str | None, otlp_headers: str | None = None) -> None:
    """Wire OpenTelemetry tracing to an OTLP endpoint (e.g. Grafana Cloud Tempo).

    No-op if `otlp_endpoint` is unset, so forks that haven't configured Grafana
    Cloud yet import this module, run create_app(), and pay zero cost - no SDK
    initialization, no exporter, no background export thread.
    """
    if not otlp_endpoint:
        logger.info("otel.disabled", reason="OTEL_EXPORTER_OTLP_ENDPOINT not set")
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": app.title}))
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, headers=_parse_headers(otlp_headers))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    logger.info("otel.enabled", endpoint=otlp_endpoint)


def _parse_headers(raw: str | None) -> dict[str, str]:
    """Parse the standard OTEL_EXPORTER_OTLP_HEADERS format: 'key1=value1,key2=value2'."""
    if not raw:
        return {}
    headers = {}
    for pair in raw.split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            headers[key.strip()] = value.strip()
    return headers
