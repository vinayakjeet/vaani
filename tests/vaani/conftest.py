from __future__ import annotations

import pytest


@pytest.fixture
def exported(monkeypatch):
    """Spans from a genuinely recording tracer, not the no-op one.

    Without this any test about span contents proves nothing. No OTLP endpoint is
    configured in the suite, so `spanlight.init` was never called and `get_tracer`
    hands back a tracer whose spans are `NonRecordingSpan`. Nothing is written to
    those, so `record_exception=True` and `record_exception=False` behave
    identically and a leak test passes against the leak.

    A local provider rather than the global one, because installing a global
    provider mid-suite is how the previous project left a metric exporter posting to
    a dead host for the rest of its run.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import vaani.spans as spans_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(
        spans_module.spanlight, "get_tracer", lambda: provider.get_tracer("test")
    )
    return exporter
