from __future__ import annotations

import pytest

import vaani.spans as spans


@pytest.fixture(autouse=True)
def strict_span_contract(monkeypatch):
    """Every test runs with the span contract set to raise.

    Production is lenient on purpose: an attribute the table does not declare is
    dropped and logged rather than allowed to take a turn down, because
    instrumentation that throws turns a typo into an outage. That leniency must not
    reach the suite, or a breach becomes a log line nobody reads and the empty-panel
    failure the contract exists to prevent walks straight through CI.
    """
    monkeypatch.setattr(spans, "strict", True)
