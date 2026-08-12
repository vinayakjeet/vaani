from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app as fastapi_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Pin the provider rather than inheriting it.

    The demo route chooses its provider per request from `Settings`, which reads
    the developer's `.env`. Left ambient, the test that asserts a mock reply talks
    to whatever this machine is configured for, so it passes in CI where no `.env`
    exists and returns 500 on a machine with `LLM_PROVIDER=groq` set. A test whose
    input comes from the environment is a test whose result is about the
    environment.
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    return TestClient(fastapi_app)
