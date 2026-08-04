from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    """Backoff/throttle waits must not slow down the suite. Individual tests can
    still override asyncio.sleep again if they need to assert on wait duration."""

    async def _no_sleep(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
