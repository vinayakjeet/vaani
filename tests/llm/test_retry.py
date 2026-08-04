from __future__ import annotations

import pytest

from llm.retry import retry_with_backoff
from llm.types import ProviderClientError, ProviderError, RateLimitError


async def test_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=5, min_wait=0, max_wait=0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("boom")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


async def test_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=3, min_wait=0, max_wait=0)
    async def always_fails() -> str:
        calls["n"] += 1
        raise ProviderError("boom")

    with pytest.raises(ProviderError):
        await always_fails()
    assert calls["n"] == 3


async def test_retry_does_not_retry_client_errors():
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=5, min_wait=0, max_wait=0)
    async def bad_request() -> str:
        calls["n"] += 1
        raise ProviderClientError("bad request")

    with pytest.raises(ProviderClientError):
        await bad_request()
    assert calls["n"] == 1


async def test_retry_treats_rate_limit_as_retryable():
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=3, min_wait=0, max_wait=0)
    async def rate_limited_then_ok() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise RateLimitError("slow down", retry_after=1.0)
        return "ok"

    assert await rate_limited_then_ok() == "ok"
    assert calls["n"] == 2
