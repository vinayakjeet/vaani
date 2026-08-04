from __future__ import annotations

import pytest
from structlog.testing import capture_logs

import llm.providers.registry as registry_module
from llm.client import ChatClient
from llm.throttle import InMemoryThrottle
from llm.types import ChatMessage, ChatResponse, ProviderError, RateLimitError


class _StubProvider:
    def __init__(self, name: str, responses: list) -> None:
        self.name = name
        self._responses = list(responses)
        self.calls = 0

    async def chat_completion(self, messages: list[ChatMessage], **kwargs: object) -> ChatResponse:
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok_response(**overrides: object) -> ChatResponse:
    defaults = {
        "text": "hi",
        "provider": "stub",
        "model": "m",
        "tokens_in": 3,
        "tokens_out": 4,
        "cost_usd": 0.001,
    }
    defaults.update(overrides)
    return ChatResponse(**defaults)


@pytest.fixture
def register_stub(monkeypatch):
    def _register(provider: _StubProvider) -> _StubProvider:
        monkeypatch.setitem(registry_module._PROVIDERS, provider.name, provider)
        return provider

    return _register


async def test_complete_retries_transient_failures_then_succeeds(register_stub):
    stub = register_stub(_StubProvider("stub", [ProviderError("boom"), _ok_response()]))
    client = ChatClient(max_retry_attempts=5)
    response = await client.complete("stub", [ChatMessage(role="user", content="hi")])
    assert response.text == "hi"
    assert stub.calls == 2


async def test_complete_exhausts_retries_and_raises(register_stub):
    stub = register_stub(_StubProvider("stub", [ProviderError("boom")] * 5))
    client = ChatClient(max_retry_attempts=3)
    with pytest.raises(ProviderError):
        await client.complete("stub", [ChatMessage(role="user", content="hi")])
    assert stub.calls == 3


async def test_complete_trips_throttle_on_rate_limit(register_stub):
    register_stub(
        _StubProvider("stub", [RateLimitError("slow down", retry_after=42.0), _ok_response()])
    )
    throttle = InMemoryThrottle()
    client = ChatClient(throttle=throttle, max_retry_attempts=3)
    await client.complete("stub", [ChatMessage(role="user", content="hi")])
    assert await throttle.is_open("stub") > 0


async def test_complete_waits_for_throttle_before_calling_provider(monkeypatch, register_stub):
    stub = register_stub(_StubProvider("stub", [_ok_response()]))

    waited = []

    async def fake_sleep(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr("llm.client.asyncio.sleep", fake_sleep)

    class StubThrottle:
        async def is_open(self, provider: str) -> float:
            return 2.5

        async def trip(self, provider: str, retry_after: float | None) -> None:
            pass

    client = ChatClient(throttle=StubThrottle())
    await client.complete("stub", [ChatMessage(role="user", content="hi")])
    assert waited == [2.5]
    assert stub.calls == 1


async def test_complete_logs_cost_and_token_fields():
    with capture_logs() as logs:
        client = ChatClient()
        await client.complete("mock", [ChatMessage(role="user", content="hello there")])

    call_events = [entry for entry in logs if entry.get("event") == "llm.call"]
    assert len(call_events) == 1
    entry = call_events[0]
    for field in ("provider", "model", "tokens_in", "tokens_out", "cost_usd", "latency_ms"):
        assert field in entry
    assert entry["provider"] == "mock"
