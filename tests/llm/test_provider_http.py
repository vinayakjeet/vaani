from __future__ import annotations

import httpx
import pytest

from llm.providers.base import OpenAICompatibleProvider
from llm.types import (
    ChatMessage,
    ProviderClientError,
    ProviderConfigError,
    ProviderError,
    RateLimitError,
)


def _client_with_handler(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://fake.test")


def _success_response(text: str = "hello", prompt_tokens: int = 10, completion_tokens: int = 5):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def _provider(handler, **kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="testprov",
        base_url="https://fake.test",
        api_key_env="TESTPROV_API_KEY",
        default_model="test-model",
        client=_client_with_handler(handler),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("TESTPROV_API_KEY", "test-key")


async def test_chat_completion_success_parses_text_and_usage():
    provider = _provider(lambda request: _success_response())
    response = await provider.chat_completion([ChatMessage(role="user", content="hi")])
    assert response.text == "hello"
    assert response.tokens_in == 10
    assert response.tokens_out == 5
    assert response.provider == "testprov"
    assert response.model == "test-model"


async def test_chat_completion_429_raises_rate_limit_error_with_retry_after():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "rate limited"})

    provider = _provider(handler)
    with pytest.raises(RateLimitError) as exc_info:
        await provider.chat_completion([ChatMessage(role="user", content="hi")])
    assert exc_info.value.retry_after == 7.0


async def test_chat_completion_429_without_retry_after_header():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    provider = _provider(handler)
    with pytest.raises(RateLimitError) as exc_info:
        await provider.chat_completion([ChatMessage(role="user", content="hi")])
    assert exc_info.value.retry_after is None


async def test_chat_completion_500_raises_provider_error():
    provider = _provider(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(ProviderError):
        await provider.chat_completion([ChatMessage(role="user", content="hi")])


async def test_chat_completion_400_raises_client_error_not_provider_error():
    provider = _provider(lambda request: httpx.Response(400, text="bad request"))
    with pytest.raises(ProviderClientError):
        await provider.chat_completion([ChatMessage(role="user", content="hi")])


async def test_chat_completion_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("TESTPROV_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(
        name="testprov",
        base_url="https://fake.test",
        api_key_env="TESTPROV_API_KEY",
        default_model="m",
    )
    with pytest.raises(ProviderConfigError):
        await provider.chat_completion([ChatMessage(role="user", content="hi")])


async def test_cost_usd_computed_from_pricing():
    provider = _provider(
        lambda request: _success_response(prompt_tokens=1000, completion_tokens=500),
        input_price_per_1m=2.0,
        output_price_per_1m=4.0,
    )
    response = await provider.chat_completion([ChatMessage(role="user", content="hi")])
    assert response.cost_usd == pytest.approx(1000 / 1_000_000 * 2.0 + 500 / 1_000_000 * 4.0)


async def test_cost_usd_none_when_pricing_unknown():
    provider = _provider(lambda request: _success_response())
    response = await provider.chat_completion([ChatMessage(role="user", content="hi")])
    assert response.cost_usd is None
