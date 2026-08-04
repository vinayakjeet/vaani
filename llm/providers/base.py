from __future__ import annotations

import os
from collections.abc import Callable
from typing import Protocol

import httpx

from llm.types import (
    ChatMessage,
    ChatResponse,
    ProviderClientError,
    ProviderConfigError,
    ProviderError,
    RateLimitError,
)


class Provider(Protocol):
    """Anything ChatClient can dispatch to. OpenAICompatibleProvider below covers
    groq/cerebras/openrouter/ollama/sarvam/gemini today; a future native provider
    with its own native message shape implements this Protocol directly instead
    of subclassing OpenAICompatibleProvider."""

    name: str

    async def chat_completion(
        self, messages: list[ChatMessage], **kwargs: object
    ) -> ChatResponse: ...


def default_usage_parser(payload: dict) -> tuple[int | None, int | None]:
    """Standard OpenAI usage shape: {"usage": {"prompt_tokens", "completion_tokens"}}."""
    usage = payload.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


class OpenAICompatibleProvider:
    """Backs any provider exposing an OpenAI-compatible POST {base_url}/chat/completions.

    Some OpenAI-compatible layers (Gemini's, Sarvam's) diverge slightly in how usage
    is reported - pass a custom `usage_parser` for those instead of assuming
    `default_usage_parser` fits every provider verbatim.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        default_model: str,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
        usage_parser: Callable[[dict], tuple[int | None, int | None]] = default_usage_parser,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._default_model = default_model
        self._input_price_per_1m = input_price_per_1m
        self._output_price_per_1m = output_price_per_1m
        self._usage_parser = usage_parser
        self._client = client

    def _require_api_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise ProviderConfigError(
                f"provider '{self.name}' requires env var {self._api_key_env} to be set"
            )
        return key

    def _cost_usd(self, tokens_in: int | None, tokens_out: int | None) -> float | None:
        if self._input_price_per_1m is None or self._output_price_per_1m is None:
            return None
        if tokens_in is None or tokens_out is None:
            return None
        return (tokens_in / 1_000_000) * self._input_price_per_1m + (
            tokens_out / 1_000_000
        ) * self._output_price_per_1m

    async def chat_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> ChatResponse:
        api_key = self._require_api_key()
        resolved_model = model or self._default_model
        payload = {
            "model": resolved_model,
            "messages": [m.model_dump() for m in messages],
            **kwargs,
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        client = self._client or httpx.AsyncClient(base_url=self._base_url)
        try:
            try:
                resp = await client.post("/chat/completions", json=payload, headers=headers)
            except httpx.RequestError as exc:
                raise ProviderError(f"{self.name}: network error: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()

        if resp.status_code == 429:
            retry_after_header = resp.headers.get("Retry-After")
            retry_after = float(retry_after_header) if retry_after_header else None
            raise RateLimitError(f"{self.name}: rate limited", retry_after=retry_after)
        if resp.status_code >= 500:
            raise ProviderError(f"{self.name}: server error {resp.status_code}")
        if resp.status_code >= 400:
            raise ProviderClientError(f"{self.name}: client error {resp.status_code}: {resp.text}")

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens_in, tokens_out = self._usage_parser(data)

        return ChatResponse(
            text=text,
            provider=self.name,
            model=resolved_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost_usd(tokens_in, tokens_out),
        )
