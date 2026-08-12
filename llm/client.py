from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import structlog

from llm.providers.registry import get_provider
from llm.retry import backoff_seconds, retry_with_backoff
from llm.throttle import InMemoryThrottle, ThrottleBackend
from llm.types import (
    ChatMessage,
    ChatResponse,
    ProviderError,
    RateLimitError,
    StreamEvent,
)

logger = structlog.get_logger(__name__)


class ChatClient:
    """Provider-agnostic chat-completion client.

    Call flow: throttle gate (queue behind an active 429 cooldown) -> retry with
    exponential backoff + jitter (transient errors only) -> provider HTTP call ->
    structured cost/token/latency log.
    """

    def __init__(
        self,
        throttle: ThrottleBackend | None = None,
        max_retry_attempts: int = 5,
    ) -> None:
        self._throttle = throttle or InMemoryThrottle()
        self._max_retry_attempts = max_retry_attempts

    async def complete(
        self, provider: str, messages: list[ChatMessage], **kwargs: object
    ) -> ChatResponse:
        provider_impl = get_provider(provider)

        @retry_with_backoff(max_attempts=self._max_retry_attempts)
        async def _attempt() -> ChatResponse:
            # The gate is checked inside the retry loop, not once before it.
            # A 429 trips the throttle with the delay the provider asked for, and
            # only a gate inside the loop makes the next attempt honour it.
            # Checking once outside meant retries fell back to exponential
            # backoff, which caps well below what a provider can ask for: a real
            # Gemini 429 requested 40 seconds while five attempts of backoff
            # totalled about 31, so every retry was spent while still rate
            # limited and the call failed with quota to spare.
            wait = await self._throttle.is_open(provider)
            if wait > 0:
                await asyncio.sleep(wait)

            try:
                return await provider_impl.chat_completion(messages, **kwargs)
            except RateLimitError as exc:
                await self._throttle.trip(provider, exc.retry_after)
                raise

        start = time.monotonic()
        response = await _attempt()
        response.latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "llm.call",
            provider=response.provider,
            model=response.model,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        )
        return response

    async def stream(
        self, provider: str, messages: list[ChatMessage], **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        """Token events as they arrive, retried only until the first one.

        The retry rule is the whole reason this does not use
        `retry_with_backoff`. A decorator sees one call succeed or fail, but a
        stream stops being repeatable the moment a token has been handed
        downstream: by then the caller may already be synthesising it, and
        reconnecting would either say the opening words twice or splice two
        different replies into one sentence. So a failure before the first event
        is transient and retried, and a failure after it propagates. That is a
        weaker guarantee than the unstreamed path gives, and it is the honest one.

        `time_to_first_event_ms` is logged rather than only the total. Wrapping a
        stream and reporting its duration measures how long the model talked for,
        which is not the number a listener experiences, and reporting only the
        total is how a streamed call and an unstreamed one come out looking alike.
        """
        provider_impl = get_provider(provider)
        start = time.monotonic()
        attempt = 0

        while True:
            attempt += 1
            # Same reasoning as `complete`: the gate belongs inside the loop, so a
            # 429 that asked for forty seconds is honoured by the next attempt
            # instead of being retried under exponential backoff that caps lower.
            wait = await self._throttle.is_open(provider)
            if wait > 0:
                await asyncio.sleep(wait)

            started = False
            try:
                async for event in provider_impl.stream_completion(messages, **kwargs):
                    if not started:
                        started = True
                        logger.info(
                            "llm.stream.first_event",
                            provider=provider,
                            time_to_first_event_ms=(time.monotonic() - start) * 1000,
                            attempts=attempt,
                        )
                    yield event
                return
            except RateLimitError as exc:
                await self._throttle.trip(provider, exc.retry_after)
                if started or attempt >= self._max_retry_attempts:
                    raise
            except ProviderError:
                if started or attempt >= self._max_retry_attempts:
                    raise

            await asyncio.sleep(backoff_seconds(attempt))
