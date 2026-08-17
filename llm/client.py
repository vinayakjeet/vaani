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

# When the second request goes out. Belongs at the primary's measured p90, so that
# hedging costs extra tokens on roughly a tenth of calls rather than most of them.
# Measured 2026-08-17 against openai/gpt-oss-120b (`bench/prompt_cache.py`'s own
# twenty real calls, time to first token): p50 499.7ms, p90 833.9ms. The previous
# value here, 300ms, was chosen before any measurement existed and turned out to
# be below every single one of those twenty samples, so the hedge as shipped fired
# on effectively 100% of calls rather than the intended tail, doubling real spend
# on ordinary traffic. See DECISIONS.
DEFAULT_HEDGE_AFTER_MS = 834


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

    async def stream_hedged(
        self,
        provider: str,
        messages: list[ChatMessage],
        *,
        hedge_to: str,
        hedge_after_ms: int = DEFAULT_HEDGE_AFTER_MS,
        **kwargs: object,
    ) -> AsyncIterator[StreamEvent]:
        """Race a second request against a slow first one, and keep the winner.

        This is for a tail, not a median. A hard deadline with a failover makes the
        tail worse rather than better: a missed deadline costs both attempts, so the
        wasted wait plus a second time-to-first-token is spent in series and can
        exceed the whole budget it was meant to protect. Hedging makes the worst case
        the larger of the two rather than their sum.

        `hedge_to` is required and must be a different provider, because hedging to
        the same one correlates the very failures it is meant to survive: a provider
        having a bad second gives you a second bad second, and the retry is spent
        against the same queue that was already slow. It also doubles the pressure on
        one rate limit, which on a free tier is how a hedge turns a slow call into a
        429.

        The loser is cancelled and its stream closed, so the abandoned request
        releases its connection rather than waiting for a collector.
        """
        if hedge_to == provider:
            raise ValueError(
                "hedge_to must differ from provider: hedging to the same provider "
                "correlates the failure it exists to survive"
            )

        primary = self.stream(provider, messages, **kwargs)
        primary_first = asyncio.ensure_future(anext(primary, None))
        done, _pending = await asyncio.wait([primary_first], timeout=hedge_after_ms / 1000)

        if done and not primary_first.cancelled() and primary_first.exception() is None:
            async for event in _continue(primary, primary_first.result()):
                yield event
            return

        secondary = self.stream(hedge_to, messages, **kwargs)
        secondary_first = asyncio.ensure_future(anext(secondary, None))
        logger.info(
            "llm.hedged", provider=provider, hedge_to=hedge_to, after_ms=hedge_after_ms
        )

        winner, loser, first = await _first_usable(
            (primary, primary_first, provider), (secondary, secondary_first, hedge_to)
        )
        await _abandon(*loser)

        async for event in _continue(winner, first):
            yield event

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


async def _continue(
    stream: AsyncIterator[StreamEvent], first: StreamEvent | None
) -> AsyncIterator[StreamEvent]:
    """Re-attach an already-taken first event to the front of its own stream."""
    if first is None:
        return
    yield first
    async for event in stream:
        yield event


async def _first_usable(
    left: tuple[AsyncIterator[StreamEvent], asyncio.Future, str],
    right: tuple[AsyncIterator[StreamEvent], asyncio.Future, str],
) -> tuple[
    AsyncIterator[StreamEvent],
    tuple[AsyncIterator[StreamEvent], asyncio.Future],
    StreamEvent | None,
]:
    """Whichever request produces an event first, with the other returned to abandon.

    A failure does not win. If the first to finish raised, the other is waited for
    rather than the whole call failing, which is most of the value: the hedge covers
    a provider that is broken as well as one that is slow.
    """
    pending = {left[1], right[1]}
    by_future = {left[1]: left, right[1]: right}
    last_error: BaseException | None = None

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for future in done:
            stream, _future, name = by_future[future]
            error = future.exception()
            if error is not None:
                logger.info("llm.hedge_lost", provider=name, error=type(error).__name__)
                last_error = error
                continue
            other = right if future is left[1] else left
            return stream, (other[0], other[1]), future.result()

    if last_error is not None:
        raise last_error
    raise ProviderError("both hedged requests produced nothing")


async def _abandon(stream: AsyncIterator[StreamEvent], pending: asyncio.Future) -> None:
    """Close the losing stream so its connection is released now, not eventually.

    The pending `anext` is cancelled first, and that order is the whole method.
    `aclose` on a generator that is currently being advanced raises "asynchronous
    generator is already running", so closing while the losing request is still
    waiting for its first token does nothing at all. It went unnoticed because the
    failure was swallowed by a bare `except Exception`, which is why only
    `RuntimeError` is tolerated here now and anything else is logged.
    """
    pending.cancel()
    # `gather` with `return_exceptions` rather than a bare await, because the loser
    # may have already failed and awaiting a failed future re-raises it. That failure
    # is the reason the hedge was fired and the backup answered, so surfacing it here
    # would turn a call the user heard answered into an error.
    await asyncio.gather(pending, return_exceptions=True)

    try:
        await stream.aclose()  # type: ignore[attr-defined]
    except RuntimeError:
        # Already finishing on its own. Nothing left to release.
        pass
    except Exception as exc:
        logger.info("llm.hedge_abandon_failed", error=type(exc).__name__)
