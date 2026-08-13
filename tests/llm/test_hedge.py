from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from llm.client import ChatClient
from llm.types import (
    ChatMessage,
    ProviderError,
    StreamCompleted,
    StreamEvent,
    TextChunk,
)


class Provider:
    """A provider with a settable time to first token, and a closed flag."""

    def __init__(
        self, name: str, ttft: float = 0.0, fail: bool = False, text: str = "reply"
    ) -> None:
        self.name = name
        self.ttft = ttft
        self.fail = fail
        self.text = text
        self.started = False
        self.closed = False

    async def chat_completion(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def stream_completion(self, messages, **kwargs) -> AsyncIterator[StreamEvent]:
        self.started = True
        try:
            await asyncio.sleep(self.ttft)
            if self.fail:
                raise ProviderError(f"{self.name}: server error 503")
            yield TextChunk(text=self.text)
            yield StreamCompleted(finish_reason="stop")
        finally:
            self.closed = True


# Captured at import, before the package conftest replaces it. That fixture makes
# `asyncio.sleep` a no-op so the retry tests do not wait out real backoff, which is
# right for them and wrong here: every test in this file is about which of two
# requests answered first, and with sleeps neutered a five second provider answers
# instantly.
_REAL_SLEEP = asyncio.sleep


@pytest.fixture(autouse=True)
def real_sleep(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _REAL_SLEEP)


def client_for(monkeypatch, **providers: Provider) -> ChatClient:
    monkeypatch.setattr("llm.client.get_provider", lambda name: providers[name])
    return ChatClient(max_retry_attempts=1)


async def collect(events: AsyncIterator[StreamEvent]) -> list[str]:
    return [event.text async for event in events if isinstance(event, TextChunk)]


def ask() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="kya main eligible hoon")]


async def test_a_prompt_primary_is_never_hedged() -> None:
    """The cost control. Hedging every call doubles token spend, so the second
    request only exists for the calls that were actually slow."""
    fast = Provider("fast", ttft=0.0)
    slow = Provider("backup", ttft=0.0)

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, fast=fast, backup=slow)
        heard = await collect(
            client.stream_hedged("fast", ask(), hedge_to="backup", hedge_after_ms=50)
        )

    assert heard == ["reply"]
    assert not slow.started


async def test_a_slow_primary_is_hedged_and_the_backup_can_win() -> None:
    slow = Provider("slow", ttft=5.0, text="from-primary")
    quick = Provider("backup", ttft=0.0, text="from-backup")

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, slow=slow, backup=quick)
        heard = await collect(
            client.stream_hedged("slow", ask(), hedge_to="backup", hedge_after_ms=10)
        )

    assert heard == ["from-backup"]
    assert slow.started


async def test_the_primary_still_wins_if_it_arrives_first_after_hedging() -> None:
    """The hedge is a race, not a handover. A primary that was merely close to the
    threshold keeps the turn, and the wasted second request is the price of the
    insurance rather than a reason to prefer the backup."""
    primary = Provider("primary", ttft=0.02, text="from-primary")
    backup = Provider("backup", ttft=5.0, text="from-backup")

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, primary=primary, backup=backup)
        heard = await collect(
            client.stream_hedged("primary", ask(), hedge_to="backup", hedge_after_ms=1)
        )

    assert heard == ["from-primary"]
    assert backup.started


async def test_the_losing_request_is_closed_rather_than_left_open() -> None:
    """An abandoned stream holds a connection against a quota somebody is counting,
    and on a free tier that is the whole allowance."""
    slow = Provider("slow", ttft=5.0)
    quick = Provider("backup", ttft=0.0)

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, slow=slow, backup=quick)
        await collect(
            client.stream_hedged("slow", ask(), hedge_to="backup", hedge_after_ms=10)
        )

    assert slow.closed


async def test_a_broken_primary_is_covered_by_the_hedge() -> None:
    """Most of the value. A failure does not win the race, so the hedge survives a
    provider that is broken as well as one that is slow."""
    broken = Provider("broken", ttft=0.0, fail=True)
    backup = Provider("backup", ttft=0.03, text="from-backup")

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, broken=broken, backup=backup)
        heard = await collect(
            client.stream_hedged("broken", ask(), hedge_to="backup", hedge_after_ms=10)
        )

    assert heard == ["from-backup"]


async def test_both_failing_still_raises() -> None:
    """A hedge is not a way to hide an outage. Two dead providers is an error the
    caller has to hear about."""
    broken = Provider("broken", ttft=0.0, fail=True)
    also_broken = Provider("backup", ttft=0.01, fail=True)

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, broken=broken, backup=also_broken)
        with pytest.raises(ProviderError):
            await collect(
                client.stream_hedged("broken", ask(), hedge_to="backup", hedge_after_ms=5)
            )


async def test_hedging_to_the_same_provider_is_refused() -> None:
    """The correlation trap. A provider having a bad second gives you a second bad
    second, and the two requests queue behind each other against one rate limit, so a
    same-provider hedge turns a slow call into a 429."""
    only = Provider("only", ttft=0.0)

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, only=only)
        with pytest.raises(ValueError):
            await collect(client.stream_hedged("only", ask(), hedge_to="only"))


async def test_the_whole_stream_follows_the_winner() -> None:
    """Not just its first token. Taking one event from the winner and the rest from
    the loser would splice two different replies into one sentence."""

    class Chatty:
        name = "chatty"

        def __init__(self, prefix: str, ttft: float) -> None:
            self._prefix = prefix
            self._ttft = ttft
            self.closed = False

        async def chat_completion(self, messages, **kwargs):  # pragma: no cover
            raise NotImplementedError

        async def stream_completion(self, messages, **kwargs) -> AsyncIterator[StreamEvent]:
            try:
                await asyncio.sleep(self._ttft)
                for index in range(3):
                    yield TextChunk(text=f"{self._prefix}{index}")
                yield StreamCompleted(finish_reason="stop")
            finally:
                self.closed = True

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, slow=Chatty("slow-", 5.0), backup=Chatty("fast-", 0.0))
        heard = await collect(
            client.stream_hedged("slow", ask(), hedge_to="backup", hedge_after_ms=10)
        )

    assert heard == ["fast-0", "fast-1", "fast-2"]


async def test_a_completion_event_still_arrives_from_the_winner() -> None:
    """Callers key off it to know the reply is whole, and the hedge must not eat it."""
    slow = Provider("slow", ttft=5.0)
    quick = Provider("backup", ttft=0.0)

    with pytest.MonkeyPatch.context() as patch:
        client = client_for(patch, slow=slow, backup=quick)
        events = [
            event
            async for event in client.stream_hedged(
                "slow", ask(), hedge_to="backup", hedge_after_ms=10
            )
        ]

    assert isinstance(events[-1], StreamCompleted)
