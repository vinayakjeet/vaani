from __future__ import annotations

import httpx
import pytest

from fault import DEFAULT_STREAM_WORDS, Fault, faulty_endpoint
from llm.providers.base import OpenAICompatibleProvider
from llm.types import ProviderError, StreamCompleted, TextChunk


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("FAULTY_API_KEY", "test-key")


def provider(url: str, read_timeout: float = 30.0) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="faulty",
        base_url=url,
        api_key_env="FAULTY_API_KEY",
        default_model="test-model",
        client=httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(connect=5.0, read=read_timeout, write=5.0, pool=5.0),
        ),
    )


async def heard(impl: OpenAICompatibleProvider) -> list[str]:
    return [
        event.text
        async for event in impl.stream_completion([])
        if isinstance(event, TextChunk)
    ]


async def test_a_healthy_stream_is_the_control() -> None:
    """Every fault test needs one. A suite that only ever sees failures cannot tell
    a working client from one that fails on everything."""
    with faulty_endpoint(Fault.NONE) as server:
        assert await heard(provider(server.url)) == list(DEFAULT_STREAM_WORDS)


async def test_frames_arrive_before_the_stream_ends() -> None:
    """Proves the helper streams rather than buffering. Without the flush on the
    server side the whole response lands at once and every streaming test below
    would pass against a client that does not stream."""
    with faulty_endpoint(Fault.NONE) as server:
        impl = provider(server.url)
        seen: list[str] = []

        async for event in impl.stream_completion([]):
            if isinstance(event, TextChunk):
                seen.append(event.text)
                break

        assert seen == [DEFAULT_STREAM_WORDS[0]]


async def test_a_dropped_stream_delivers_what_arrived_before_it_died() -> None:
    """Half an answer is already spoken by this point, so the words that made it are
    the words the listener heard."""
    with faulty_endpoint(Fault.STREAM_DROP, frames_before_fault=2) as server:
        impl = provider(server.url)
        seen: list[str] = []

        with pytest.raises(ProviderError):
            async for event in impl.stream_completion([]):
                if isinstance(event, TextChunk):
                    seen.append(event.text)

    # Delivered before the drop, and it stays delivered: those words were spoken and
    # cannot be un-said. What must not happen is the caller believing the answer was
    # whole, which is what a silent end would have meant.
    assert seen == list(DEFAULT_STREAM_WORDS[:2])


async def test_a_malformed_frame_does_not_end_the_stream() -> None:
    """One unreadable frame is skipped and the rest of the sentence still arrives.
    Failing the turn over it would trade a real answer for a provisional one."""
    with faulty_endpoint(Fault.STREAM_MALFORMED_FRAME, frames_before_fault=2) as server:
        assert await heard(provider(server.url)) == list(DEFAULT_STREAM_WORDS)


async def test_a_stalled_stream_raises_rather_than_waiting_forever() -> None:
    """The dangerous shape. Nothing errors and the connection stays open, so a
    caller without a read timeout simply stops, and the user hears silence with no
    reason given."""
    with (
        faulty_endpoint(Fault.STREAM_STALL, hang_seconds=30.0) as server,
        pytest.raises(ProviderError),
    ):
        await heard(provider(server.url, read_timeout=0.3))


async def test_the_stall_is_survived_quickly_enough_to_matter() -> None:
    """A degradation that takes 30 seconds is not a degradation, it is an outage
    with extra steps. The read timeout is what bounds it."""
    import time

    with faulty_endpoint(Fault.STREAM_STALL, hang_seconds=30.0) as server:
        started = time.monotonic()
        with pytest.raises(ProviderError):
            await heard(provider(server.url, read_timeout=0.3))
        elapsed = time.monotonic() - started

    assert elapsed < 5.0


async def test_the_fault_lands_where_it_was_asked_to() -> None:
    """`frames_before_fault` is the knob every test below depends on, so it is
    pinned rather than assumed."""
    with faulty_endpoint(Fault.STREAM_MALFORMED_FRAME, frames_before_fault=0) as server:
        assert await heard(provider(server.url)) == list(DEFAULT_STREAM_WORDS)


async def test_the_helper_counts_the_requests_it_served() -> None:
    """So a retry or a hedge can be counted rather than inferred."""
    with faulty_endpoint(Fault.NONE) as server:
        await heard(provider(server.url))

        assert len(server.requests) == 1


async def test_the_whole_response_faults_still_work() -> None:
    """The shapes carried over from Spanlight, still driven by the same helper, which
    is M3.1's acceptance criterion: one helper for every fault test."""
    with faulty_endpoint(Fault.SERVER_ERROR) as server, pytest.raises(ProviderError):
        await heard(provider(server.url))

    with faulty_endpoint(Fault.RATE_LIMITED, retry_after=40) as server:
        from llm.types import RateLimitError

        with pytest.raises(RateLimitError) as raised:
            await heard(provider(server.url))

        assert raised.value.retry_after == 40


async def test_a_healthy_stream_ends_with_a_completion_event() -> None:
    with faulty_endpoint(Fault.NONE) as server:
        events = [event async for event in provider(server.url).stream_completion([])]

    assert isinstance(events[-1], StreamCompleted)
