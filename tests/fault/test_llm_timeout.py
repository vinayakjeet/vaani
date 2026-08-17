from __future__ import annotations

import asyncio

import httpx
import pytest

import llm.client as client_module
from fault import Fault, faulty_endpoint
from llm.client import ChatClient
from llm.providers.base import OpenAICompatibleProvider
from vaani.llm_turn import COULD_NOT_CHECK, StreamedTurn


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("FAULTY_API_KEY", "test-key")


def wire_faulty_provider(monkeypatch, url: str, read_timeout: float = 0.2) -> None:
    """Point `ChatClient`'s provider lookup at a real, faulty HTTP endpoint.

    A short client-side read timeout, well inside the server's own hang, is what
    turns SPEC S5's dependency-that-is-slow into a bounded failure rather than a
    turn that waits out the server's own 30 second hang before this test does.
    """
    impl = OpenAICompatibleProvider(
        name="faulty",
        base_url=url,
        api_key_env="FAULTY_API_KEY",
        default_model="test-model",
        client=httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(connect=2.0, read=read_timeout, write=2.0, pool=2.0),
        ),
    )
    monkeypatch.setattr(client_module, "get_provider", lambda _name: impl)


async def drain(turn: StreamedTurn, question: str) -> list[str]:
    return [chunk async for chunk in turn.run(question, history=[])]


async def test_a_hung_provider_ends_the_turn_rather_than_the_turn_hanging(monkeypatch) -> None:
    """SPEC S5's shape at the model rather than the recogniser: a dependency that
    is slow is more dangerous than one that is down, because nothing reports an
    error and the caller simply stops. `ChatClient.stream`'s retry loop only
    retries until the first token, so one hung attempt against a tight read
    timeout, with retries capped, has to end the turn in a bounded time."""
    with faulty_endpoint(Fault.HANG, hang_seconds=30.0) as server:
        wire_faulty_provider(monkeypatch, server.url)
        turn = StreamedTurn(llm=ChatClient(max_retry_attempts=1), provider="faulty")

        with pytest.raises(Exception):
            await asyncio.wait_for(
                drain(turn, "kya main eligible hoon"), timeout=5.0
            )


async def test_a_hung_tool_round_degrades_to_a_stated_failure(monkeypatch) -> None:
    """The other shape S5 names: a hang after the model has already committed to a
    tool call. `_rounds` only catches `ProviderClientError`, not a bare
    `ProviderError` from a dropped connection, and a hang wrapped as one must not
    read as `_rounds` having nothing left to say."""
    with faulty_endpoint(Fault.HANG, hang_seconds=30.0) as server:
        wire_faulty_provider(monkeypatch, server.url)
        turn = StreamedTurn(llm=ChatClient(max_retry_attempts=1), provider="faulty")

        with pytest.raises(Exception) as raised:
            await asyncio.wait_for(
                drain(turn, "PM Kisan ke baare mein bataiye"), timeout=5.0
            )

        # Never a silent empty result, and never the confident degradation text
        # standing in for a real exception being swallowed: this path is meant to
        # surface the failure to `_play`'s own reporting, not paper over it here.
        assert COULD_NOT_CHECK not in str(raised.value)
