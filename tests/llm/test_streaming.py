from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

import llm.providers.registry as registry_module
from llm.client import ChatClient
from llm.providers.base import OpenAICompatibleProvider
from llm.types import (
    ChatMessage,
    ProviderClientError,
    ProviderError,
    RateLimitError,
    StreamCompleted,
    StreamEvent,
    TextChunk,
    ToolCallsRequested,
)


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("TESTPROV_API_KEY", "test-key")


def sse(*frames: str) -> bytes:
    """Frames exactly as a provider writes them, blank line separated."""
    return "".join(f"{frame}\n\n" for frame in frames).encode()


def chunk(**delta: object) -> str:
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


def usage_frame(**usage: int) -> str:
    """The trailing frame a provider sends when usage was asked for."""
    return "data: " + json.dumps({"choices": [], "usage": usage})


def provider(handler, **kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="testprov",
        base_url="https://fake.test",
        api_key_env="TESTPROV_API_KEY",
        default_model="test-model",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://fake.test"
        ),
        **kwargs,
    )


def streaming(body: bytes, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    return handler


async def drain(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


async def test_text_arrives_in_order() -> None:
    body = sse(
        chunk(content="Aap "),
        chunk(content="eligible "),
        chunk(content="hain."),
        "data: [DONE]",
    )

    events = await drain(provider(streaming(body)).stream_completion([]))

    assert [event.text for event in events if isinstance(event, TextChunk)] == [
        "Aap ",
        "eligible ",
        "hain.",
    ]


async def test_the_last_event_is_always_the_completion() -> None:
    """Callers key off it to know the reply is whole. A stream that ends without
    one looks identical to a stream still in progress."""
    body = sse(chunk(content="hi"), "data: [DONE]")

    events = await drain(provider(streaming(body)).stream_completion([]))

    assert isinstance(events[-1], StreamCompleted)


async def test_tool_call_arguments_are_assembled_across_frames() -> None:
    """The reason the provider owns assembly rather than the caller.

    Arguments arrive a few characters at a time and the last frame is often just a
    closing brace. A caller handed the fragments would reassemble them, and every
    caller would get the indexing wrong in a different way.
    """
    body = sse(
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {"name": "check_eligibility", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"scheme_id"'}}]),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": ': "pm-kisan"}'}}]),
        "data: " + json.dumps({"choices": [{"finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    )

    events = await drain(provider(streaming(body)).stream_completion([]))
    requested = next(event for event in events if isinstance(event, ToolCallsRequested))

    assert len(requested.calls) == 1
    call = requested.calls[0]
    assert call.id == "call_abc"
    assert call.name == "check_eligibility"
    assert json.loads(call.arguments) == {"scheme_id": "pm-kisan"}


async def test_two_parallel_tool_calls_stay_separate() -> None:
    """Interleaved by index, which is the only thing distinguishing them, and the
    frames do not arrive grouped by call."""
    body = sse(
        chunk(
            tool_calls=[
                {"index": 0, "id": "a", "function": {"name": "find_schemes", "arguments": '{"q'}},
                {
                    "index": 1,
                    "id": "b",
                    "function": {"name": "check_eligibility", "arguments": "{"},
                },
            ]
        ),
        chunk(tool_calls=[{"index": 1, "function": {"arguments": '"scheme_id": "pm-jay"}'}}]),
        chunk(tool_calls=[{"index": 0, "function": {"arguments": 'uery": "ghar"}'}}]),
        "data: [DONE]",
    )

    events = await drain(provider(streaming(body)).stream_completion([]))
    calls = next(event for event in events if isinstance(event, ToolCallsRequested)).calls

    assert [call.id for call in calls] == ["a", "b"]
    assert json.loads(calls[0].arguments) == {"query": "ghar"}
    assert json.loads(calls[1].arguments) == {"scheme_id": "pm-jay"}


async def test_usage_comes_from_the_final_frame() -> None:
    """Streamed responses omit usage unless asked, so a streamed call with no
    tokens and no cost would put a blank column beside the unstreamed baseline the
    ablation compares against."""
    body = sse(
        chunk(content="hi"),
        usage_frame(prompt_tokens=11, completion_tokens=4),
        "data: [DONE]",
    )

    completed = (await drain(provider(streaming(body)).stream_completion([])))[-1]

    assert completed.tokens_in == 11
    assert completed.tokens_out == 4


async def test_cost_is_computed_from_the_streamed_usage() -> None:
    body = sse(
        usage_frame(prompt_tokens=1_000_000, completion_tokens=0),
        "data: [DONE]",
    )
    impl = provider(streaming(body), input_price_per_1m=2.0, output_price_per_1m=8.0)

    completed = (await drain(impl.stream_completion([])))[-1]

    assert completed.cost_usd == pytest.approx(2.0)


async def test_the_usage_parser_choice_is_respected_when_streaming() -> None:
    """Gemini bills reasoning tokens that appear in neither itemised field, so the
    non-streaming path takes `total_aware`. A streaming path that quietly used the
    default would undercount the same call by roughly eighteen times."""
    from llm.providers.base import total_aware_usage_parser

    body = sse(
        usage_frame(prompt_tokens=2, completion_tokens=9, total_tokens=197),
        "data: [DONE]",
    )
    impl = provider(streaming(body), usage_parser=total_aware_usage_parser)

    completed = (await drain(impl.stream_completion([])))[-1]

    assert completed.tokens_out == 195


async def test_keepalives_and_blank_lines_are_ignored() -> None:
    """A provider holds the connection open with comment frames while a reasoning
    model thinks. Treating one as data fails the turn before a word is spoken."""
    body = b": keepalive\n\n" + sse(chunk(content="hi"), "data: [DONE]")

    events = await drain(provider(streaming(body)).stream_completion([]))

    assert [event.text for event in events if isinstance(event, TextChunk)] == ["hi"]


async def test_an_unreadable_frame_does_not_end_the_stream() -> None:
    """Half a sentence has already been spoken by this point. Failing the turn over
    one bad frame is worse than continuing without it, and M3.2 is where the drop
    becomes a counted metric rather than only a log line."""
    body = sse(
        chunk(content="Aap "),
        "data: {not json at all",
        chunk(content="eligible hain."),
        "data: [DONE]",
    )

    events = await drain(provider(streaming(body)).stream_completion([]))

    assert [event.text for event in events if isinstance(event, TextChunk)] == [
        "Aap ",
        "eligible hain.",
    ]


async def test_frames_after_done_are_not_read() -> None:
    body = sse(chunk(content="hi"), "data: [DONE]", chunk(content="ignored"))

    events = await drain(provider(streaming(body)).stream_completion([]))

    assert [event.text for event in events if isinstance(event, TextChunk)] == ["hi"]


def unread(status: int, body: bytes):
    """A response whose body has not been buffered, as a real connection gives it.

    `httpx.Response(content=b"...")` is already read, so `.text` and `.json()`
    work without asking, and a test built that way passes whether or not the code
    reads the stream. Deleting the `aread` call left all twenty of these green
    until this helper existed. An async iterator body makes the response lazy, so
    touching `.json()` before reading raises the way it does in production.
    """

    async def chunks():
        yield body

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=chunks())

    return handler


async def test_a_429_carries_the_delay_the_provider_asked_for() -> None:
    """Reading the body at all is the point. A streamed response has to be read
    explicitly before `.json()` exists, so a status check written for the
    unstreamed path finds an unread body and loses the retry-after hint."""
    body = json.dumps({"error": {"message": "Please retry in 42s."}}).encode()

    with pytest.raises(RateLimitError) as raised:
        await drain(provider(unread(429, body)).stream_completion([]))

    assert raised.value.retry_after == 42.0


async def test_a_500_is_transient_and_a_400_is_not() -> None:
    with pytest.raises(ProviderError):
        await drain(provider(streaming(b"", status=503)).stream_completion([]))

    with pytest.raises(ProviderClientError):
        await drain(provider(streaming(b"", status=400)).stream_completion([]))


async def test_a_client_error_still_reports_the_body() -> None:
    """Proof the body was read rather than assumed. An empty message here is how a
    bad model name becomes an unexplained 400."""
    with pytest.raises(ProviderClientError) as raised:
        await drain(provider(unread(400, b"unknown model 'test-model'")).stream_completion([]))

    assert "unknown model" in str(raised.value)


async def test_usage_is_requested_so_it_can_be_reported() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=sse("data: [DONE]"))

    await drain(provider(handler).stream_completion([]))

    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}


async def test_usage_options_can_be_dropped_for_a_provider_that_rejects_them() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=sse("data: [DONE]"))

    await drain(provider(handler).stream_completion([], stream_options=None))

    assert "stream_options" not in seen[0]


async def test_default_params_are_merged_into_every_request() -> None:
    """`reasoning_effort` for Groq's gpt-oss models is the case this exists for:
    a provider-scoped default that every call needs, not something each call
    site should have to remember to ask for."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=sse("data: [DONE]"))

    impl = provider(handler, default_params={"reasoning_effort": "low"})
    await drain(impl.stream_completion([]))

    assert seen[0]["reasoning_effort"] == "low"


async def test_a_call_site_kwarg_overrides_a_default_param() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=sse("data: [DONE]"))

    impl = provider(handler, default_params={"reasoning_effort": "low"})
    await drain(impl.stream_completion([], reasoning_effort="high"))

    assert seen[0]["reasoning_effort"] == "high"


async def test_a_reasoning_budget_spent_with_no_content_is_a_failure_not_a_silent_reply() -> None:
    """A reasoning model can spend its whole token budget on the hidden reasoning
    channel and never reach an answer or a tool call. `finish_reason: length`
    with real content or a tool call already in hand is an ordinary truncation;
    this is the same shape as a dropped connection, from a different cause, and
    the same guard has to catch it or the turn ends having said nothing with
    nothing in the log to say why."""
    body = sse(
        chunk(role="assistant", content=""),
        chunk(reasoning="thinking and thinking"),
        "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        "data: [DONE]",
    )

    with pytest.raises(ProviderError, match="length limit"):
        await drain(provider(streaming(body)).stream_completion([]))


async def test_length_with_real_content_already_delivered_is_not_raised() -> None:
    """The new guard is for zero content, not for any truncation. Text already
    spoken must stay delivered, which is the property `provider.stream.truncated`
    already protects; this test is what stops the new guard from widening past it."""
    body = sse(
        chunk(content="Aap eligible"),
        "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        "data: [DONE]",
    )

    events = await drain(provider(streaming(body)).stream_completion([]))

    assert [e.text for e in events if isinstance(e, TextChunk)] == ["Aap eligible"]
    completed = events[-1]
    assert isinstance(completed, StreamCompleted)
    assert completed.finish_reason == "length"


class FlakyProvider:
    """Fails a set number of times before the first event, then streams."""

    name = "flaky"

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self.failures = failures
        self.attempts = 0
        self._error = error or ProviderError("flaky: server error 503")

    async def chat_completion(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def stream_completion(self, messages, **kwargs) -> AsyncIterator[StreamEvent]:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self._error
        yield TextChunk(text="recovered")
        yield StreamCompleted(finish_reason="stop")


class FailsMidStream:
    """Yields a token and then fails, which is the case that must not be retried."""

    name = "midstream"

    def __init__(self) -> None:
        self.attempts = 0

    async def chat_completion(self, messages, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def stream_completion(self, messages, **kwargs) -> AsyncIterator[StreamEvent]:
        self.attempts += 1
        yield TextChunk(text="Aap ")
        raise ProviderError("midstream: connection reset")


def client_for(impl, monkeypatch, **kwargs) -> ChatClient:
    monkeypatch.setattr("llm.client.get_provider", lambda name: impl)
    return ChatClient(**kwargs)


async def test_a_failure_before_the_first_event_is_retried(monkeypatch) -> None:
    impl = FlakyProvider(failures=2)
    client = client_for(impl, monkeypatch, max_retry_attempts=5)

    events = await drain(client.stream("flaky", [ChatMessage(role="user", content="hi")]))

    assert impl.attempts == 3
    assert [event.text for event in events if isinstance(event, TextChunk)] == ["recovered"]


async def test_a_failure_after_the_first_event_is_not_retried(monkeypatch) -> None:
    """The rule the whole streaming path is shaped around.

    Once a token is downstream the caller may already be synthesising it, so
    reconnecting would either say the opening words twice or splice two different
    replies together. The tokens delivered before the failure stay delivered and
    the error propagates.
    """
    impl = FailsMidStream()
    client = client_for(impl, monkeypatch, max_retry_attempts=5)

    seen: list[str] = []
    with pytest.raises(ProviderError):
        async for event in client.stream("midstream", [ChatMessage(role="user", content="hi")]):
            if isinstance(event, TextChunk):
                seen.append(event.text)

    assert impl.attempts == 1
    assert seen == ["Aap "]


async def test_retries_give_up_at_the_configured_limit(monkeypatch) -> None:
    impl = FlakyProvider(failures=99)
    client = client_for(impl, monkeypatch, max_retry_attempts=3)

    with pytest.raises(ProviderError):
        await drain(client.stream("flaky", [ChatMessage(role="user", content="hi")]))

    assert impl.attempts == 3


async def test_a_rate_limit_trips_the_throttle_before_the_next_attempt(monkeypatch) -> None:
    """The gate lives inside the retry loop for the reason the unstreamed path
    records: a 429 that asked for forty seconds is honoured by the next attempt
    rather than retried under a backoff curve that caps lower."""
    tripped: list[tuple[str, float | None]] = []

    class Throttle:
        async def is_open(self, provider: str) -> float:
            return 0.0

        async def trip(self, provider: str, retry_after: float | None) -> None:
            tripped.append((provider, retry_after))

    impl = FlakyProvider(failures=1, error=RateLimitError("flaky: 429", retry_after=40.0))
    client = client_for(impl, monkeypatch, throttle=Throttle(), max_retry_attempts=3)

    await drain(client.stream("flaky", [ChatMessage(role="user", content="hi")]))

    assert tripped == [("flaky", 40.0)]


async def test_the_gate_is_waited_on_before_streaming(monkeypatch) -> None:
    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    class CoolingThrottle:
        async def is_open(self, provider: str) -> float:
            return 2.5

        async def trip(self, provider: str, retry_after: float | None) -> None:  # pragma: no cover
            pass

    monkeypatch.setattr("llm.client.asyncio.sleep", record)
    impl = FlakyProvider(failures=0)
    client = client_for(impl, monkeypatch, throttle=CoolingThrottle())

    await drain(client.stream("flaky", [ChatMessage(role="user", content="hi")]))

    assert waited == [2.5]


# The frame Groq actually sent, with `failed_generation` left in at full length
# because the point of two of these tests is that it does not come back out.
TOOL_REJECTION = json.dumps(
    {
        "error": {
            "message": (
                "tool call validation failed: parameters for tool check_eligibility did "
                "not match schema: errors: [`/applicant/annual_income_inr`: expected "
                "integer, but got string]"
            ),
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": (
                '<function=check_eligibility>{"applicant": {"annual_income_inr": '
                '"50000", "state": "Bihar"}}</function>'
            ),
            "status_code": 400,
        }
    }
)


async def test_a_refusal_sent_with_a_200_is_reported_as_a_refusal() -> None:
    """The failure this whole path was built blind to.

    A provider writes the status before it has anything to refuse, so a tool call it
    rejects arrives as HTTP 200 with an error frame and no choices. Every branch in the
    parser looked at `choices`, so the stream ended having yielded nothing and hit the
    truncation guard, and a schema rejection came back worded as a dropped connection.

    On the deployed service that meant every reply was filler and then silence, with
    "stream ended before any content" in the log and no way to tell which of the two it
    had been.
    """
    stream = provider(streaming(sse("event: error", f"data: {TOOL_REJECTION}")))

    with pytest.raises(ProviderClientError) as raised:
        async for _event in stream.stream_completion([ChatMessage(role="user", content="hi")]):
            pass

    assert "tool_use_failed" in str(raised.value)
    assert "annual_income_inr" in str(raised.value)


async def test_a_refusal_is_not_retried(monkeypatch) -> None:
    """`ProviderClientError` is deliberately not a `ProviderError`, so the retry loop
    does not catch it. Before this the same rejection was sent five times over ten
    seconds of backoff, and it could never have succeeded on any of them: the arguments
    were wrong by construction and the model was given no new information."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, content=sse("event: error", f"data: {TOOL_REJECTION}"))

    monkeypatch.setitem(registry_module._PROVIDERS, "testprov", provider(handler))

    client = ChatClient()
    with pytest.raises(ProviderClientError):
        async for _event in client.stream(
            "testprov", [ChatMessage(role="user", content="hi")]
        ):
            pass

    assert attempts == 1


async def test_the_refusal_never_carries_the_arguments_back() -> None:
    """`failed_generation` holds what the model tried to send, which on this pipeline is
    an applicant's income. It is the one part of the body that must not travel, and the
    rest of the message is field paths and types, which is what makes it diagnosable."""
    stream = provider(streaming(sse("event: error", f"data: {TOOL_REJECTION}")))

    with pytest.raises(ProviderClientError) as raised:
        async for _event in stream.stream_completion([ChatMessage(role="user", content="hi")]):
            pass

    assert "50000" not in str(raised.value)
    assert "Bihar" not in str(raised.value)
    assert "failed_generation" not in str(raised.value)
