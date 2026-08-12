from __future__ import annotations

import json
from collections.abc import AsyncIterator

from llm.types import (
    ChatMessage,
    ChatResponse,
    StreamCompleted,
    StreamEvent,
    TextChunk,
    ToolCall,
    ToolCallsRequested,
)

# Enough of a trigger to exercise the tool loop without a network or a key. It
# exists so the wiring can be run and interrupted locally, never to measure
# anything: a stub that answers on cue measures the person who wrote the cue.
# Every number in the ablation comes from a real provider.
TOOL_TRIGGER = "scheme"


class MockProvider:
    """Deterministic, no-network provider. Always registered, needs no API key.

    This is the default LLM_PROVIDER so `docker compose up`, CI, and the demo
    route all work with zero API keys out of the box.
    """

    name = "mock"

    async def chat_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> ChatResponse:
        last_user = _last_user(messages)
        return ChatResponse(
            text=f"mock reply: {last_user}",
            provider=self.name,
            model=model or "mock-echo",
            tokens_in=len(last_user.split()),
            tokens_out=len(last_user.split()) + 2,
            cost_usd=0.0,
        )

    async def stream_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        """The same reply, word by word, so the streamed path has something to run.

        Chunked per word rather than per character. A single chunk carrying the
        whole reply would let a caller that never streams pass every test.
        """
        last_user = _last_user(messages)

        if TOOL_TRIGGER in last_user.casefold() and not _has_tool_result(messages):
            yield ToolCallsRequested(
                calls=[
                    ToolCall(
                        id="mock-call-1",
                        name="find_schemes",
                        # Split across no fragments here, but the provider layer
                        # that assembles real fragments is tested separately.
                        arguments=json.dumps({"query": last_user}),
                    )
                ]
            )
            yield StreamCompleted(
                finish_reason="tool_calls", tokens_in=0, tokens_out=0, cost_usd=0.0
            )
            return

        reply = f"mock reply: {last_user}"
        for word in reply.split(" "):
            yield TextChunk(text=word + " ")

        yield StreamCompleted(
            finish_reason="stop",
            tokens_in=len(last_user.split()),
            tokens_out=len(reply.split()),
            cost_usd=0.0,
        )


def _last_user(messages: list[ChatMessage]) -> str:
    return next((m.content for m in reversed(messages) if m.role == "user"), "")


def _has_tool_result(messages: list[ChatMessage]) -> bool:
    """Whether this turn has already been round-tripped through a tool.

    Without this the mock asks for the same tool forever, which is a loop rather
    than a fixture, and the second half of the round trip never gets exercised.
    """
    return any(message.role == "tool" for message in messages)
