from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from llm.types import (
    ChatMessage,
    StreamCompleted,
    StreamEvent,
    TextChunk,
    ToolCall,
    ToolCallsRequested,
)
from vaani.llm_turn import COULD_NOT_CHECK, StreamedTurn


class ScriptedClient:
    """A ChatClient stand-in that replays one scripted stream per round.

    Records the messages it was handed each time, because the tool round trip is
    mostly a claim about what the second request contains.
    """

    def __init__(self, *rounds: list[StreamEvent]) -> None:
        self._rounds = list(rounds)
        self.seen: list[list[ChatMessage]] = []
        self.kwargs: list[dict] = []

    async def stream(
        self, provider: str, messages: list[ChatMessage], **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        self.seen.append([message.model_copy(deep=True) for message in messages])
        self.kwargs.append(kwargs)

        if not self._rounds:
            raise AssertionError("the turn asked for more rounds than were scripted")

        for event in self._rounds.pop(0):
            yield event


def text(*words: str) -> list[StreamEvent]:
    return [TextChunk(text=word) for word in words] + [StreamCompleted(finish_reason="stop")]


def asks_for(name: str, arguments: str, call_id: str = "call_1") -> list[StreamEvent]:
    return [
        ToolCallsRequested(calls=[ToolCall(id=call_id, name=name, arguments=arguments)]),
        StreamCompleted(finish_reason="tool_calls"),
    ]


def turn(*rounds: list[StreamEvent]) -> tuple[StreamedTurn, ScriptedClient]:
    client = ScriptedClient(*rounds)
    return StreamedTurn(llm=client, provider="scripted"), client


async def collect(streamed: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in streamed])


async def test_reply_text_streams_through_in_order() -> None:
    streamed, _ = turn(text("Aap ", "eligible ", "hain."))

    assert await collect(streamed.run("kya main eligible hoon")) == "Aap eligible hain."


async def test_the_first_chunk_arrives_before_the_reply_is_complete() -> None:
    """The acceptance criterion for M1.3, and it has to be about ordering.

    Asserting only the assembled text would pass against an implementation that
    buffers the whole reply and returns it at the end, which is the M0 baseline
    this milestone exists to beat.
    """
    streamed, _ = turn(text("Aap ", "eligible ", "hain."))
    chunks: list[str] = []

    async for chunk in streamed.run("kya main eligible hoon"):
        chunks.append(chunk)
        if len(chunks) == 1:
            break

    assert chunks == ["Aap "]


async def test_a_tool_round_trip_completes_inside_the_turn() -> None:
    streamed, client = turn(
        asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("Aapke liye PM Awas Yojana hai."),
    )

    assert await collect(streamed.run("mujhe ghar chahiye")) == "Aapke liye PM Awas Yojana hai."
    assert len(client.seen) == 2


async def test_the_tool_result_reaches_the_model() -> None:
    """Not merely that a second round happened, but that it carried the answer.

    A loop that re-asked without the result would still produce two rounds and
    still produce a reply, and the reply would be invented.
    """
    streamed, client = turn(
        asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("PM Awas Yojana."),
    )
    await collect(streamed.run("mujhe ghar chahiye"))

    tool_messages = [m for m in client.seen[1] if m.role == "tool"]

    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0].content)["schemes"][0]["scheme_id"] == "pmay-g"


async def test_the_tool_result_is_tied_to_the_call_that_asked_for_it() -> None:
    """Providers reject a `tool` message whose id matches no call in the preceding
    assistant message, so this is what makes the second round accepted at all."""
    streamed, client = turn(
        asks_for("find_schemes", json.dumps({"query": "ghar"}), call_id="call_xyz"),
        text("ok"),
    )
    await collect(streamed.run("ghar"))

    assistant = next(m for m in client.seen[1] if m.role == "assistant")
    tool_message = next(m for m in client.seen[1] if m.role == "tool")

    assert assistant.tool_calls[0]["id"] == "call_xyz"
    assert assistant.tool_calls[0]["function"]["name"] == "find_schemes"
    assert tool_message.tool_call_id == "call_xyz"


async def test_the_tools_are_advertised_on_every_request() -> None:
    """A model cannot call what it was not offered, and the second round is the
    one where it is easy to forget."""
    streamed, client = turn(
        asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("ok"),
    )
    await collect(streamed.run("ghar"))

    for kwargs in client.kwargs:
        advertised = {schema["function"]["name"] for schema in kwargs["tools"]}
        assert advertised == {"check_eligibility", "find_schemes"}


async def test_a_rejected_tool_call_is_reported_rather_than_raised() -> None:
    """The model is the only thing that can turn "no such scheme" into a sentence
    a listener understands, so the failure goes back to it as a result."""
    streamed, client = turn(
        asks_for("check_eligibility", json.dumps({"scheme_id": "not-real", "applicant": {}})),
        text("Mujhe woh scheme nahi mili."),
    )

    assert await collect(streamed.run("kya main eligible hoon"))
    tool_message = next(m for m in client.seen[1] if m.role == "tool")
    payload = json.loads(tool_message.content)

    assert payload["completed"] is False
    assert payload["error"]


async def test_truncated_tool_arguments_do_not_crash_the_turn() -> None:
    """Models run out of tokens mid-JSON often enough that this is an ordinary
    branch. The turn has to stay audible."""
    streamed, client = turn(
        asks_for("find_schemes", '{"query": "gha'),
        text("Ek baar phir poochhiye."),
    )

    assert await collect(streamed.run("ghar")) == "Ek baar phir poochhiye."
    assert json.loads(next(m for m in client.seen[1] if m.role == "tool").content)[
        "completed"
    ] is False


async def test_arguments_that_are_not_an_object_are_reported() -> None:
    streamed, client = turn(asks_for("find_schemes", "[1, 2, 3]"), text("ok"))
    await collect(streamed.run("ghar"))

    assert json.loads(next(m for m in client.seen[1] if m.role == "tool").content)[
        "completed"
    ] is False


async def test_an_unknown_tool_name_is_reported_rather_than_guessed() -> None:
    streamed, client = turn(asks_for("check_my_horoscope", "{}"), text("ok"))
    await collect(streamed.run("ghar"))

    assert json.loads(next(m for m in client.seen[1] if m.role == "tool").content)[
        "completed"
    ] is False


async def test_the_tool_loop_is_bounded_and_the_user_is_told() -> None:
    """A model that keeps asking must not keep being asked. Silence reads as a
    hang and answering anyway would be a confident reply built on a check that
    never ran, so the turn says which of the two happened."""
    asking = asks_for("find_schemes", json.dumps({"query": "ghar"}))
    streamed, client = turn(asking, list(asking), list(asking))

    assert await collect(streamed.run("ghar")) == COULD_NOT_CHECK
    assert len(client.seen) == 3


async def test_a_reply_needing_no_tools_makes_exactly_one_request() -> None:
    streamed, client = turn(text("Haan."))
    await collect(streamed.run("namaste"))

    assert len(client.seen) == 1


async def test_the_question_is_the_only_user_message() -> None:
    """Guards the prompt assembly. A turn that appended the question twice, or
    dropped the system prompt, would still stream a plausible reply."""
    streamed, client = turn(text("ok"))
    await collect(streamed.run("kya main eligible hoon"))

    roles = [message.role for message in client.seen[0]]

    assert roles == ["system", "user"]
    assert client.seen[0][1].content == "kya main eligible hoon"


async def test_text_emitted_before_a_tool_call_is_not_dropped() -> None:
    """Some models narrate before calling a tool. Swallowing it would lose the
    filler that covers the tool round trip, which is exactly the dead air the
    ablation is trying to remove."""
    streamed, _ = turn(
        [TextChunk(text="Ek minute. ")] + asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("PM Awas Yojana."),
    )

    assert await collect(streamed.run("ghar")) == "Ek minute. PM Awas Yojana."


async def test_a_scripted_stream_that_runs_out_is_an_error_not_a_pass() -> None:
    """Guards every test above. If the turn stopped looping early, a script with
    rounds left over would go unnoticed and the assertions would still hold."""
    streamed, _ = turn(asks_for("find_schemes", json.dumps({"query": "ghar"})))

    with pytest.raises(AssertionError):
        await collect(streamed.run("ghar"))
