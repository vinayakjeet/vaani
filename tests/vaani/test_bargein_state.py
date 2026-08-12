from __future__ import annotations

import json

from vaani.llm_turn import StreamedTurn
from vaani.state import State, TurnState

from .test_llm_turn import ScriptedClient, asks_for, text


async def collect(streamed) -> str:
    return "".join([chunk async for chunk in streamed])


class InterruptsMidTurn(ScriptedClient):
    """Barges in while the tool call is in flight, which is the only moment that
    matters. An interruption arriving after the turn finished proves nothing: the
    guard is consulted between the tool call and its result being applied, so the
    test has to land there."""

    def __init__(self, *rounds, on_first_round) -> None:
        super().__init__(*rounds)
        self._on_first_round = on_first_round

    async def stream(self, provider, messages, **kwargs):
        first = not self.seen
        async for event in super().stream(provider, messages, **kwargs):
            yield event
        if first:
            self._on_first_round()


async def test_an_interrupted_turn_discards_its_tool_result() -> None:
    """SPEC S4. The tool was already in flight when the user cut in, so its result
    arrives for a question nobody is waiting on any more. Folding it into the next
    turn produces an answer that is wrong and sounds exactly as confident as one
    that is right.
    """
    state = TurnState()
    generation = state.begin()
    state.to(State.THINKING)

    client = InterruptsMidTurn(
        asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("this reply must never be produced"),
        on_first_round=state.begin,
    )

    turn = StreamedTurn(llm=client, provider="scripted")
    spoken = await collect(turn.run("mujhe ghar chahiye", lambda: state.owns(generation)))

    assert spoken == ""
    assert len(client.seen) == 1
    assert state.interrupted_previous


async def test_a_turn_that_is_still_current_completes_normally() -> None:
    """The other side of the same check. A guard that abandoned every turn would
    pass the test above and break the product."""
    client = ScriptedClient(
        asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("Aapke liye PM Awas Yojana hai."),
    )
    state = TurnState()
    generation = state.begin()

    turn = StreamedTurn(llm=client, provider="scripted")
    spoken = await collect(turn.run("mujhe ghar chahiye", lambda: state.owns(generation)))

    assert spoken == "Aapke liye PM Awas Yojana hai."
    assert len(client.seen) == 2


async def test_a_turn_with_no_guard_is_never_abandoned() -> None:
    """The unstreamed baseline passes no guard, and it must keep behaving as it did
    or the ablation would be measuring the guard rather than the technique."""
    client = ScriptedClient(
        asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("PM Awas Yojana."),
    )
    turn = StreamedTurn(llm=client, provider="scripted")

    assert await collect(turn.run("ghar")) == "PM Awas Yojana."


async def test_the_new_utterance_is_a_fresh_turn_not_an_appended_one() -> None:
    """SPEC S3. The interrupted turn is recorded as interrupted and the next one
    starts clean, so nothing the abandoned turn produced can be attributed to it."""
    state = TurnState()
    first = state.begin()
    state.to(State.THINKING)
    state.to(State.SPEAKING)

    second = state.begin()

    assert second != first
    assert state.interrupted_previous
    assert state.state is State.LISTENING
    assert not state.owns(first)


async def test_interrupting_during_generation_counts_too() -> None:
    """A user can cut in while the model is still writing, not only while it is
    speaking. Only the second is audible, so only the second tends to get tested."""
    state = TurnState()
    state.begin()
    state.to(State.THINKING)

    state.begin()

    assert state.interrupted_previous
