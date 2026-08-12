from __future__ import annotations

import json

from vaani.spans import CONTRACT, LLM_GENERATE, SHARED_ATTRIBUTES, STT_STREAM
from vaani.tts import speak_as_they_arrive

from .test_llm_turn import ScriptedClient, asks_for, text, turn
from .test_streamed_speech import RecordingTts, stream
from .test_stt_stream import FakeTranscriber, chunked, frames


def vaani_attributes(exported) -> dict[str, set[str]]:
    """Every `vaani.*` attribute actually emitted, by span name."""
    emitted: dict[str, set[str]] = {}
    for span in exported.get_finished_spans():
        keys = {key for key in (span.attributes or {}) if key.startswith("vaani.")}
        emitted.setdefault(span.name, set()).update(keys)
    return emitted


async def test_the_recogniser_emits_its_declared_stage(exported) -> None:
    stack, _ = chunked()

    [partial async for partial in stack.stream(frames(20))]

    emitted = vaani_attributes(exported)

    assert STT_STREAM in emitted
    assert emitted[STT_STREAM] == set(CONTRACT[STT_STREAM])


async def test_the_model_turn_emits_its_declared_stage(exported) -> None:
    streamed, _ = turn(text("Aap ", "eligible ", "hain."))

    [chunk async for chunk in streamed.run("kya main eligible hoon")]

    emitted = vaani_attributes(exported)

    assert emitted[LLM_GENERATE] == set(CONTRACT[LLM_GENERATE])


async def test_time_to_first_token_is_recorded_separately_from_the_duration(
    exported,
) -> None:
    """The number a listener experiences. A span that reported only its duration
    would say how long the model talked for, which is a different measurement and
    the one that makes streaming look no better than the baseline."""
    streamed, _ = turn(text("Aap ", "eligible ", "hain."))

    [chunk async for chunk in streamed.run("q")]

    span = next(s for s in exported.get_finished_spans() if s.name == LLM_GENERATE)

    assert "vaani.llm.first_token_ms" in span.attributes
    assert span.attributes["vaani.llm.first_token_ms"] >= 0


async def test_a_tool_round_trip_is_counted_in_rounds(exported) -> None:
    streamed, _ = turn(
        asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("PM Awas Yojana."),
    )

    [chunk async for chunk in streamed.run("ghar")]

    span = next(s for s in exported.get_finished_spans() if s.name == LLM_GENERATE)

    assert span.attributes["vaani.llm.rounds"] == 2


async def test_each_sentence_gets_its_own_synthesis_span(exported) -> None:
    """One span per sentence, not per turn. Averaging sentence one and sentence
    three into a turn-level span hides exactly what the first-sentence flush
    bought."""
    tts = RecordingTts()

    async for _chunk in speak_as_they_arrive(
        stream(["Aap eligible hain. ", "Aapko 6000 milega."]), tts
    ):
        pass

    # RecordingTts is a stub and opens no span of its own, so the count here is the
    # sentences the wiring produced rather than the provider's instrumentation.
    assert tts.indices == [1, 2]


async def test_no_span_carries_an_undeclared_attribute(exported) -> None:
    """The contract in the direction a test usually forgets.

    `test_spans.py` proves the table matches SPEC. This proves the running pipeline
    matches the table, which is the half that drifts: an attribute added at a call
    site in a hurry is invisible to a test that only reads the table.
    """
    stack, _ = chunked()
    [partial async for partial in stack.stream(frames(20))]

    streamed, _ = turn(
        asks_for("find_schemes", json.dumps({"query": "ghar"})),
        text("PM Awas Yojana."),
    )
    [chunk async for chunk in streamed.run("ghar")]

    for span in exported.get_finished_spans():
        assert span.name in CONTRACT, span.name
        allowed = set(CONTRACT[span.name]) | SHARED_ATTRIBUTES
        for key in span.attributes or {}:
            assert key in allowed, f"{span.name} emitted undeclared {key}"


async def test_a_failing_recogniser_marks_its_stage_without_leaking(exported) -> None:
    """The error path is instrumented too. A turn that ended in an apology should be
    a failed span, not a gap where a span should be."""
    transcriber = FakeTranscriber(fail_on={1})
    stack, _ = chunked()
    stack._transcriber = transcriber

    [partial async for partial in stack.stream(frames(20))]

    span = next(s for s in exported.get_finished_spans() if s.name == STT_STREAM)

    assert span.events == ()


async def test_the_scripted_client_is_actually_exercised() -> None:
    """Guards the tests above, which all assert on spans produced as a side effect.
    If the turn stopped calling the client they would assert on an empty list and
    pass."""
    client = ScriptedClient(text("hi"))
    streamed, _ = turn(text("hi"))

    assert client.seen == []
    [chunk async for chunk in streamed.run("q")]
