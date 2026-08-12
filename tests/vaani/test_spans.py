from __future__ import annotations

import pytest
from opentelemetry.trace import StatusCode

from vaani.spans import (
    CONTRACT,
    FORBIDDEN_SUBSTRINGS,
    LLM_GENERATE,
    TTS_SYNTHESIZE,
    UndeclaredAttribute,
    stage_span,
)

# SPEC's integration contract, copied out by hand rather than imported.
#
# Asserting `CONTRACT` against itself proves only that a dict equals itself, which
# is the frozen-baseline failure in miniature: renaming an attribute in the module
# would rename it here too and nothing would go red. This list is the second
# opinion, and it is the one that has to be reconciled with SPEC by a human.
SPEC_TABLE = {
    "turn": {"vaani.turn.index", "vaani.turn.interrupted"},
    "vad.endpoint": {
        "vaani.vad.speech_ms",
        "vaani.vad.trailing_silence_ms",
        "vaani.vad.aggressiveness",
    },
    "stt.stream": {"vaani.stt.partials", "vaani.stt.final_chars", "vaani.stt.streaming"},
    "stt.request": {"vaani.stt.index", "vaani.stt.audio_ms"},
    "llm.generate": {"vaani.llm.rounds", "vaani.llm.first_token_ms"},
    "tts.synthesize": {
        "vaani.tts.voice",
        "vaani.tts.sentence_index",
        "vaani.tts.chars",
        "vaani.tts.chunks",
        "vaani.tts.first_chunk_ms",
    },
    "playback.first_audio": {"vaani.playback.queued_ms", "vaani.playback.reported"},
}


def test_the_contract_matches_the_spec_table_in_both_directions() -> None:
    """Neither side may gain or lose a name without the other being updated."""
    assert set(CONTRACT) == set(SPEC_TABLE)
    for name, attributes in SPEC_TABLE.items():
        assert set(CONTRACT[name]) == attributes, name


def test_every_stage_declares_at_least_one_attribute() -> None:
    """A stage with no attributes is a duration with no context, and it is usually
    a stage somebody meant to finish instrumenting."""
    for name, attributes in CONTRACT.items():
        assert attributes, name


def test_attribute_names_are_namespaced() -> None:
    """An un-namespaced attribute collides with a provider's or the SDK's, and the
    winner is whichever wrote last."""
    for attributes in CONTRACT.values():
        for attribute in attributes:
            assert attribute.startswith("vaani.")


def test_no_attribute_can_hold_speech_or_text() -> None:
    """The threat model, as a test rather than a promise.

    Audio and everything derived from it stays out of exported spans, so no
    attribute may be a transcript, a reply, or a tool argument. Counts and
    durations say what an operator needs and cannot be read back. The previous
    project found its own leak only because a canary swept every field, and the
    field that explained the failure was the one redaction existed to withhold.
    """
    for name, attributes in CONTRACT.items():
        for attribute in attributes:
            leaf = attribute.rsplit(".", 1)[-1]
            for forbidden in FORBIDDEN_SUBSTRINGS:
                assert forbidden not in leaf, f"{name}.{attribute}"


def test_a_declared_attribute_is_accepted() -> None:
    with stage_span(LLM_GENERATE, **{"vaani.llm.rounds": 2}):
        pass


def test_an_undeclared_attribute_is_refused() -> None:
    """Dropping it silently would give a green test, a working service, and an empty
    dashboard panel nobody can explain."""
    with pytest.raises(UndeclaredAttribute), stage_span(LLM_GENERATE, **{"vaani.llm.roundz": 2}):
        pass


def test_an_attribute_from_another_stage_is_refused() -> None:
    """Stage names are similar enough that a copied line lands on the wrong span,
    and a `tts` attribute on the model span silently breaks both panels."""
    with pytest.raises(UndeclaredAttribute), stage_span(LLM_GENERATE, **{"vaani.tts.voice": "x"}):
        pass


def test_an_undeclared_stage_is_refused() -> None:
    with pytest.raises(UndeclaredAttribute), stage_span("llm.generat"):
        pass


def test_an_undeclared_attribute_recorded_late_is_refused() -> None:
    """The hole worth closing. Most stage attributes are only known at the end, so
    checking only the ones passed at entry leaves the late ones unchecked, and the
    late ones are where a hurried change lands."""
    with pytest.raises(UndeclaredAttribute), stage_span(LLM_GENERATE) as stage:
        stage.record(**{"vaani.llm.roundz": 1})


CANARY = "mera-naam-canary-hai"


def test_a_recording_span_actually_carries_its_attributes(exported) -> None:
    """Guards every test below, which are all about what a recorded span contains.
    If the fixture were not recording they would all pass against anything."""
    with stage_span(TTS_SYNTHESIZE, **{"vaani.tts.chunks": 3}):
        pass

    span = exported.get_finished_spans()[0]

    assert span.name == TTS_SYNTHESIZE
    assert span.attributes["vaani.tts.chunks"] == 3


def test_a_failing_stage_records_the_class_and_not_the_message(exported) -> None:
    """The exception body can quote a provider's response, which can quote the
    transcript back. The class name is safe; nothing else about it is."""
    with pytest.raises(ValueError), stage_span(TTS_SYNTHESIZE):
        raise ValueError(CANARY)

    span = exported.get_finished_spans()[0]

    assert span.attributes["error.type"] == "ValueError"
    assert span.status.status_code is StatusCode.ERROR


def test_nothing_about_a_failure_leaks_into_the_span(exported) -> None:
    """A canary swept across every field, which is the only method that works here.

    `start_as_current_span` defaults `record_exception=True`, attaching
    `exception.message` and a full stack trace as an event, and
    `set_status_on_exception=True` writes the message into the status description.
    The previous project's error contract was correct and had a passing test proving
    an email address never reached the attributes. The message was in the events,
    and an allowlist of attributes known to be safe can only catch predicted leaks.
    """
    with pytest.raises(ValueError), stage_span(TTS_SYNTHESIZE):
        raise ValueError(CANARY)

    span = exported.get_finished_spans()[0]
    haystack = [
        str(span.name),
        str(span.status.description),
        *(f"{key}={value}" for key, value in (span.attributes or {}).items()),
        *(str(event.name) + str(dict(event.attributes or {})) for event in span.events),
    ]

    assert span.events == ()
    for field in haystack:
        assert CANARY not in field, field
