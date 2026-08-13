from __future__ import annotations

import pytest

from vaani.turn_taking import (
    AudioStatus,
    TurnTaking,
    heard_prefix,
    whole_words,
)


def talking() -> TurnTaking:
    turn = TurnTaking()
    turn.note_user_speaking()
    return turn


@pytest.mark.parametrize(
    "backchannel",
    ["haan", "achha", "hmm", "ji", "theek hai", "ok ok", "हाँ", "अच्छा", "ji haan"],
)
def test_a_backchannel_does_not_interrupt(backchannel: str) -> None:
    """The behaviour that makes an agent feel neurotic when it is wrong. Hinglish speakers
    backchannel constantly, and none of these means stop."""
    assert not TurnTaking().should_interrupt(backchannel, agent_speaking=True)


@pytest.mark.parametrize("stop", ["ruko", "nahi", "bas", "wait", "stop", "रुको", "नहीं", "ek minute"])
def test_a_short_stop_word_interrupts_despite_being_short(stop: str) -> None:
    """Why the closed set exists. Word count alone would ignore "ruko", which is the one word
    a person says when they most need to be heard."""
    assert TurnTaking().should_interrupt(stop, agent_speaking=True)


def test_a_content_bearing_utterance_interrupts_on_word_count() -> None:
    """No lexicon involved. Anything long enough is a real turn, which is what makes the
    approach work on words nobody enumerated."""
    assert TurnTaking().should_interrupt("nahi nahi mera income kam hai", agent_speaking=True)


def test_nothing_interrupts_when_the_agent_is_not_speaking() -> None:
    """There is nothing to interrupt, and treating ordinary speech as a barge-in would abandon
    the turn the user is in the middle of starting."""
    assert not TurnTaking().should_interrupt("ruko", agent_speaking=False)


def test_punctuation_and_case_do_not_defeat_the_stop_words() -> None:
    """A recogniser punctuates and capitalises as it pleases, and a lexicon that only matches
    bare lowercase is a lexicon that misses."""
    assert TurnTaking().should_interrupt("Ruko!", agent_speaking=True)
    assert TurnTaking().should_interrupt("  नहीं।  ", agent_speaking=True)


@pytest.mark.parametrize("short", ["haan", "hmm", "achha ji"])
def test_a_short_final_transcript_is_discarded_as_a_backchannel(short: str) -> None:
    """Bolna's `is_false_interruption`. Answering it produces a reply to "hmm"."""
    assert TurnTaking().is_backchannel(short, agent_speaking=True)


def test_a_real_question_is_not_discarded() -> None:
    turn = TurnTaking()

    assert not turn.is_backchannel("kya main eligible hoon", agent_speaking=True)
    assert not turn.is_backchannel("ruko", agent_speaking=True)


def test_audio_from_a_stale_generation_is_blocked() -> None:
    assert TurnTaking().status(1, live={2}) is AudioStatus.BLOCK


def test_audio_is_held_while_the_user_is_speaking() -> None:
    """The state we did not have. The generation is still valid, so this audio is still the
    right answer; it is only the wrong moment to say it."""
    assert talking().status(1, live={1}) is AudioStatus.WAIT


def test_audio_flows_when_nobody_is_speaking() -> None:
    assert TurnTaking().status(1, live={1}) is AudioStatus.SEND


def test_a_grace_period_holds_audio_just_after_an_utterance() -> None:
    """A pause that turns out to be mid-thought must not be talked over."""
    turn = TurnTaking()
    turn.turns = 5
    turn.note_user_speaking()
    turn.note_user_stopped()

    assert turn.status(1, live={1}) is AudioStatus.WAIT


def test_the_grace_period_is_skipped_for_the_first_turns() -> None:
    """A greeting delayed by politeness reads as a slow service."""
    turn = TurnTaking()
    turn.note_user_speaking()
    turn.note_user_stopped()

    assert turn.turns <= 2
    assert turn.status(1, live={1}) is AudioStatus.SEND


def test_more_than_one_generation_can_be_live() -> None:
    """A set rather than one integer, so background audio and a reply can both be valid.
    Our single counter could not express that."""
    turn = TurnTaking()

    assert turn.status(-1, live={-1, 7}) is AudioStatus.SEND
    assert turn.status(7, live={-1, 7}) is AudioStatus.SEND


def test_playout_advances_by_duration_not_by_send_time() -> None:
    """Audio is handed over faster than real time, so a chunk starts when the previous one
    ends. Measuring the send and calling it playback is what our span did."""
    turn = TurnTaking()
    turn.note_audio_queued(1.0)
    turn.note_audio_queued(1.0)

    assert turn.playing_ms_remaining() > 1500


def test_dropping_queued_audio_ends_the_playout_estimate() -> None:
    turn = TurnTaking()
    turn.note_audio_queued(5.0)
    turn.drop_playout()

    assert turn.playing_ms_remaining() == 0


def test_a_zero_length_chunk_does_not_move_the_estimate() -> None:
    turn = TurnTaking()
    turn.note_audio_queued(0)

    assert turn.playing_ms_remaining() == 0


def test_the_heard_prefix_is_what_was_actually_played() -> None:
    """Without this the model believes it said things nobody heard, and refers back to them
    two turns later."""
    spoken = "Aap eligible hain. Aapko 6000 milega."

    assert heard_prefix(spoken, "Aap eligible hain.") == "Aap eligible hain."


def test_heard_text_that_cannot_be_located_is_treated_as_unheard() -> None:
    """The branch worth copying. Stale or mismatched turn data must not splice foreign text
    into the conversation record, which would be worse than losing it."""
    assert heard_prefix("Aap eligible hain.", "something else entirely") == ""


def test_a_trailing_space_mismatch_still_finds_a_prefix() -> None:
    """Synthesisers add and drop whitespace, so an exact match cannot be the only path."""
    assert heard_prefix("Aapko 6000 milega.", "Aapko 6000 ") == "Aapko 6000"


def test_nothing_heard_leaves_nothing_in_the_record() -> None:
    assert heard_prefix("Aap eligible hain.", "") == ""
    assert heard_prefix("Aap eligible hain.", "   ") == ""


def test_a_partial_is_trimmed_to_its_last_whole_word() -> None:
    """A fragment in Hinglish is frequently a different word, so half of one must not enter
    the record."""
    assert whole_words("mera income pacha") == "mera income"


def test_a_single_partial_word_leaves_nothing() -> None:
    assert whole_words("pacha") == ""
    assert whole_words("   ") == ""
