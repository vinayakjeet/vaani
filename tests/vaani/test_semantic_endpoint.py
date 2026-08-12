from __future__ import annotations

import pytest

from vaani.completeness import looks_complete
from vaani.endpoint import DEFAULT_EARLY_SILENCE_MS, Endpointer
from vaani.protocol import FRAME_MS

from .test_endpoint import SILENCE, SPEECH, feed

COMPLETE = [
    "kya main eligible hoon",
    "mujhe ghar chahiye",
    "aapki aay kitni hai",
    "मुझे कौन सी योजना मिलेगी",
    "क्या मैं पात्र हूँ",
    "mujhe pension milegi kya hai",
]

INCOMPLETE = [
    "mera income",
    "meri aay",
    "mujhe ghar chahiye lekin",
    "main apply karna chahta hoon aur",
    "meri aay pachaas",
    "मेरी आय",
    "मुझे घर चाहिए लेकिन",
    "haan",
    "kya",
]


@pytest.mark.parametrize("partial", COMPLETE)
def test_a_finished_question_is_recognised(partial: str) -> None:
    assert looks_complete(partial)


@pytest.mark.parametrize("partial", INCOMPLETE)
def test_an_unfinished_utterance_is_not(partial: str) -> None:
    """The acceptance criterion's other half, and the one that protects users.

    A trailing "mera income" is somebody drawing breath before the number. Reading
    it as finished cuts them off, and the number is exactly the part that decides
    the answer.
    """
    assert not looks_complete(partial)


def test_a_terminator_settles_it_whatever_the_last_word_is() -> None:
    """What catches a Hinglish question ending on an English word. A recogniser
    that heard question intonation usually punctuates it, and the word order rule
    has nothing to say about "documents"."""
    assert looks_complete("mujhe kaun se documents chahiye?")
    assert looks_complete("iske liye kya process hai.")


def test_an_english_final_question_without_a_terminator_waits() -> None:
    """Deliberate, and the conservative direction. "main eligible" could be the whole
    question or the first half of "main eligible hoon ya nahi", and nothing local to
    the words says which. Waiting costs dead air the full timeout would have spent
    anyway; guessing wrong talks over somebody."""
    assert not looks_complete("main eligible")
    assert not looks_complete("mera pincode")


def test_a_trailing_number_is_never_complete() -> None:
    """Amounts arrive in pieces. "pachaas" becomes "pachaas hazaar", and an income
    cut in half is a wrong eligibility answer delivered confidently."""
    assert not looks_complete("meri aay 50000")
    assert not looks_complete("mera pincode 8000")


def test_a_punctuated_number_is_still_not_complete() -> None:
    """Where the number rule earns itself, and it did not until this test existed.

    A recogniser punctuates after a number as a matter of course, so
    "meri aay 50000." arrives looking finished while the speaker is still saying
    "hazaar rupaye". Without checking the number before the terminator the rule was
    dead code: deleting it changed no test, because a digit is never a final marker
    and the shortcut had already returned.
    """
    assert not looks_complete("meri aay 50000.")
    assert not looks_complete("mera pincode 8000?")
    assert not looks_complete("मेरी आय 50000।")


def test_completeness_is_not_latched() -> None:
    """A partial can go back to unfinished as more words arrive, and it must."""
    endpointer = Endpointer(semantic=True)

    endpointer.note_partial("mujhe ghar chahiye")
    assert endpointer.silence_needed_ms == DEFAULT_EARLY_SILENCE_MS

    endpointer.note_partial("mujhe ghar chahiye lekin")
    assert endpointer.silence_needed_ms == endpointer.trailing_silence_ms


def test_a_complete_partial_endpoints_early() -> None:
    endpointer = Endpointer(semantic=True)
    feed(endpointer, SPEECH, 600)
    endpointer.note_partial("kya main eligible hoon")

    assert feed(endpointer, SILENCE, DEFAULT_EARLY_SILENCE_MS)


def test_an_incomplete_partial_holds_the_full_timeout() -> None:
    endpointer = Endpointer(semantic=True)
    feed(endpointer, SPEECH, 600)
    endpointer.note_partial("mera income")

    assert not feed(endpointer, SILENCE, DEFAULT_EARLY_SILENCE_MS)
    assert feed(
        endpointer, SILENCE, endpointer.trailing_silence_ms - DEFAULT_EARLY_SILENCE_MS
    )


def test_the_baseline_ignores_partials_entirely() -> None:
    """`semantic` is off by default on purpose. The unstreamed baseline has no
    partial to read, and if it quietly used this it would be measured against
    itself and the technique would appear to buy nothing."""
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 600)
    endpointer.note_partial("kya main eligible hoon")

    assert endpointer.silence_needed_ms == endpointer.trailing_silence_ms
    assert not feed(endpointer, SILENCE, DEFAULT_EARLY_SILENCE_MS)


def test_the_early_wait_never_exceeds_the_configured_one() -> None:
    """At the most aggressive setting the full timeout is already short, and an
    early wait longer than it would make the optimisation a pessimisation."""
    endpointer = Endpointer.at(3, semantic=True, early_silence_ms=5000)
    endpointer.note_partial("kya main eligible hoon")

    assert endpointer.silence_needed_ms == endpointer.trailing_silence_ms


def test_reset_clears_the_completeness_flag() -> None:
    """Otherwise the next turn inherits the last one's verdict and endpoints early
    on its first pause."""
    endpointer = Endpointer(semantic=True)
    endpointer.note_partial("kya main eligible hoon")
    endpointer.reset()

    assert endpointer.silence_needed_ms == endpointer.trailing_silence_ms


def test_an_empty_partial_is_not_complete() -> None:
    assert not looks_complete("")
    assert not looks_complete("   ")


def test_a_bare_terminator_is_not_complete() -> None:
    """A recogniser emitting punctuation before any words would otherwise endpoint
    the turn on nothing."""
    assert not looks_complete("?")
    assert not looks_complete("।")


def test_the_early_wait_is_a_whole_number_of_frames() -> None:
    assert DEFAULT_EARLY_SILENCE_MS % FRAME_MS == 0
