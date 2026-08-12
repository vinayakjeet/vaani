from __future__ import annotations

import array
import math

import pytest

from vaani.endpoint import AGGRESSIVENESS, Endpointer, rms
from vaani.protocol import FRAME_MS, SAMPLES_PER_FRAME


def tone(amplitude: int) -> bytes:
    """One frame of a sine at a given amplitude, as PCM16."""
    samples = array.array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * 220 * n / SAMPLES_PER_FRAME))
            for n in range(SAMPLES_PER_FRAME)
        ),
    )
    return samples.tobytes()


SILENCE = b"\x00" * SAMPLES_PER_FRAME * 2
SPEECH = tone(8000)
ROOM_NOISE = tone(200)


def feed(endpointer: Endpointer, frame: bytes, ms: int) -> bool:
    ended = False
    for _ in range(ms // FRAME_MS):
        ended = endpointer.accept(frame) or ended
    return ended


def test_silence_alone_never_ends_a_turn() -> None:
    """A user who takes three seconds to start speaking must not have the turn
    end before it began. Leading silence is discarded, not counted."""
    endpointer = Endpointer()

    assert not feed(endpointer, SILENCE, 3000)
    assert not endpointer.started


def test_speech_then_silence_ends_the_turn() -> None:
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 600)

    assert feed(endpointer, SILENCE, endpointer.trailing_silence_ms)


def test_a_pause_shorter_than_the_timeout_does_not_end_the_turn() -> None:
    """The failure a user actually notices: being cut off mid-sentence because
    they drew breath. The trailing-silence timeout exists for this and it is one
    of the five knobs the ablation measures."""
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 600)

    assert not feed(endpointer, SILENCE, endpointer.trailing_silence_ms - FRAME_MS)
    assert not feed(endpointer, SPEECH, 200)


def test_a_cough_does_not_start_a_turn() -> None:
    """Short noise below the minimum speech duration is not a turn. Without this
    every door slam opens a turn that ends in silence and transcribes to
    nothing, and the user gets an answer to a question they did not ask."""
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 100)

    assert not endpointer.started
    assert not feed(endpointer, SILENCE, 2000)


def test_room_noise_below_the_threshold_is_not_speech() -> None:
    endpointer = Endpointer()

    assert not feed(endpointer, ROOM_NOISE, 2000)
    assert not endpointer.started


def test_reset_returns_it_to_a_fresh_turn() -> None:
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 600)
    endpointer.reset()

    assert not endpointer.started
    assert endpointer.speech_ms == 0


def test_one_stray_frame_does_not_reset_the_endpoint_timer() -> None:
    """The failure that keeps a turn open forever.

    A keyboard press or a door during the trailing pause used to clear the timer
    outright, so the user was put back to waiting by a noise they did not make. In
    a room with a television it meant the turn never ended at all. Sustained speech
    still cancels the endpoint; a single frame does not.
    """
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 600)
    feed(endpointer, SILENCE, endpointer.trailing_silence_ms - FRAME_MS * 2)

    assert not feed(endpointer, SPEECH, FRAME_MS)

    assert feed(endpointer, SILENCE, FRAME_MS * 2)


def test_sustained_speech_still_cancels_a_pending_endpoint() -> None:
    """The other side of the same rule. Drawing breath mid-sentence and carrying on
    must not be read as the end of the turn."""
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 600)
    feed(endpointer, SILENCE, endpointer.trailing_silence_ms - FRAME_MS * 2)
    feed(endpointer, SPEECH, endpointer.min_resume_ms)

    assert endpointer.silence_ms == 0
    assert not feed(endpointer, SILENCE, FRAME_MS * 2)


@pytest.mark.parametrize("level", sorted(AGGRESSIVENESS))
def test_every_aggressiveness_setting_still_ends_a_turn(level: int) -> None:
    """The acceptance criterion, across the whole knob rather than at its default.

    A setting that never endpoints leaves the user talking into a microphone that
    has stopped listening, and the ablation sweeps all four.
    """
    endpointer = Endpointer.at(level)
    feed(endpointer, SPEECH, 600)

    assert feed(endpointer, SILENCE, endpointer.trailing_silence_ms)
    assert endpointer.aggressiveness == level


def test_a_higher_setting_endpoints_sooner() -> None:
    """The trade the ablation is measuring: dead air against cutting people off.
    If the ordering were not monotonic the knob would not be a knob."""
    trailing = [Endpointer.at(level).trailing_silence_ms for level in sorted(AGGRESSIVENESS)]

    assert trailing == sorted(trailing, reverse=True)


@pytest.mark.parametrize("level", sorted(AGGRESSIVENESS))
def test_every_timeout_is_a_whole_number_of_frames(level: int) -> None:
    """Audio arrives in fixed frames, so a timeout of 850ms fires at 860. A
    configured number the system can never hit is one the ablation would report as
    though it had been used."""
    assert Endpointer.at(level).trailing_silence_ms % FRAME_MS == 0


def test_an_unknown_aggressiveness_is_refused() -> None:
    """A sweep that reads its levels from a config file will eventually pass 4, and
    silently clamping it would put an unlabelled row in the results table."""
    with pytest.raises(ValueError):
        Endpointer.at(9)


def test_rms_of_silence_is_zero_and_of_speech_is_not() -> None:
    """Guards every test above, which are all expressed in terms of a threshold
    this function produces. If it returned a constant they would still pass."""
    assert rms(SILENCE) == 0.0
    assert rms(SPEECH) > rms(ROOM_NOISE) > 0.0
