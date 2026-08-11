from __future__ import annotations

import array
import math

from vaani.endpoint import Endpointer, rms
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


def test_rms_of_silence_is_zero_and_of_speech_is_not() -> None:
    """Guards every test above, which are all expressed in terms of a threshold
    this function produces. If it returned a constant they would still pass."""
    assert rms(SILENCE) == 0.0
    assert rms(SPEECH) > rms(ROOM_NOISE) > 0.0
