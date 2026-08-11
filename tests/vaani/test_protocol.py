from __future__ import annotations

import pytest

from vaani.protocol import (
    FRAME_BYTES,
    FRAME_MS,
    SAMPLE_RATE,
    Frame,
    ProtocolError,
    decode,
    duration_ms,
    encode,
)

SILENCE = b"\x00" * FRAME_BYTES


def test_a_frame_is_twenty_milliseconds_of_sixteen_kilohertz_audio() -> None:
    """The constants have to agree with each other or every duration derived
    from a frame count is wrong, and a wrong duration reads as a slow model."""
    assert FRAME_BYTES == SAMPLE_RATE * FRAME_MS // 1000 * 2
    assert duration_ms(50) == 1000


def test_a_frame_round_trips() -> None:
    frame = Frame(generation=7, pcm=SILENCE)

    assert decode(encode(frame)) == frame


def test_a_short_frame_is_refused() -> None:
    """Browsers deliver audio in 128-sample blocks that do not divide evenly into
    a 20ms frame, so a client that forgets to buffer sends ragged ones. Accepting
    them means audio that drifts out of alignment and transcribes to nonsense,
    which looks like a bad model rather than a framing bug."""
    with pytest.raises(ProtocolError):
        Frame(generation=1, pcm=SILENCE[:-2])


def test_a_long_frame_is_refused() -> None:
    with pytest.raises(ProtocolError):
        Frame(generation=1, pcm=SILENCE + b"\x00\x00")


def test_a_payload_with_no_generation_is_refused() -> None:
    with pytest.raises(ProtocolError):
        decode(b"\x00")


def test_the_generation_survives_the_wire() -> None:
    """The whole point of putting it on every frame: a server has to be able to
    tell audio from the turn the user abandoned from audio from the new one."""
    assert decode(encode(Frame(generation=513, pcm=SILENCE))).generation == 513
