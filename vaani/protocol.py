"""The wire format between the browser and the server.

Kept in one module with no dependency on FastAPI or on the audio pipeline, so
the framing can be tested without a microphone, a socket, or a provider. A test
that needs a real microphone is a test that never runs in CI, and the framing is
where an off-by-one costs you a garbled transcript that looks like a bad model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# 16 kHz mono PCM16. Whisper and Saaras both want 16 kHz, and browsers capture at
# 48 kHz, so the downsample happens in the AudioWorklet rather than on the
# server: sending 48 kHz would triple the bytes over a socket for audio nobody
# consumes at that rate.
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
CHANNELS = 1

# 20ms frames. The VAD in M1 works on 10, 20 or 30ms frames and 20 is the middle
# option; smaller frames mean more socket overhead per second of speech, larger
# ones mean coarser endpoint detection, which is one of the ablation's knobs.
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE * CHANNELS

# A ceiling on one utterance. Without it a stuck client streams until the dyno
# dies, and on a free tier that is the whole service rather than one session.
MAX_UTTERANCE_MS = 30_000
MAX_UTTERANCE_FRAMES = MAX_UTTERANCE_MS // FRAME_MS


class ClientMessage(StrEnum):
    """Control messages the browser sends. Audio itself is binary, not JSON."""

    START = "start"
    STOP = "stop"
    INTERRUPT = "interrupt"


class ServerMessage(StrEnum):
    READY = "ready"
    PARTIAL = "partial"
    TRANSCRIPT = "transcript"
    REPLY = "reply"
    AUDIO_START = "audio_start"
    AUDIO_END = "audio_end"
    ERROR = "error"


class ProtocolError(ValueError):
    """The client sent something the server cannot interpret."""


@dataclass(frozen=True)
class Frame:
    """One 20ms slice of audio, with the turn it belongs to.

    The generation is on every frame in both directions. Coming in it lets the
    server discard audio from a turn the user already abandoned; going out it
    lets the browser drop audio from a turn it already interrupted. Stopping the
    producer is not enough on either side, because it does not unsend what is
    already on the wire.
    """

    generation: int
    pcm: bytes

    def __post_init__(self) -> None:
        if len(self.pcm) != FRAME_BYTES:
            raise ProtocolError(
                f"frame is {len(self.pcm)} bytes, expected {FRAME_BYTES} "
                f"({FRAME_MS}ms of {SAMPLE_RATE}Hz PCM16)"
            )


# Two bytes of generation, little endian, then the samples. A JSON envelope per
# frame would be 50 frames a second of parsing for four bytes of metadata.
GENERATION_BYTES = 2


def encode(frame: Frame) -> bytes:
    return frame.generation.to_bytes(GENERATION_BYTES, "little") + frame.pcm


def decode(payload: bytes) -> Frame:
    if len(payload) < GENERATION_BYTES:
        raise ProtocolError(f"payload of {len(payload)} bytes carries no generation")
    generation = int.from_bytes(payload[:GENERATION_BYTES], "little")
    return Frame(generation=generation, pcm=payload[GENERATION_BYTES:])


def duration_ms(frames: int) -> int:
    return frames * FRAME_MS
