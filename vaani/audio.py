"""Turning raw frames into something a transcription API will accept.

Providers take a file, not a stream of PCM, so the bytes have to be wrapped. The
wrapping is here rather than inside a provider because both providers need it
and because a WAV header written wrong produces audio that is silent, doubled in
speed, or half a second of noise, and all three look like a transcription
failure rather than a header bug.
"""

from __future__ import annotations

import io
import wave

from vaani.protocol import BYTES_PER_SAMPLE, CHANNELS, SAMPLE_RATE


def to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap little-endian PCM16 in a WAV container, in memory.

    In memory rather than a temp file: this runs once per turn on the latency
    path, and a 512MB Render instance with a full disk fails in a way that reads
    as a provider outage.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(BYTES_PER_SAMPLE)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    return buffer.getvalue()


def pcm_duration_ms(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    samples = len(pcm) / (BYTES_PER_SAMPLE * CHANNELS)
    return samples / sample_rate * 1000
