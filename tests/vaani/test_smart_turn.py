"""smart-turn as an arm, and the feature extractor as the part that can be wrong quietly.

The model's accuracy on Hinglish is a measurement against the recorded corpus and is not
claimed here. What is checked here is everything that would make the arm meaningless
while the suite stayed green: the filterbank against OpenAI's published reference, the
feature geometry the graph's input shape fixes, the once-per-turn call site, and the
un-latching that stops a verdict taken mid-pause from deciding the turn.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from vaani.endpoint import Endpointer
from vaani.models import ModelMissing
from vaani.protocol import FRAME_BYTES, SAMPLE_RATE, SAMPLES_PER_FRAME

numpy = pytest.importorskip("numpy")

REFERENCE = Path(__file__).resolve().parent / "data" / "mel_filters_80.npy"


def speechlike(frames: int = 40) -> list[bytes]:
    """Harmonics with an amplitude wobble. Not speech, and not silence either."""
    samples = []
    for index in range(SAMPLES_PER_FRAME * frames):
        t = index / SAMPLE_RATE
        envelope = 0.6 + 0.4 * math.sin(2 * math.pi * 4.0 * t)
        value = sum(
            math.sin(2 * math.pi * hz * t) / (n + 1)
            for n, hz in enumerate((120, 240, 360, 2400))
        )
        samples.append(int(6000 * envelope * value / 2))
    packed = struct.pack(f"<{len(samples)}h", *samples)
    return [packed[start : start + FRAME_BYTES] for start in range(0, len(packed), FRAME_BYTES)]


def test_the_mel_filterbank_matches_openais_published_reference() -> None:
    """The one part of this that is easy to get subtly wrong and impossible to notice.
    A filterbank off by a scale factor produces confident numbers from noise: the model
    still runs, the suite still passes, and the ablation concludes neural endpointing
    does not help."""
    if not REFERENCE.exists():
        pytest.skip(f"reference filterbank not committed at {REFERENCE}")

    from vaani.smart_turn import mel_filters

    assert numpy.allclose(mel_filters(numpy), numpy.load(REFERENCE), atol=1e-6)


def test_the_features_have_the_geometry_the_graph_requires() -> None:
    """80 by 800 is 8 seconds at a 10ms hop, and it is fixed by the exported input shape
    rather than chosen. A frame count off by one is a shape error at best and a silent
    misalignment at worst."""
    from vaani.smart_turn import FRAMES, MAX_SAMPLES, N_MELS, log_mel, mel_filters

    audio = numpy.zeros(MAX_SAMPLES)
    window = numpy.hanning(400 + 1)[:-1]
    features = log_mel(numpy, audio, mel_filters(numpy), window)

    assert features.shape == (N_MELS, FRAMES)
    assert features.dtype == numpy.float32


def test_whispers_two_normalisation_steps_are_applied() -> None:
    """The clamp to eight decades below the maximum and the affine rescale are the model's
    input distribution, not cosmetics. Dropping either leaves features the graph consumes
    happily and reasons about wrongly."""
    from vaani.smart_turn import MAX_SAMPLES, log_mel, mel_filters

    window = numpy.hanning(400 + 1)[:-1]
    filters = mel_filters(numpy)

    # The clamp is a ceiling on dynamic range rather than a fixed span, so it binds only
    # when the audio exceeds eight decades. A pure tone does: one band carries everything
    # and the rest are numerically empty.
    index = numpy.arange(MAX_SAMPLES)
    tone = numpy.sin(2 * numpy.pi * 440.0 * index / SAMPLE_RATE)
    clamped = log_mel(numpy, tone, filters, window)
    assert clamped.max() - clamped.min() == pytest.approx(2.0, abs=1e-5)

    # And it is never exceeded, whatever the input.
    rng = numpy.random.default_rng(0)
    noisy = log_mel(numpy, rng.normal(0, 0.1, MAX_SAMPLES), filters, window)
    assert noisy.max() - noisy.min() <= 2.0 + 1e-5


@pytest.fixture
def smart_turn():
    from vaani.smart_turn import SmartTurn

    try:
        return SmartTurn()
    except ModelMissing as exc:
        pytest.skip(f"optional arm not available: {exc}")


def test_the_model_returns_a_probability(smart_turn) -> None:
    probability = smart_turn.accept(b"\x00" * FRAME_BYTES * 10)

    assert 0.0 <= probability <= 1.0


def test_it_is_asked_once_per_turn_rather_than_once_per_frame(smart_turn) -> None:
    """At up to 100ms an inference, fifty times a second is not an option. The one moment
    the answer changes anything is when trailing silence reaches the short timeout."""
    calls: list[int] = []

    def counting(audio: bytes) -> bool:
        calls.append(len(audio))
        return True

    endpointer = Endpointer(semantic=True, completion=counting)
    for frame in speechlike():
        endpointer.accept(frame)
    for _ in range(40):
        endpointer.accept(b"\x00" * FRAME_BYTES)

    assert len(calls) == 1


def test_a_verdict_taken_mid_pause_does_not_latch() -> None:
    """"mujhe ghar chahiye" is finished and "mujhe ghar chahiye lekin" is not. A verdict
    taken during the pause before "lekin" would cut the caller off exactly as they
    qualified what they said, which is the failure the rule's non-latching flag avoids."""
    verdicts = iter([True, False])

    def alternating(_audio: bytes) -> bool:
        return next(verdicts)

    endpointer = Endpointer(semantic=True, completion=alternating, early_silence_ms=100)

    for frame in speechlike(20):
        endpointer.accept(frame)
    # A pause long enough to be asked, but shorter than the full timeout.
    for _ in range(6):
        endpointer.accept(b"\x00" * FRAME_BYTES)
    assert endpointer.silence_needed_ms == endpointer.early_silence_ms

    # They carry on. The first verdict must not still be deciding the turn.
    for frame in speechlike(20):
        endpointer.accept(frame)
    for _ in range(6):
        endpointer.accept(b"\x00" * FRAME_BYTES)

    assert endpointer.silence_needed_ms == endpointer.trailing_silence_ms


def test_the_audio_buffer_is_capped_at_what_the_model_reads() -> None:
    """An uncapped buffer on a 512MB instance is a way to lose the whole service to
    somebody who leaves a microphone open."""
    from vaani.endpoint import COMPLETION_AUDIO_BYTES

    endpointer = Endpointer(semantic=True, completion=lambda _audio: False)
    for _ in range(COMPLETION_AUDIO_BYTES // FRAME_BYTES + 100):
        endpointer.accept(b"\x00" * FRAME_BYTES)

    assert len(endpointer._heard) <= COMPLETION_AUDIO_BYTES
