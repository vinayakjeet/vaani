"""Silero as an arm against frame energy.

What these can check without a recorded corpus is the direction the two detectors
disagree in, and that turns out to be the whole argument: a loud tone is speech to an
energy threshold and is not speech to a model. A fan, a hum and a television all live
in that gap, and so does the reason a real microphone in a real room either never
starts a turn or never ends one.

What these cannot check is the positive side. Whether Silero fires on Hinglish spoken
into a laptop microphone at a distance is a measurement against the corpus in M4.2, and
until that exists this is an arm that has been wired up rather than one that has been
shown to be better. Saying so here is cheaper than discovering later that a green suite
was agreeing with a guess.
"""

from __future__ import annotations

import math
import struct

import pytest

from vaani.endpoint import DEFAULT_THRESHOLD_RMS, Endpointer, rms
from vaani.models import ModelMissing
from vaani.protocol import FRAME_BYTES, SAMPLE_RATE, SAMPLES_PER_FRAME

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def tone(hz: float, amplitude: int = 8000, frames: int = 40) -> list[bytes]:
    """A steady tone: loud enough to clear the energy threshold, and not speech."""
    samples = []
    for index in range(SAMPLES_PER_FRAME * frames):
        samples.append(int(amplitude * math.sin(2 * math.pi * hz * index / SAMPLE_RATE)))
    packed = struct.pack(f"<{len(samples)}h", *samples)
    return [
        packed[start : start + FRAME_BYTES]
        for start in range(0, len(packed), FRAME_BYTES)
    ]


@pytest.fixture
def silero():
    from vaani.vad_silero import SileroVad

    try:
        return SileroVad()
    except ModelMissing as exc:
        pytest.skip(f"optional arm not available: {exc}")


def test_a_loud_tone_is_speech_to_the_energy_threshold() -> None:
    """The control, and the defect. Without this the next test proves nothing: it has to
    be shown that energy really is fooled before showing that the model is not."""
    frames = tone(200.0)

    assert rms(frames[0]) > DEFAULT_THRESHOLD_RMS

    endpointer = Endpointer()
    for frame in frames:
        endpointer.accept(frame)
    assert endpointer.started


def test_a_loud_tone_is_not_speech_to_silero(silero) -> None:
    """A fan, a hum and a television all live in the gap between these two tests."""
    endpointer = Endpointer(speech=silero)
    for frame in tone(200.0):
        endpointer.accept(frame)

    assert not endpointer.started
    assert silero.probability < 0.5


def test_digital_silence_is_not_speech(silero) -> None:
    endpointer = Endpointer(speech=silero)
    for _ in range(40):
        endpointer.accept(b"\x00" * FRAME_BYTES)

    assert not endpointer.started


def test_the_probability_holds_between_windows_rather_than_dropping_to_zero(silero) -> None:
    """The integration detail that silently breaks this. Silero v5 takes 512 samples and
    the transport carries 320, so two frames in three complete no window. Reporting zero
    on those would make speech stutter at 50Hz and the trailing-silence timer would never
    run to completion."""
    frames = tone(200.0, frames=4)

    first = silero.accept(frames[0])
    # 320 samples in: no window has closed, so this is the initial verdict.
    assert first == silero.probability

    silero.accept(frames[1])
    after_window = silero.probability
    # 640 samples: one window closed, 128 left over.
    assert silero.accept(frames[2]) == after_window


def test_reset_clears_the_recurrent_state(silero) -> None:
    """Carrying hidden state across turns lets the previous utterance decide the first
    frames of the next one, and it is invisible because the numbers stay plausible."""
    for frame in tone(200.0):
        silero.accept(frame)

    silero.reset()

    assert silero.probability == 0.0


def test_the_endpointer_resets_the_detector_it_was_given(silero) -> None:
    """The wiring, not the model. An endpointer that resets itself and not its detector
    is the version of this bug that ships."""
    endpointer = Endpointer(speech=silero)
    for frame in tone(200.0):
        endpointer.accept(frame)

    endpointer.reset()

    assert silero.probability == 0.0


def test_the_microphone_diagnosis_still_works_with_a_model_detector(silero) -> None:
    """A model says "not speech" and cannot tell a muted input from a quiet room. Those
    need different things said to the user, so the level is still measured."""
    endpointer = Endpointer(speech=silero, silence_timeout_ms=100)
    for _ in range(20):
        endpointer.accept(b"\x00" * FRAME_BYTES)

    from vaani.endpoint import MicState

    assert endpointer.diagnose() is MicState.SILENT
