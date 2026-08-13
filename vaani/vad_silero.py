"""Silero VAD as the answer to "is this frame speech", replacing frame energy.

The energy detector cannot tell speech from a fan, and its threshold is a guess: 500
RMS, written down as roughly a quiet room and never measured. That guess is the most
likely reason a real microphone in a real room either never starts a turn or never
ends one, and it is the one threshold in the pipeline whose failure looks like the
whole demo being broken.

Silero is the default in both LiveKit and Pipecat, 2MB of ONNX, and under a
millisecond per chunk on one CPU thread. It stays an arm rather than a replacement:
`Endpointer` still runs energy by default, the ablation measures both, and their
false-endpoint rates are reported together. A model adopted without measuring it
against what it replaced is an upgrade nobody can defend.

**The window size is the integration detail that silently breaks this.** Silero v5
takes exactly 512 samples at 16kHz, 32ms, and our transport carries 20ms frames of 320
samples. Feeding it a 320-sample frame does not raise: it returns a number, and the
number is wrong. So frames are buffered into 512-sample windows and the last verdict
stands until the next window closes, which costs at most one frame of lag.
"""

from __future__ import annotations

import structlog

from vaani.models import SILERO_VAD, session
from vaani.protocol import SAMPLE_RATE

logger = structlog.get_logger(__name__)

# Exactly what the v5 graph expects at 16kHz. Not a tuning knob: a different length
# produces a confident number from a model that was handed the wrong shape of input.
WINDOW_SAMPLES = 512

# The authors' recommended speech threshold, and the one LiveKit and Pipecat ship. Worth
# distinguishing from our own provisional numbers: this one is a default published by the
# people who trained the model, so it starts as evidence rather than as a guess. It is
# still swept in the ablation, because a threshold that suits their evaluation set is not
# automatically right for Hinglish on a laptop microphone.
DEFAULT_SPEECH_PROBABILITY = 0.5

# The state tensor's shape, per the graph: two layers, one batch, 128 hidden.
_STATE_SHAPE = (2, 1, 128)

_PCM16_FULL_SCALE = 32768.0


class SileroVad:
    """Frame-by-frame speech detection, with the recurrent state carried across calls."""

    name = "silero"

    def __init__(self, threshold: float = DEFAULT_SPEECH_PROBABILITY) -> None:
        import numpy

        self._numpy = numpy
        self._session = session(SILERO_VAD)
        self._threshold = threshold
        self._pending = numpy.zeros(0, dtype=numpy.float32)
        self._state = numpy.zeros(_STATE_SHAPE, dtype=numpy.float32)
        # No window has closed yet, so there is no verdict. False rather than True: a
        # detector that reports speech before it has heard any would start a turn on the
        # first frame of every session.
        self.probability = 0.0

    def __call__(self, pcm: bytes) -> bool:
        """Whether this frame is speech, in the shape `Endpointer` asks for."""
        return self.accept(pcm) >= self._threshold

    def accept(self, pcm: bytes) -> float:
        """Feed one frame and return the current speech probability.

        The probability is the model's, and a frame that does not complete a window
        returns the previous one rather than zero. Returning zero on the frames between
        windows would make speech look like it stutters at 50Hz, and the trailing-silence
        timer would then never run to completion.
        """
        numpy = self._numpy
        samples = (
            numpy.frombuffer(pcm, dtype="<i2").astype(numpy.float32) / _PCM16_FULL_SCALE
        )
        self._pending = numpy.concatenate((self._pending, samples))

        while len(self._pending) >= WINDOW_SAMPLES:
            window = self._pending[:WINDOW_SAMPLES]
            self._pending = self._pending[WINDOW_SAMPLES:]
            output, self._state = self._session.run(
                None,
                {
                    "input": window.reshape(1, WINDOW_SAMPLES),
                    "state": self._state,
                    "sr": numpy.array(SAMPLE_RATE, dtype="int64"),
                },
            )
            self.probability = float(output[0][0])

        return self.probability

    def reset(self) -> None:
        """Forget the utterance. The recurrent state is per conversation turn.

        Carrying it across turns is not obviously wrong and is not obviously right, and
        an unmeasured choice in a detector is exactly what this project keeps paying for,
        so the state is cleared and the decision is written down here: a turn is the unit
        everything else in this pipeline resets at.
        """
        numpy = self._numpy
        self._pending = numpy.zeros(0, dtype=numpy.float32)
        self._state = numpy.zeros(_STATE_SHAPE, dtype=numpy.float32)
        self.probability = 0.0
