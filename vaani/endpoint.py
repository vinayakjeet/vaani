"""Decide when the user has stopped talking.

Deliberately the naive version: frame energy against a threshold, with a
trailing-silence timer. M1.1 replaces it with a real VAD, and this stays as the
baseline the ablation measures against, because "what did VAD aggressiveness
buy" is unanswerable without a before.

Energy alone cannot tell speech from a fan, a keyboard, or a television. That is
a known limit rather than a bug to fix here, and it is why the M1 replacement is
a separate task instead of a tweak to this one.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass, field

from vaani.protocol import FRAME_MS

# Roughly the level of a quiet room on a laptop microphone, in RMS over PCM16.
# Provisional, and it is not allowed to stay that way: M1.1 sets it from the
# recorded corpus rather than from this guess, the same way every threshold in
# the previous project had to be measured before it shipped.
DEFAULT_THRESHOLD_RMS = 500.0

# How much quiet ends a turn. Long enough to survive the pause in the middle of
# a sentence, short enough that the user is not left waiting. One of the five
# knobs in the ablation.
DEFAULT_TRAILING_SILENCE_MS = 700

# Speech shorter than this is a cough, a door, or a lip smack. Without it every
# stray noise starts a turn that then ends in silence and transcribes to nothing.
DEFAULT_MIN_SPEECH_MS = 200


def rms(pcm: bytes) -> float:
    """Root mean square of one frame of little-endian PCM16."""
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm)
    return math.sqrt(sum(s * s for s in samples) / len(samples))


@dataclass
class Endpointer:
    threshold_rms: float = DEFAULT_THRESHOLD_RMS
    trailing_silence_ms: int = DEFAULT_TRAILING_SILENCE_MS
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS

    speech_ms: int = field(default=0, init=False)
    silence_ms: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)

    @property
    def started(self) -> bool:
        """Whether enough speech has arrived to call this a turn."""
        return self._started

    def accept(self, pcm: bytes) -> bool:
        """Feed one frame. Returns True when the turn has ended.

        Silence before speech is discarded rather than counted, so a user who
        takes three seconds to begin does not end the turn before starting it.
        """
        if rms(pcm) >= self.threshold_rms:
            self.speech_ms += FRAME_MS
            self.silence_ms = 0
            if self.speech_ms >= self.min_speech_ms:
                self._started = True
            return False

        if not self._started:
            return False

        self.silence_ms += FRAME_MS
        return self.silence_ms >= self.trailing_silence_ms

    def reset(self) -> None:
        self.speech_ms = 0
        self.silence_ms = 0
        self._started = False
