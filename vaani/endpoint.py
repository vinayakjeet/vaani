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

from vaani.completeness import looks_complete
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

# How much sustained speech it takes to cancel a pending endpoint. Without this a
# single loud frame resets the trailing-silence timer, so one keyboard press or
# one clatter during the pause puts the user back to waiting, and in a room with a
# television the turn never ends at all. Three frames is enough to tell a syllable
# from a click without adding audible lag to a genuine resumption.
DEFAULT_MIN_RESUME_MS = 60

# The ablation's VAD knob, as four named settings rather than two loose numbers.
# Higher is more eager to call something silence: it endpoints sooner and risks
# cutting people off, which is the trade SPEC states and the ablation measures.
# The false-endpoint rate is reported beside the milliseconds, never instead.
#
# Provisional, and not allowed to stay that way. Every threshold in the previous
# project had to be measured before it shipped, and these are guesses at the level
# of a quiet room. M4.2's recorded corpus is what sets them, and until it exists
# the numbers here are labelled rather than trusted.
# Trailing values are whole numbers of frames on purpose. Audio arrives in 20ms
# frames, so a timeout of 850ms is reached at 860 and the configured number is one
# the system can never actually hit.
AGGRESSIVENESS: dict[int, tuple[float, int]] = {
    0: (400.0, 1000),
    1: (500.0, 840),
    2: (600.0, 700),
    3: (750.0, 500),
}

DEFAULT_AGGRESSIVENESS = 2

# The endpoint wait once the partial transcript already sounds finished. Short on
# purpose: it is the largest single term in the optimised budget, and it is also the
# one that risks cutting people off, so it is a named number reported beside its
# false-endpoint rate rather than tuned quietly.
DEFAULT_EARLY_SILENCE_MS = 200


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
    min_resume_ms: int = DEFAULT_MIN_RESUME_MS
    early_silence_ms: int = DEFAULT_EARLY_SILENCE_MS
    # Off unless a caller feeds partials. The unstreamed baseline has no partial to
    # read, and it must keep measuring the full timeout or the ablation would be
    # comparing the optimised path against itself.
    semantic: bool = False
    # Which named setting produced this one, when a named setting did. Recorded so
    # a waterfall row can say which knob position it was measured at rather than
    # leaving the reader to infer it from two numbers.
    aggressiveness: int | None = None

    speech_ms: int = field(default=0, init=False)
    silence_ms: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)
    _resume_ms: int = field(default=0, init=False)
    _partial_complete: bool = field(default=False, init=False)

    @classmethod
    def at(cls, aggressiveness: int = DEFAULT_AGGRESSIVENESS, **overrides: object) -> Endpointer:
        """An endpointer at one of the named settings the ablation sweeps."""
        if aggressiveness not in AGGRESSIVENESS:
            raise ValueError(
                f"aggressiveness must be one of {sorted(AGGRESSIVENESS)}, got {aggressiveness}"
            )
        threshold, trailing = AGGRESSIVENESS[aggressiveness]
        return cls(
            threshold_rms=threshold,
            trailing_silence_ms=trailing,
            aggressiveness=aggressiveness,
            **overrides,
        )

    def note_partial(self, partial: str) -> None:
        """Feed the latest partial transcript, so the wait can be shortened.

        A partial can go from complete back to incomplete as more words arrive, and
        the flag follows it rather than latching. "mujhe ghar chahiye" is finished;
        "mujhe ghar chahiye lekin" is not, and latching would cut the caller off
        exactly when they were about to qualify what they said.
        """
        self._partial_complete = looks_complete(partial)

    @property
    def silence_needed_ms(self) -> int:
        """How much quiet ends the turn right now.

        The number moves during a turn, which is the whole point of the technique
        and also why the span records both timeouts: a waterfall showing only the
        configured one would report a wait that never happened.
        """
        if self.semantic and self._partial_complete:
            return min(self.early_silence_ms, self.trailing_silence_ms)
        return self.trailing_silence_ms

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
            self._resume_ms += FRAME_MS
            if self.speech_ms >= self.min_speech_ms:
                self._started = True
            # Only sustained speech cancels a pending endpoint. One loud frame is a
            # click or a door, and letting it clear the timer is how a turn in a
            # noisy room never ends.
            if self._resume_ms >= self.min_resume_ms:
                self.silence_ms = 0
            return False

        self._resume_ms = 0
        if not self._started:
            return False

        self.silence_ms += FRAME_MS
        return self.silence_ms >= self.silence_needed_ms

    def reset(self) -> None:
        self.speech_ms = 0
        self.silence_ms = 0
        self._started = False
        self._resume_ms = 0
        self._partial_complete = False
