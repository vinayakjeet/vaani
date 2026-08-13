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
from enum import StrEnum

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

# How long a microphone may produce nothing before the user is told. Longer than
# somebody taking a moment to think, short enough that a muted microphone is
# diagnosed rather than experienced as a dead demo. Provisional like every other
# threshold here, and it trades a false complaint against a silent wait.
DEFAULT_SILENCE_TIMEOUT_MS = 5000

# Below this an input is not quiet, it is off. Full scale for PCM16 is 32767, so a
# reading under one sample of amplitude is digital zero plus dither rather than a
# room. The distinction matters because the two have different answers: a muted
# microphone needs unmuting, and a quiet one needs the speaker to move closer.
SILENT_RMS = 1.0


class MicState(StrEnum):
    OK = "ok"
    SILENT = "silent"
    TOO_QUIET = "too_quiet"


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

    silence_timeout_ms: int = DEFAULT_SILENCE_TIMEOUT_MS

    speech_ms: int = field(default=0, init=False)
    silence_ms: int = field(default=0, init=False)
    # Everything heard before this turn started, so a microphone that is producing
    # nothing can be told apart from one producing something too quiet to use.
    waiting_ms: int = field(default=0, init=False)
    peak_rms: float = field(default=0.0, init=False)
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

    def diagnose(self) -> MicState:
        """Whether the microphone is working, judged only while waiting for speech.

        SPEC S6. An endpoint that will never arrive is not a state to wait in: a
        muted microphone produces frames forever and the old behaviour discarded all
        of them silently, so the demo looked dead rather than saying what was wrong.

        Two different failures with two different answers. Digital silence means the
        input is off, muted at the operating system or never permitted. Audible but
        below the threshold means the microphone works and the speaker is too far
        away or too quiet. Reporting one as the other sends somebody to the wrong
        setting.
        """
        if self._started or self.waiting_ms < self.silence_timeout_ms:
            return MicState.OK
        return MicState.SILENT if self.peak_rms < SILENT_RMS else MicState.TOO_QUIET

    def accept(self, pcm: bytes) -> bool:
        """Feed one frame. Returns True when the turn has ended.

        Silence before speech is discarded rather than counted, so a user who
        takes three seconds to begin does not end the turn before starting it.
        """
        level = rms(pcm)
        # Accumulated unconditionally. Guarding this on "not yet started" was
        # redundant: `diagnose` already returns OK once a turn is under way, so the
        # guard changed no behaviour and deleting it changed no test. What protects a
        # mid-turn pause from being read as a fault is that check, not this one, and
        # `test_a_pause_mid_turn_is_not_a_microphone_fault` is where that is pinned.
        self.waiting_ms += FRAME_MS
        self.peak_rms = max(self.peak_rms, level)

        if level >= self.threshold_rms:
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
        self.waiting_ms = 0
        self.peak_rms = 0.0
