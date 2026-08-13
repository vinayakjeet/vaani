"""Whether to interrupt, whether to send audio, and what the listener actually heard.

Taken from Bolna's `interruption_manager.py` and `task_manager.py`, read directly. Four
mechanisms, each of which we were missing or had backwards.

The audio gate has three states rather than two. We had SEND or BLOCK by generation, which
answers "is this audio stale" and never answers "should this audio go out right now". WAIT
is the missing one: the user has started talking, so hold what is queued instead of either
sending it over them or throwing it away.

Interruption is decided by word count plus a closed set of stop words, not by a lexicon of
backchannels. That is the inverse of what reading a blueprint suggested, and it is right:
backchannels are an open set in Hinglish, where "haan", "achha", "hmm", "ji", "theek hai",
"accha ji" and a dozen more all mean keep going. The short phrases that mean *stop* are a
closed set, and word count decides everything else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)

# How many words of speech during playback count as a real interruption. Bolna's default is
# 3. Provisional here like every other threshold, and it is the knob that trades talking over
# people against ignoring them.
DEFAULT_WORDS_FOR_INTERRUPTION = 3

# Short phrases that interrupt despite being short, in both scripts. A closed set on purpose:
# the alternative is enumerating every way a Hinglish speaker says "carry on", which cannot be
# done. Anything not here and under the word count is treated as a backchannel.
STOP_WORDS: frozenset[str] = frozenset(
    {
        "ruko", "ruk", "rukiye", "nahi", "nahin", "na", "band", "bas", "sunno", "suno",
        "arre", "wait", "stop", "no", "hold", "hold on", "one minute", "ek minute",
        "galat", "wrong", "enough", "chup",
        "रुको", "रुकिए", "नहीं", "बंद", "बस", "सुनो", "गलत", "एक मिनट",
    }
)

# How long after the user stops speaking we keep holding audio, so a pause that turns out to
# be mid-thought does not get talked over. Bolna uses 900ms and skips it for the first turns
# so a greeting is not delayed.
DEFAULT_GRACE_MS = 900
TURNS_BEFORE_GRACE = 2


class AudioStatus(StrEnum):
    SEND = "send"
    BLOCK = "block"
    WAIT = "wait"


@dataclass
class TurnTaking:
    """The decisions that make a conversation feel like one."""

    words_for_interruption: int = DEFAULT_WORDS_FOR_INTERRUPTION
    grace_ms: int = DEFAULT_GRACE_MS
    stop_words: frozenset[str] = STOP_WORDS

    user_speaking: bool = field(default=False, init=False)
    turns: int = field(default=0, init=False)
    _utterance_ended_at: float | None = field(default=None, init=False)
    # Wall clock at which queued audio is expected to finish playing.
    _playing_until: float = field(default=0.0, init=False)

    def status(self, generation: int, live: set[int]) -> AudioStatus:
        """Whether a chunk of this generation should go out now, later, or never."""
        if generation not in live:
            return AudioStatus.BLOCK

        if self.user_speaking:
            # Held rather than dropped. The generation is still valid, so this audio is
            # still the right answer; it is only the wrong moment to say it.
            return AudioStatus.WAIT

        # Skipped for the first turns, because a greeting delayed by a grace period reads as
        # a slow service rather than a polite one.
        if self.turns > TURNS_BEFORE_GRACE and self.waiting_since_utterance_ms() < self.grace_ms:
            return AudioStatus.WAIT

        return AudioStatus.SEND

    def should_interrupt(self, transcript: str, *, agent_speaking: bool) -> bool:
        """Whether speech during playback is an interruption or a backchannel.

        Word count first, then the stop words. A single "haan" is neither long enough nor on
        the list, so the agent keeps talking, which is the whole point.
        """
        if not agent_speaking:
            return False
        if self.words_for_interruption <= 0:
            return False

        cleaned = _normalise(transcript)
        if not cleaned:
            return False
        if cleaned in self.stop_words:
            return True
        return len(cleaned.split()) > self.words_for_interruption

    def is_backchannel(self, transcript: str, *, agent_speaking: bool) -> bool:
        """The same decision from the other side, for a final transcript to discard.

        Bolna's `is_false_interruption`. A short non-stop utterance heard while the agent was
        speaking is not a question, and answering it produces a reply to "hmm".
        """
        if not agent_speaking:
            return False
        cleaned = _normalise(transcript)
        if not cleaned:
            return True
        if cleaned in self.stop_words:
            return False
        return len(cleaned.split()) <= self.words_for_interruption

    def note_user_speaking(self) -> None:
        self.user_speaking = True

    def note_user_stopped(self, *, count_turn: bool = True) -> None:
        self.user_speaking = False
        self._utterance_ended_at = time.monotonic()
        if count_turn:
            self.turns += 1

    def waiting_since_utterance_ms(self) -> float:
        if self._utterance_ended_at is None:
            return float("inf")
        return (time.monotonic() - self._utterance_ended_at) * 1000

    def note_audio_queued(self, duration_s: float) -> None:
        """Advance the playout estimate by a chunk's own duration.

        Audio is handed to the transport far faster than real time, so a chunk starts playing
        when the previous one finishes, not when it was sent. Our `queued_ms` measured the send
        and called it playback. This is Bolna's line, and it needs no acknowledgement from the
        client to be useful.
        """
        if duration_s <= 0:
            return
        self._playing_until = max(self._playing_until, time.monotonic()) + duration_s

    def playing_ms_remaining(self) -> float:
        return max(0.0, (self._playing_until - time.monotonic()) * 1000)

    def drop_playout(self) -> None:
        """Queued audio was discarded, so nothing is playing."""
        self._playing_until = 0.0


def heard_prefix(spoken: str, heard: str) -> str:
    """The part of a reply the listener actually heard, for the conversation record.

    Without this the model believes it said things nobody heard, and two turns later it refers
    back to them. Bolna's rule, and the last branch is the one worth copying: if the heard text
    cannot be located in what was spoken, the turn data is stale or mismatched, and the honest
    answer is that nothing was heard. Splicing foreign text into the record would be worse than
    losing it.
    """
    if not spoken:
        return heard.strip()
    heard = heard.strip()
    if not heard:
        return ""

    index = spoken.find(heard)
    if index != -1:
        return spoken[: index + len(heard)]

    # Synthesisers add and drop trailing whitespace, so try shorter prefixes before giving up.
    if len(heard) > 3 and spoken[:3] == heard[:3]:
        for end in range(len(heard), 0, -1):
            candidate = heard[:end].strip()
            if candidate and spoken.startswith(candidate):
                return candidate

    logger.warning("turn.heard_text_not_found", spoken_chars=len(spoken), heard_chars=len(heard))
    return ""


def whole_words(text: str) -> str:
    """Trim a partial to its last complete word.

    A transcript cut mid-word puts a fragment into the record, and a fragment in Hinglish is
    frequently a different word. Nothing is kept if there is no boundary at all.
    """
    trimmed = (text or "").strip()
    if not trimmed:
        return ""
    boundary = trimmed.rfind(" ")
    return trimmed[:boundary] if boundary > 0 else ""


def _normalise(text: str) -> str:
    return " ".join((text or "").strip().casefold().split()).strip(".,!?।॥")
