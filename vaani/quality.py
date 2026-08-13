"""What the latency budget cannot see: whether the conversation felt like one.

Every number in `vaani/budget.py` measures one axis, responsiveness, and the survey
literature is consistent that spoken dialogue quality is pause handling, turn-taking,
backchanneling and graceful interruption. Our whole budget is a measurement of the
third-and-a-half of one of those. That is not wrong, it is narrow, and this module is
the beginning of a fix.

**One of these has a threshold somebody else defends, and the rest do not.** Bolna
annotates barge-in recovery rate with a Hamming benchmark of above 90% good and below
80% critical. Across two sessions of reading orchestration code and turn-taking papers,
it is the only conversational-quality figure found that arrives with a bar an outside
party will argue for. Everything else here is a ratio this project reports and does not
yet grade, and saying which is which matters more than the numbers: this portfolio has
twice shipped a threshold it intuited.

**The counterpart nobody instruments.** `agent_interrupted_user` counts the agent
starting to talk while the user was still going and being cancelled for it. Our latency
numbers score that as a fast turn, because it is one. It is also the agent talking over
somebody, and a configuration that improves time to first audio by doing more of it has
made the product worse in a way the headline number rewards.

**Filler is excluded from the answer's numbers and not from the playout estimate**, and
the distinction is the point. For deciding whether to hold a chunk, filler is audio the
listener is hearing and counts. For asking whether synthesis kept up with playback, or
how long the agent talked for, filler is not the reply and counting it would flatter both.
`TurnClock` already keeps that distinction as its entire reason for existing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)

# Bolna's annotation on the same ratio, and the only externally defended bar in here.
RECOVERY_RATE_GOOD = 90.0
RECOVERY_RATE_CRITICAL = 80.0

# Below this, synthesis is producing audio slower than it is played, so the answer will
# stutter. Not a tuning threshold: one is the point at which a stream stops keeping up,
# which is arithmetic rather than taste.
SPEED_RATIO_STUTTERS = 1.0


class Interruption(StrEnum):
    """Who talked over whom. Both directions are failures and only one gets noticed."""

    USER_INTERRUPTED_AGENT = "user_interrupted_agent"
    AGENT_INTERRUPTED_USER = "agent_interrupted_user"


@dataclass
class Interactivity:
    """One session's conversational quality, accumulated as it happens.

    Wall-clock rather than monotonic nowhere: every duration here is a difference of
    `time.monotonic()`, so a machine adjusting its clock mid-conversation cannot produce
    a negative monologue or an infinite talk ratio.
    """

    user_interrupted_agent: int = 0
    agent_interrupted_user: int = 0
    recoveries: int = 0
    backchannels_ignored: int = 0
    # Turns where the agent was cut off and the user never got a full reply afterwards.
    # Derived rather than counted, from recoveries against interruptions.

    agent_speaking_ms: float = 0.0
    user_speaking_ms: float = 0.0
    longest_agent_monologue_ms: float = 0.0

    # Answer audio only. Filler is deliberately not in either of these.
    answer_audio_s: float = 0.0
    answer_wall_clock_s: float = 0.0

    # Where in the reply each interruption landed. The blueprint calls this the best
    # script-optimisation signal there is, and it is the one thing a barge-in produces
    # that nothing else can: a count of interruptions says the agent talks too much, and
    # a distribution of positions says which sentence it should stop saying. Recorded as
    # a position rather than as text, so it carries no transcript.
    interrupted_at: list[dict[str, int]] = field(default_factory=list)

    _agent_started: float | None = field(default=None, init=False)
    _user_started: float | None = field(default=None, init=False)
    _answer_first_chunk: float | None = field(default=None, init=False)
    _awaiting_recovery: bool = field(default=False, init=False)
    _turn_sentences: int = field(default=0, init=False)
    _turn_answer_s: float = field(default=0.0, init=False)

    def turn_started(self) -> None:
        """A new answer begins, so the per-turn position counters start again.

        Separate from the session totals on purpose. "Interrupted 4 seconds in" is only
        meaningful against the reply it interrupted, and a cumulative figure would say
        every later interruption happened deeper than the first one.
        """
        self._turn_sentences = 0
        self._turn_answer_s = 0.0

    def sentence_spoken(self) -> None:
        self._turn_sentences += 1

    def agent_started_speaking(self) -> None:
        """First audio chunk of a turn reaching the transport, not being produced."""
        if self._agent_started is None:
            self._agent_started = time.monotonic()

    def agent_stopped_speaking(self) -> None:
        if self._agent_started is None:
            return
        spoken = (time.monotonic() - self._agent_started) * 1000
        self.agent_speaking_ms += spoken
        self.longest_agent_monologue_ms = max(self.longest_agent_monologue_ms, spoken)
        self._agent_started = None

    def user_started_speaking(self) -> None:
        if self._user_started is None:
            self._user_started = time.monotonic()

    def user_stopped_speaking(self) -> None:
        if self._user_started is None:
            return
        self.user_speaking_ms += (time.monotonic() - self._user_started) * 1000
        self._user_started = None

    def note_answer_audio(self, duration_s: float) -> None:
        """A chunk of the answer, with the audio time it represents.

        Wall clock is measured from the first answer chunk rather than from the turn's
        start, because the question is whether synthesis keeps up with playback once
        playback has begun. Including the time before the first chunk would fold time to
        first audio into a ratio that is about a different failure.
        """
        if duration_s <= 0:
            return
        now = time.monotonic()
        if self._answer_first_chunk is None:
            self._answer_first_chunk = now
        self.answer_audio_s += duration_s
        self._turn_answer_s += duration_s
        self.answer_wall_clock_s = now - self._answer_first_chunk

    def interrupted(self, kind: Interruption) -> None:
        self.agent_stopped_speaking()
        self.interrupted_at.append(
            {
                # Which sentence was in flight, counting from one. Zero means the user cut
                # in before any sentence had been reported, which is its own finding: the
                # agent was interrupted before it said anything at all.
                "sentence": self._turn_sentences,
                "answer_ms": round(self._turn_answer_s * 1000),
            }
        )
        if kind is Interruption.USER_INTERRUPTED_AGENT:
            self.user_interrupted_agent += 1
            # Only user barge-ins open a recovery, so the rate's denominator stays the
            # thing the benchmark is about.
            self._awaiting_recovery = True
        else:
            self.agent_interrupted_user += 1
        logger.info("quality.interrupted", kind=str(kind))

    def backchannel_ignored(self) -> None:
        """Speech over the agent that turned out to be an acknowledgement.

        Counted because the alternative reading of a low interruption count is that the
        detector never fires, and those two look identical from the latency numbers.
        """
        self.backchannels_ignored += 1

    def response_delivered(self) -> None:
        """A turn's audio finished without being cut off."""
        self.agent_stopped_speaking()
        if not self._awaiting_recovery:
            return
        self._awaiting_recovery = False
        self.recoveries += 1

    @property
    def total_interruptions(self) -> int:
        return self.user_interrupted_agent + self.agent_interrupted_user

    @property
    def recovery_rate(self) -> float | None:
        """Of the turns the user cut off, how many got a full reply afterwards.

        None when nobody has interrupted, which is different from zero and must not be
        rendered as it: a session with no barge-ins has no recovery rate, and charting it
        at 0% would put every quiet conversation below the critical bar.
        """
        if self.user_interrupted_agent == 0:
            return None
        return round(self.recoveries / self.user_interrupted_agent * 100, 1)

    @property
    def talk_to_listen(self) -> float | None:
        """Share of speaking time that was the agent's. None when nobody has spoken."""
        total = self.agent_speaking_ms + self.user_speaking_ms
        if total <= 0:
            return None
        return round(self.agent_speaking_ms / total * 100, 1)

    @property
    def tts_speed_ratio(self) -> float | None:
        """Seconds of answer audio produced per second of wall clock.

        Below one means synthesis cannot keep up with playback and the answer will
        stutter, which no latency number in this project can see: time to first audio is
        excellent in exactly the run where the rest of the reply arrives in pieces.
        """
        if self.answer_wall_clock_s <= 0:
            return None
        return round(self.answer_audio_s / self.answer_wall_clock_s, 2)

    @property
    def stuttering(self) -> bool:
        ratio = self.tts_speed_ratio
        return ratio is not None and ratio < SPEED_RATIO_STUTTERS

    def summary(self) -> dict[str, float | int | None]:
        """The session's numbers, for a log line and for the ablation's per-run data.

        Counts and durations only. Nothing here can be read back into what was said,
        which is the same rule the span contract enforces.
        """
        return {
            "user_interrupted_agent": self.user_interrupted_agent,
            "agent_interrupted_user": self.agent_interrupted_user,
            "total_interruptions": self.total_interruptions,
            "backchannels_ignored": self.backchannels_ignored,
            "recoveries": self.recoveries,
            "barge_in_recovery_rate": self.recovery_rate,
            "agent_speaking_ms": round(self.agent_speaking_ms),
            "user_speaking_ms": round(self.user_speaking_ms),
            "talk_to_listen": self.talk_to_listen,
            "longest_agent_monologue_ms": round(self.longest_agent_monologue_ms),
            "tts_speed_ratio": self.tts_speed_ratio,
        }
