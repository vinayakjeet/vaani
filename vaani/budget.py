"""The latency budget, and the clock that decides whether it was met.

Two targets, and they are different problems. p50 under 500ms is bought by
optimisation: remove a wait, overlap two stages, skip a request. p95 under 800ms
cannot be, because the tail is a provider having a bad second and no pipeline
design makes a free-tier endpoint answer faster when it is queueing. A tail target
is met by bounding what can go wrong, which means a deadline and something audible
when the deadline passes.

That mechanism is also where this measurement could quietly become dishonest, so
the accounting is built to resist it. A filler acknowledgement is audio, and
counting it as time to first audio would let any system hit any target by learning
to say "achha" quickly. So the clock keeps two numbers: when audio of any kind
started, and when the answer's audio started. The headline is the second. A
configuration that met the target by talking over the gap reports how often it did,
and a turn covered by filler is a turn the answer was late in.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# The published targets, from SPEC's Proof Artifact. Named here so a bench script
# and a regression assertion read the same numbers rather than each carrying a copy.
FIRST_AUDIO_P50_MS = 500
FIRST_AUDIO_P95_MS = 800

# When the filler goes out. Set below the p95 floor rather than at it: the filler
# itself has to be synthesised and reach the browser, and a deadline at 800ms
# guarantees a breach instead of preventing one.
FILLER_DEADLINE_MS = 600


@dataclass
class TurnClock:
    """Time to first audio, measured from the last frame of user speech.

    From end of speech rather than from endpoint detection, which is stricter than
    most published figures and includes the entire endpoint wait. Two systems
    quoting the same number can differ by a factor of two in the only thing a
    listener perceives, and this is the harder of the two clocks to look good on.
    """

    started_ns: int = field(default_factory=time.monotonic_ns)
    first_audio_ms: float | None = None
    first_answer_audio_ms: float | None = None
    filler_spoken: bool = False

    def elapsed_ms(self) -> float:
        return (time.monotonic_ns() - self.started_ns) / 1_000_000

    def mark_audio(self, *, is_answer: bool) -> None:
        """Record that audio went out, and whether it was the answer.

        Both fields are set on the first answer chunk when no filler preceded it, so
        a turn that met the target honestly reports the same number twice rather
        than leaving the headline field empty.
        """
        now = self.elapsed_ms()
        if self.first_audio_ms is None:
            self.first_audio_ms = now
        if is_answer and self.first_answer_audio_ms is None:
            self.first_answer_audio_ms = now

    @property
    def met_p50_target(self) -> bool:
        """Against the answer's audio, never the filler's."""
        return (
            self.first_answer_audio_ms is not None
            and self.first_answer_audio_ms < FIRST_AUDIO_P50_MS
        )

    @property
    def met_p95_floor(self) -> bool:
        return (
            self.first_answer_audio_ms is not None
            and self.first_answer_audio_ms < FIRST_AUDIO_P95_MS
        )


def remaining_ms(clock: TurnClock, deadline_ms: int) -> float:
    """How much of the listener's patience is left, not how long the pipeline has run.

    The deadline is a promise to the person waiting, so it is measured from when they
    stopped speaking. By the time an answer starts being produced the trailing silence
    has already been spent, several hundred milliseconds of it, and a deadline that
    restarts there promises 600ms and delivers 1300. Measured against the real clock a
    turn can arrive already overdue, and then the filler is due immediately, which is
    the correct answer rather than an edge case.
    """
    return max(0.0, deadline_ms - clock.elapsed_ms())


async def speak_within(
    answer: AsyncIterator[bytes],
    filler: Callable[[], AsyncIterator[bytes]],
    clock: TurnClock,
    deadline_ms: int = FILLER_DEADLINE_MS,
) -> AsyncIterator[bytes]:
    """Yield the answer's audio, covering the gap with filler if it is late.

    The answer is never abandoned and never restarted. The filler is spoken while it
    is still coming, and then the answer follows: a listener hears an
    acknowledgement and then a reply, which is what a person does when they need a
    moment. Cancelling the answer instead would trade a slow answer for none.

    The pending first chunk is kept rather than cancelled when the deadline passes.
    Cancelling `anext` would close the generator mid-flight and throw away the work
    already done, which is the opposite of what a deadline is for.
    """
    first = asyncio.ensure_future(anext(answer, None))
    done, _pending = await asyncio.wait([first], timeout=remaining_ms(clock, deadline_ms) / 1000)

    if not done:
        clock.filler_spoken = True
        logger.info(
            "budget.filler", deadline_ms=deadline_ms, elapsed_ms=round(clock.elapsed_ms())
        )
        async for chunk in filler():
            clock.mark_audio(is_answer=False)
            yield chunk

    chunk = await first
    if chunk is None:
        # The answer produced nothing at all. Whatever the filler said, the turn
        # has no reply in it, and the caller has to say so rather than let the
        # acknowledgement stand in for one.
        return

    clock.mark_audio(is_answer=True)
    yield chunk

    async for chunk in answer:
        clock.mark_audio(is_answer=True)
        yield chunk
