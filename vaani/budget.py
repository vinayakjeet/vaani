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

`BargeClock` is here rather than beside the turn-taking decisions it times, because
the two clocks share the thing that is easy to get wrong. Both are backdated to a
moment nothing had noticed yet, one to the last frame of speech and one to the first,
and both would report a considerably better system if they were started where the code
happens to find out.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    # When the listener could actually hear the answer, which is not when it was sent.
    #
    # The filler is yielded in full before the answer's first chunk, so the answer is
    # queued behind however long the filler takes to play, between 1.8 and 3.5 seconds
    # for the committed bank. Audio leaves the process far faster than real time, so
    # `first_answer_audio_ms` records a moment at which the listener is still hearing
    # "ek minute". Judging the target on it flatters every turn the filler fired on,
    # which is most of them, because the deadline is shorter than the endpoint wait.
    #
    # This is the same error SPEC was written to prevent, one stage further down: a
    # measurement taken where the code was convenient rather than where the listener is.
    first_answer_heard_ms: float | None = None
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

    def mark_heard(self, queued_ahead_ms: float) -> None:
        """When the answer reaches the ear, given what is still playing in front of it.

        Called from the transport rather than from the producer, because only the
        transport knows how much audio is queued ahead. Recorded once: the first answer
        chunk is the one a listener perceives as the reply starting.
        """
        if self.first_answer_heard_ms is None:
            self.first_answer_heard_ms = self.elapsed_ms() + max(0.0, queued_ahead_ms)

    @property
    def target_ms(self) -> float | None:
        """The number the targets are judged on: heard, falling back to sent.

        The fallback is for a transport that cannot estimate playout, where the sent
        time is the best available answer and is known to be optimistic. A configuration
        measured that way says so rather than being compared as though it were the same
        number.
        """
        return (
            self.first_answer_heard_ms
            if self.first_answer_heard_ms is not None
            else self.first_answer_audio_ms
        )

    @property
    def met_p50_target(self) -> bool:
        """Against the answer as heard, never the filler and never the send time."""
        target = self.target_ms
        return target is not None and target < FIRST_AUDIO_P50_MS

    @property
    def met_p95_floor(self) -> bool:
        target = self.target_ms
        return target is not None and target < FIRST_AUDIO_P95_MS


@dataclass
class BargeClock:
    """How long an interruption took, from the frame the interrupting speech began on.

    Three numbers rather than one, because speech over the agent is a hypothesis before
    it is a decision. Audio stops early and provisionally; the turn is abandoned later
    and only once something is sure. Publishing the first alone would say the agent shuts
    up in a hundred milliseconds while the reply it is no longer allowed to give is still
    being generated behind it. Publishing the second alone would understate the thing the
    listener perceives. `bench/stages.md` defines all three and this is the code written
    to it.

    The start is backdated the same way `TurnClock`'s is and for the same reason. Barge-in
    detection needs `min_speech_ms` of speech before it fires, so a clock started at the
    detection would silently remove that from every number.
    """

    started_ns: int
    # When the server told the client to stop. The other half of the number
    # `bench/stages.md` defines lives in the browser, which pauses on its own level
    # detector without telling anyone, so the earlier of the two can only be taken
    # somewhere that can see both ends. This field is the half the server can honestly
    # claim, and it is named for that rather than for the number it contributes to.
    server_paused_ms: float | None = None
    committed_ms: float | None = None
    resumed_ms: float | None = None
    # What ended it. The paths have different costs by construction, so a median over
    # both is a median over two distributions: the duration path waits out `commit_ms`
    # of speech, and the verified path pays a transcription round trip instead.
    outcome: str = ""

    @classmethod
    def from_speech(cls, run_ms: int) -> BargeClock:
        """A clock started at the first frame of speech, given how much has been heard."""
        return cls(started_ns=time.monotonic_ns() - run_ms * 1_000_000)

    def elapsed_ms(self) -> float:
        return (time.monotonic_ns() - self.started_ns) / 1_000_000

    def mark_paused(self) -> None:
        if self.server_paused_ms is None:
            self.server_paused_ms = self.elapsed_ms()

    def mark_committed(self, outcome: str) -> None:
        """The turn is gone: the generation has moved and production has been stopped.

        Recorded after the cancellation is awaited rather than after it is requested.
        A number taken at the request would say the interruption completed while the
        synthesiser was still writing into a queue, which is the failure the awaited
        cancel in `SpeakingTurn.cancel` exists to prevent and would then be invisible.
        """
        if self.committed_ms is None:
            self.committed_ms = self.elapsed_ms()
            self.outcome = outcome

    def mark_unresolved(self) -> None:
        """The reply finished while the suspicion was still open.

        Reachable through the stuck-hold release: the user says something short, playback
        is held, nothing ever ends the utterance, and after `STUCK_HOLD_S` the rest of the
        answer goes out over a suspicion nobody ever settled. It is neither an
        interruption nor a backchannel, and it has no end boundary to measure, so it is
        kept with none rather than given one. A bench that quietly dropped these would
        report a median over the cases that went well.
        """
        if self.committed_ms is None and self.resumed_ms is None:
            self.outcome = "unresolved"

    def mark_resumed(self, outcome: str = "backchannel") -> None:
        """Held audio is flowing again, on a turn that was never interrupted at all.

        The price of asking rather than assuming. It is charged only on the verified arm
        and only when the answer was no, so it belongs beside the interruptions rather
        than inside them.
        """
        if self.resumed_ms is None:
            self.resumed_ms = self.elapsed_ms()
            self.outcome = outcome


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


_DONE = object()


async def _drive(answer: AsyncIterator[bytes], queue: asyncio.Queue[object]) -> None:
    """Pull `answer` to exhaustion on this task, forwarding every chunk.

    A dedicated task for `answer`'s whole lifetime, not just its first chunk. The
    stages `answer` is built from (`vaani.llm_turn`, `vaani.tts`) hold an
    OpenTelemetry span open across their own `yield`s, and OpenTelemetry's context
    token is only valid to detach in the task that attached it. The previous
    version raced only the first `anext()` in its own task via `ensure_future` and
    then drove the rest of the generator from the caller's task, which opened the
    span in one task and closed it in another on every turn: harmless to the
    answer itself, but it corrupted the tracer's context stack, and enough turns
    in a row made spans stop recording at all with no exception anywhere in this
    module. Driving the whole generator from one task, start to finish, is what
    keeps attach and detach in the same place.
    """
    try:
        async for chunk in answer:
            await queue.put(chunk)
    except Exception as exc:
        await queue.put(exc)
    finally:
        # `speak_within` cancels this task on an early exit rather than waiting
        # for `answer` to end on its own (see there). Cancellation lands wherever
        # this task happened to be suspended, and `await queue.put(chunk)` is a
        # real checkpoint even on an unbounded queue: caught there, `answer` has
        # already produced its chunk and returned control here, so it is idle at
        # a `yield`, not inside an `await` `CancelledError` could reach. Nothing
        # then closes it, which is the same abandoned-generator shape this
        # function exists to fix, one level further in. `aclose` is a no-op on a
        # generator that already finished, so it costs nothing on the ordinary
        # path.
        with contextlib.suppress(asyncio.CancelledError):
            await answer.aclose()
        await queue.put(_DONE)


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
    queue: asyncio.Queue[object] = asyncio.Queue()
    driver = asyncio.ensure_future(_drive(answer, queue))
    first = asyncio.ensure_future(queue.get())
    done, _pending = await asyncio.wait([first], timeout=remaining_ms(clock, deadline_ms) / 1000)

    if not done:
        clock.filler_spoken = True
        logger.info(
            "budget.filler", deadline_ms=deadline_ms, elapsed_ms=round(clock.elapsed_ms())
        )
        # Cut short, not run to its own end. `first` is a second, independent
        # future watching the same queue `driver` fills, so it keeps resolving
        # in the background while this loop is mid-filler. The bench/ablation.py
        # n=20 run found that the previous version, `async for chunk in
        # filler(): yield chunk` with no such check, let the filler finish in
        # full regardless of how much of the real answer was already
        # synthesised by then, which made the streamed pipeline measure slower
        # than the naive baseline: `TurnClock.mark_heard` queues the answer
        # behind whatever filler audio is still unplayed, and a filler that
        # never yields early never shortens that queue. `filler()`'s own
        # generator is closed either way, an ordinary exhaustion or this break,
        # since `aclose` on one already finished costs nothing.
        filler_gen = filler()
        try:
            async for chunk in filler_gen:
                clock.mark_audio(is_answer=False)
                yield chunk
                if first.done():
                    break
        finally:
            await filler_gen.aclose()

    try:
        item = await first
        if item is _DONE:
            # The answer produced nothing at all. Whatever the filler said, the turn
            # has no reply in it, and the caller has to say so rather than let the
            # acknowledgement stand in for one.
            return
        if isinstance(item, Exception):
            raise item

        clock.mark_audio(is_answer=True)
        yield item

        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            clock.mark_audio(is_answer=True)
            yield item
    finally:
        # Reached on ordinary exhaustion, where `driver` is already done and this
        # is a formality, and also on this generator being closed early by an
        # interruption, where it is not. `answer` is never abandoned on purpose by
        # this function, but the caller closing `speak_within` itself is a
        # different decision than this function makes, and the first version of
        # this awaited `driver` unconditionally: a barge-in tore down the consumer
        # side, `driver` kept running because nothing had told it to stop, and the
        # turn's audio task sat blocked on `await driver` for however long the
        # rest of the answer took to finish talking to itself. Cancelling here is
        # what closes `answer` in the task that opened it, which is the property
        # this whole rewrite exists for, and it is also what makes an interruption
        # interrupt again.
        if not driver.done():
            driver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await driver
