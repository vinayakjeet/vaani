"""M2.4. How long an interruption takes, measured rather than asserted.

Three numbers, defined in `bench/stages.md` before this was written: when audio stops
reaching the listener, when the turn is actually abandoned, and what it costs to resume
after a backchannel. The 100ms figure in the tail-control brief is not a target here and
this script does not compare against one. This portfolio has twice paid for an intuited
threshold, once gating on 2 points against a judge whose noise floor was 20, so the
number is published as whatever it is and a budget is verified against it afterwards.

**The browser is modelled here, and that is the only way this number can be taken.**
Audio stops reaching the listener at the earlier of two events: the client pausing on
its own level detector, and the server's PAUSE arriving. The first happens in the
browser and is never reported, so no clock inside the server can see it, and a client
that did report it would be reporting across an unsynchronised pair of clocks. So
`Browser` below runs the same rule `web/index.html` runs, on the same frames, in this
process, and both ends are read off one monotonic clock. The rule is duplicated rather
than shared, which is a real cost: change `PREEMPT_FRAMES` or `THRESHOLD` in the page
and this bench measures a client that no longer exists. `test_bargein_latency.py` pins
both constants against the page so that drift fails a test instead of quietly moving
the number.

**What this excludes, stated rather than discovered.** Everything runs in one process,
so no number here carries a network round trip. The client-side pause never had one to
carry, since the browser pauses locally, and that is the whole reason M2.11 exists. The
server's PAUSE does, and on the deployed stack it arrives one one-way trip later than
it does here. So the client half is a deployment number and the server half is a floor.
Where the two are close, the deployed system pauses on the client and the round trip
does not matter, which is the claim this measurement is really testing.

Frames are paced at real time, twenty milliseconds apart. Feeding them in a tight loop
would make every duration here a measure of how fast this script can push bytes.

    uv run python -m bench.bargein --runs 20
"""

from __future__ import annotations

import argparse
import array
import asyncio
import contextlib
import json
import math
import statistics
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from vaani.budget import BargeClock
from vaani.endpoint import DEFAULT_MIN_SPEECH_MS, Endpointer
from vaani.protocol import FRAME_MS, SAMPLES_PER_FRAME, ClientMessage, Frame, ServerMessage
from vaani.session import Incoming, VoiceSession
from vaani.turn_taking import DEFAULT_COMMIT_MS

# The browser's own preemption rule, from `web/index.html`. Three frames is 60ms, the
# same figure the server's endpointer uses to tell a syllable from a click.
PREEMPT_FRAMES = 3
PREEMPT_THRESHOLD_RMS = 500

# How far behind schedule the frame clock has to be before it is treated as an idle
# microphone rather than a slow one. Ordinary jitter on this machine is a few
# milliseconds against a 20ms interval; the gap between the scripted audio and the
# interrupting audio is however long the pipeline took to produce its first chunk.
IDLE_GAP_FRAMES = 4

# How far the realised frame interval may exceed 20ms before the absolute columns stop
# describing the system and start describing the machine. Ten percent, which is well
# inside what a raised Windows timer achieves on an idle machine and well outside what it
# achieves under a full test suite: measured at 20.7ms quiet and 28.0ms loaded, and the
# second of those inflated the headline pause from 62ms to 84ms.
PACING_TOLERANCE = 0.10

# What the harness feeds. A tone rather than recorded speech, and the distinction
# matters less here than anywhere else in this project: barge-in latency is a property
# of the control path, and the only thing the audio has to do is clear a level
# threshold at both ends. M4.2's corpus is what the recognition numbers need.
SILENCE = b"\x00" * SAMPLES_PER_FRAME * 2


def tone(amplitude: int) -> bytes:
    samples = array.array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * 220 * n / SAMPLES_PER_FRAME))
            for n in range(SAMPLES_PER_FRAME)
        ),
    )
    return samples.tobytes()


SPEECH = tone(8000)

# How much of the reply is synthesised, and how fast. Long enough that there is always
# something still playing when the interruption arrives: an interruption of a turn that
# had already finished is not an interruption, and a bench that measured those would be
# timing an empty cancellation.
ANSWER_CHUNKS = 200
ANSWER_CHUNK_INTERVAL_S = 0.02
ANSWER_BYTES_PER_SECOND = 6000


@dataclass
class Run:
    """One interruption, from both ends of it."""

    arm: str
    # When the first frame of interrupting speech left this script. Ground truth for the
    # start boundary, and it is available here for the reason `bench/stages.md` asks for:
    # determined after the fact from the frame log rather than from when detection fired.
    speech_began_ns: int
    client_paused_ns: int | None = None
    server_paused_ns: int | None = None
    clock: BargeClock | None = None
    # What the harness actually achieved, against the 20ms it was aiming for. Reported
    # rather than assumed: every duration below is denominated in frames that were meant
    # to be 20ms apart, so a bench running slow reports a system running slow.
    frame_interval_ms: float = 0.0
    # When each frame of the interrupting speech went out, so the detection window can be
    # measured rather than modelled from the run's average.
    interrupt_sent_ns: tuple[int, ...] = ()

    def _ms(self, at: int | None) -> float | None:
        return None if at is None else (at - self.speech_began_ns) / 1_000_000

    @property
    def client_paused_ms(self) -> float | None:
        return self._ms(self.client_paused_ns)

    @property
    def server_paused_ms(self) -> float | None:
        return self._ms(self.server_paused_ns)

    @property
    def paused_ms(self) -> float | None:
        """The earlier of the two, which is when audio stopped reaching the listener."""
        ends = [end for end in (self.client_paused_ms, self.server_paused_ms) if end is not None]
        return min(ends) if ends else None

    @property
    def committed_ms(self) -> float | None:
        return self._from_speech_onset(None if self.clock is None else self.clock.committed_ms)

    @property
    def resumed_ms(self) -> float | None:
        return self._from_speech_onset(None if self.clock is None else self.clock.resumed_ms)

    def _from_speech_onset(self, session_ms: float | None) -> float | None:
        """Put a number the session measured back on the harness's own start.

        Everything in this report has to be measured from one boundary or the columns
        cannot be compared, and the two ends are timed in different places: the pauses
        are observed here, and the commit is observed inside the session against a start
        it inferred. Reported as they came, `committed_ms` was 798.5 against a threshold
        of 800 that it waits out by construction, because the session's inferred start is
        a few milliseconds late and the number was quietly measured from there.

        A floor that a measurement comes in under is the cheapest evidence there is that
        two clocks have been mixed, and it is the only reason this was noticed.
        """
        if session_ms is None or self.clock_error_ms is None:
            return None
        return session_ms + self.clock_error_ms

    @property
    def clock_error_ms(self) -> float | None:
        """How far the session's own backdated start is from where the speech began.

        The check that makes the rest of this falsifiable. Every number the deployed
        system reports about barge-in is measured from `BargeClock.started_ns`, which is
        inferred from the endpointer rather than observed, and nothing in production can
        tell whether that inference is right. Here the harness emitted the frame, so it
        knows.
        """
        if self.clock is None:
            return None
        return (self.clock.started_ns - self.speech_began_ns) / 1_000_000

    def frame_time_ms(self, frames: int) -> float | None:
        """When the nth frame of the interrupting speech actually went out.

        Every threshold in this pipeline is counted in frames of audio, not in
        milliseconds of wall clock: the browser preempts on three, the endpointer starts
        on `min_speech_ms` of them, the duration path commits on `commit_ms` of them. So
        the moment a decision becomes *possible* is the moment its nth frame arrived, and
        on a machine pacing frames at 28ms rather than 20 that is a different moment.

        Reading it off the frames the harness sent, rather than multiplying out a nominal
        20ms, is what separates this pipeline's cost from this script's. Without it every
        number here moves with the load on the machine and the report cannot tell which
        of the two it is describing.
        """
        if frames <= 0 or len(self.interrupt_sent_ns) < frames:
            return None
        return (self.interrupt_sent_ns[frames - 1] - self.speech_began_ns) / 1_000_000

    def overhead_ms(self, value: float | None, frames: int) -> float | None:
        """What a number cost above the frame of audio that first made it possible."""
        earliest = self.frame_time_ms(frames)
        if value is None or earliest is None:
            return None
        return value - earliest

    @property
    def pacing_drift_ms(self) -> float:
        """The part of the clock error this script caused rather than found.

        The session counts frames and assumes each is worth 20ms of audio, which a sound
        card guarantees and a `sleep` loop does not. Every millisecond the harness runs
        slow is a millisecond the count falls behind the wall clock, nine times over
        before detection fires.
        """
        window = DEFAULT_MIN_SPEECH_MS // FRAME_MS
        arrived = self.frame_time_ms(window)
        return 0.0 if arrived is None else arrived - window * FRAME_MS

    @property
    def clock_error_net_ms(self) -> float | None:
        """The error the session is answerable for, with the harness's share removed.

        Both are published. The raw error is what a reader should see, since it is the
        disagreement that actually occurred; this one is what a regression test can
        assert on without failing whenever the machine running it is busy, which is a
        test that gets disabled rather than fixed.
        """
        raw = self.clock_error_ms
        return None if raw is None else raw - self.pacing_drift_ms


class Browser:
    """The transport and the microphone, on one clock, running the page's own rule.

    Both halves in one object on purpose. The number being measured is the earlier of
    something the client did and something the server said, and splitting them across
    two objects would mean comparing two clocks.
    """

    def __init__(self, script: list[Incoming]) -> None:
        # A queue rather than a list, because the interrupting frames are appended once
        # playback has started, which is after the session is already awaiting a receive.
        # Appending to a list the receive loop had already run off the end of would
        # deliver them to nobody.
        self._incoming: asyncio.Queue[Incoming] = asyncio.Queue()
        for item in script:
            self._incoming.put_nowait(item)
        # An absolute schedule, not a sleep per frame. A microphone produces a frame
        # every 20ms because a sound card says so, and it does not fall behind. Sleeping
        # 20ms per frame does fall behind: this machine's timer granularity is coarser
        # than the interval, so each sleep overshoots and the error accumulates. Measured
        # at roughly 26ms a frame before this was a deadline, which inflated every
        # duration in the report and showed up as the session backdating its clock 37ms
        # into the future. The instrument was wrong and the thing it measures was right.
        self._frame_due: float | None = None
        self._last_sent_at: float | None = None
        self.frame_intervals: list[float] = []

        self._loud_frames = 0
        self.preempted = False
        self.playing = False
        self.audio_started = asyncio.Event()
        self.paused = asyncio.Event()
        self.resumed = asyncio.Event()
        self.ready_after_audio = asyncio.Event()

        self.client_paused_ns: int | None = None
        self.server_paused_ns: int | None = None
        self.speech_began_ns: int | None = None
        # When each frame of the interrupting speech went out. Kept so the detection
        # window can be measured rather than assumed to have been 20ms a frame: it is the
        # span the session's backdating is checked against, and modelling it from the
        # run's average interval was wrong whenever the pacing was uneven, which under a
        # full test suite is often.
        self.interrupt_sent_ns: list[int] = []
        self.sent_audio = 0

    def feed(self, items: list[Incoming]) -> None:
        for item in items:
            self._incoming.put_nowait(item)

    async def receive(self) -> Incoming:
        item = await self._incoming.get()
        if item.frame is not None:
            await self._wait_for_frame_time()
            self._on_outgoing_frame(item.frame)
        return item

    async def _wait_for_frame_time(self) -> None:
        """Hold until this frame is due, by the schedule rather than by the last sleep."""
        interval = FRAME_MS / 1000
        now = time.monotonic()
        due = now if self._frame_due is None else self._frame_due

        if self._last_sent_at is not None:
            # Never sooner than one interval after the last frame, whatever the schedule
            # says. Advancing the deadline by a fixed interval stops drift accumulating,
            # and on its own it also lets a run that fell behind catch up by sending the
            # next few frames back to back. Speech does not arrive faster than real time,
            # and a burst of it put the client's own pause 56ms after a speech onset that
            # its three-frame detector cannot beat 60ms on. A floor that a measurement
            # comes in under is a measurement of the harness.
            due = max(due, self._last_sent_at + interval)

        if now - due > IDLE_GAP_FRAMES * interval:
            # The microphone was not producing for a while: the interrupting frames are
            # handed over only once playback has started, which is well after the last
            # scripted frame went out. A deadline still counting through that gap is far
            # in the past, and every frame after it would go out immediately until the
            # arithmetic caught up. That is a burst of speech arriving faster than anyone
            # can talk, measured as an interruption detected impossibly quickly.
            due = now

        if due > now:
            await asyncio.sleep(due - now)

        sent = time.monotonic()
        since_last = None if self._last_sent_at is None else sent - self._last_sent_at
        # The idle gap is not a frame interval and averaging it in would report the
        # harness as having paced audio at several hundred milliseconds a frame.
        if since_last is not None and since_last <= IDLE_GAP_FRAMES * interval:
            self.frame_intervals.append(since_last)
        self._last_sent_at = sent
        # Advanced by the interval, never reset to now. A frame delivered late does not
        # move the ones behind it, which is what stops one slow scheduler tick becoming
        # a permanent offset in every number measured after it.
        self._frame_due = due + interval

    def _on_outgoing_frame(self, frame: Frame) -> None:
        """The page's `preempt`, on the frame about to be sent, at the moment it is sent."""
        level = _rms(frame.pcm)

        # Every frame from the first loud one onward, silence included. The verified path
        # is settled by the endpointer waiting out trailing silence, so the frame that
        # makes that possible is a silent one, and a log of only the loud frames could
        # not say when it arrived.
        if self.speech_began_ns is not None or (self.playing and level >= PREEMPT_THRESHOLD_RMS):
            self.interrupt_sent_ns.append(time.monotonic_ns())

        if self.playing and level >= PREEMPT_THRESHOLD_RMS and self.speech_began_ns is None:
            # Ground truth for the clock's start: the first loud frame sent over
            # playback, which is the frame that later turns out to have begun the
            # interruption rather than the frame anything decided on.
            #
            # One frame back from the send, because a frame is 20ms of audio and cannot
            # be sent before it has been captured. The speech began at the start of that
            # window and the send is its end. Getting this wrong is worth exactly one
            # frame, which is also the entire size of the disagreement it would show
            # against the session's own clock, so a check calibrated the other way would
            # report a 20ms bias in the code and be wrong about which end had it.
            self.speech_began_ns = time.monotonic_ns() - FRAME_MS * 1_000_000

        if not self.playing or self.preempted:
            if level < PREEMPT_THRESHOLD_RMS:
                self._loud_frames = 0
            return

        self._loud_frames = self._loud_frames + 1 if level >= PREEMPT_THRESHOLD_RMS else 0
        if self._loud_frames < PREEMPT_FRAMES:
            return

        self.preempted = True
        self.playing = False
        self.client_paused_ns = time.monotonic_ns()

    async def send_json(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == ServerMessage.AUDIO_START:
            self.playing = True
            self.audio_started.set()
        elif kind == ServerMessage.PAUSE:
            if self.server_paused_ns is None:
                self.server_paused_ns = time.monotonic_ns()
            self.playing = False
            self.paused.set()
        elif kind == ServerMessage.RESUME:
            self.playing = True
            self.preempted = False
            self._loud_frames = 0
            self.resumed.set()
        elif kind == ServerMessage.READY and self.audio_started.is_set():
            # READY after audio has started is the server saying the turn is gone. The
            # page treats it the same way: the held buffer is never coming back.
            self.ready_after_audio.set()

    async def send_bytes(self, _data: bytes) -> None:
        self.sent_audio += 1


@dataclass
class Arm:
    """One way an interruption can go, as the audio that produces it."""

    name: str
    interrupting_frames: int
    trailing_silence_frames: int
    backchannel: bool | None
    # How many frames of speech the duration path needs before it can commit. Zero on the
    # verified arms, where what settles the turn is the utterance ending rather than a
    # count, so there is no frame to measure an overhead against.
    commit_frames: int = 0
    # What the verifier costs. Zero by default and reported as zero, because the stub
    # here answers instantly and a real recogniser does not. `bench/stages.md` puts the
    # transcription inside `committed_ms` on the grounds that the user waits through it
    # either way, so the deployed number is this one plus whatever the provider took.
    verify_ms: float = 0.0
    ends: tuple[str, ...] = ()


# The duration path needs speech past `DEFAULT_COMMIT_MS`, which is 800ms, so 50 frames
# is comfortably over. The verified arms need speech past `DEFAULT_VERIFY_MS` and under
# the commit threshold, then enough silence to end the utterance so it can be identified.
ARMS: tuple[Arm, ...] = (
    Arm("duration", interrupting_frames=50, trailing_silence_frames=0, backchannel=None,
        commit_frames=DEFAULT_COMMIT_MS // FRAME_MS, ends=("paused", "committed")),
    Arm("verified_interruption", interrupting_frames=20, trailing_silence_frames=60,
        backchannel=False, ends=("paused", "committed")),
    Arm("verified_backchannel", interrupting_frames=20, trailing_silence_frames=60,
        backchannel=True, ends=("paused", "resumed")),
)


async def one_run(arm: Arm) -> Run:
    """Drive a real `VoiceSession` through one turn and one interruption."""
    script: list[Incoming] = [Incoming(control=ClientMessage.START)]
    script += [_frame(SPEECH) for _ in range(30)]
    script += [_frame(SILENCE) for _ in range(60)]

    browser = Browser(script)

    async def verdict(_pcm: bytes) -> bool:
        if arm.verify_ms:
            await asyncio.sleep(arm.verify_ms / 1000)
        return bool(arm.backchannel)

    voice = VoiceSession(
        transport=browser,
        answer=_answer,
        filler=_filler,
        endpointer=Endpointer(),
        bytes_per_second=ANSWER_BYTES_PER_SECOND,
        backchannel=None if arm.backchannel is None else verdict,
    )

    task = asyncio.create_task(voice.run())
    try:
        await asyncio.wait_for(browser.audio_started.wait(), timeout=10.0)
        # Audio is flowing. Interrupt it now rather than from the script, so the
        # interrupting frames cannot be sent before there is anything to interrupt.
        browser.feed(
            [_frame(SPEECH) for _ in range(arm.interrupting_frames)]
            + [_frame(SILENCE) for _ in range(arm.trailing_silence_frames)]
        )

        settled = browser.resumed if arm.backchannel else browser.ready_after_audio
        await asyncio.wait_for(settled.wait(), timeout=20.0)
        # The clock is retired inside the handler that just sent that message, so it is
        # already in the list by the time the message is observed here.
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    if browser.speech_began_ns is None:
        raise RuntimeError(f"{arm.name}: no interrupting speech was ever sent over playback")

    intervals = browser.frame_intervals
    return Run(
        arm=arm.name,
        speech_began_ns=browser.speech_began_ns,
        client_paused_ns=browser.client_paused_ns,
        server_paused_ns=browser.server_paused_ns,
        clock=voice.barges[0] if voice.barges else None,
        frame_interval_ms=1000 * statistics.fmean(intervals) if intervals else 0.0,
        interrupt_sent_ns=tuple(browser.interrupt_sent_ns),
    )


@contextlib.contextmanager
def steady_frame_clock() -> Iterator[None]:
    """Ask Windows for a timer this bench can pace 20ms frames against.

    The default system period is 15.6ms, which is most of a frame, so every
    `asyncio.sleep(0.02)` lands on the next tick after the one it wanted: measured here
    at 22.3ms mean and 35.7ms worst against a 20ms target. That is not a small error in
    a bench whose shortest number is three frames long. Raising the period to 1ms brings
    it to 20.6ms mean and 21.3ms worst, and the residual is reported rather than assumed
    away, because it is still a real 3% and it is still in every number below.

    A no-op everywhere else. Linux already schedules this accurately, and CI is Ubuntu.
    """
    if sys.platform != "win32":
        yield
        return

    import ctypes

    timer = ctypes.WinDLL("winmm")
    timer.timeBeginPeriod(1)
    try:
        yield
    finally:
        # Paired, because the period is a process-wide request the operating system
        # honours until it is dropped, and leaving it raised makes every other process
        # on the machine wake more often for as long as this one lives.
        timer.timeEndPeriod(1)


async def measure(runs: int) -> list[Run]:
    """Every arm, interleaved rather than blocked.

    Interleaved because a machine that gets busy halfway through would otherwise put
    the slow half of the session into whichever arm ran second, and the comparison
    between arms is the whole point of running three.
    """
    out: list[Run] = []
    with steady_frame_clock():
        for index in range(runs):
            for arm in ARMS:
                out.append(await one_run(arm))
            print(f"  run {index + 1}/{runs}", file=sys.stderr)
    return out


def summarise(runs: list[Run]) -> dict:
    report: dict = {
        "runs_per_arm": len(runs) // len(ARMS),
        "frame_interval_ms": _distribution([run.frame_interval_ms for run in runs]),
        "arms": {},
    }
    for arm in ARMS:
        rows = [run for run in runs if run.arm == arm.name]
        unresolved = [
            row for row in rows if row.clock is None or row.clock.outcome == "unresolved"
        ]
        report["arms"][arm.name] = {
            "n": len(rows),
            # Stated rather than filtered out quietly. An interruption that was neither
            # committed nor resumed has no end boundary, so it cannot be in a median,
            # and a median over a silently filtered population is the failure this whole
            # project is about.
            "unresolved": len(unresolved),
            "verify_ms": arm.verify_ms,
            "paused_ms": _distribution([row.paused_ms for row in rows]),
            "client_paused_ms": _distribution([row.client_paused_ms for row in rows]),
            "server_paused_ms": _distribution([row.server_paused_ms for row in rows]),
            "committed_ms": _distribution([row.committed_ms for row in rows]),
            "resumed_ms": _distribution([row.resumed_ms for row in rows]),
            "clock_error_ms": _distribution([row.clock_error_ms for row in rows]),
            "clock_error_net_ms": _distribution([row.clock_error_net_ms for row in rows]),
            # What each decision cost above the frame of audio that first made it
            # possible. These are the rows a regression watches, because they are the
            # only ones that do not move when the machine running the bench does.
            "client_pause_overhead_ms": _distribution(
                [row.overhead_ms(row.client_paused_ms, PREEMPT_FRAMES) for row in rows]
            ),
            "server_pause_overhead_ms": _distribution(
                [
                    row.overhead_ms(row.server_paused_ms, DEFAULT_MIN_SPEECH_MS // FRAME_MS)
                    for row in rows
                ]
            ),
            "commit_overhead_ms": _distribution(
                [row.overhead_ms(row.committed_ms, arm.commit_frames) for row in rows]
            ),
        }
    return report


def _distribution(values: list[float | None]) -> dict | None:
    present = sorted(value for value in values if value is not None)
    if not present:
        return None
    return {
        "n": len(present),
        "p50": round(statistics.median(present), 1),
        "p95": round(_percentile(present, 95), 1),
        "min": round(present[0], 1),
        "max": round(present[-1], 1),
    }


def _percentile(sorted_values: list[float], percentile: int) -> float:
    """Nearest rank, which is the honest choice at n=20.

    Interpolating between the two neighbouring samples invents a value that was never
    measured, and at this sample size the p95 is the second-largest observation however
    it is dressed up.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = math.ceil(percentile / 100 * len(sorted_values))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def render(report: dict) -> str:
    pacing = report["frame_interval_ms"]
    lines = [
        f"Barge-in latency, {report['runs_per_arm']} runs per arm.",
        "",
        "Milliseconds from the first frame of interrupting speech. One process, so no",
        "number here carries a network round trip: the client pause never had one, and",
        "the server pause has one on the deployed stack that this does not show.",
        "",
        # First, because every number under it is denominated in these. A run whose
        # frames went out 26ms apart was feeding speech at three quarters of real time,
        # and reporting its milliseconds without saying so would publish the harness.
        f"Frames were sent {pacing['p50']}ms apart against a target of {FRAME_MS}, "
        f"worst {pacing['max']}ms.",
        _pacing_verdict(pacing),
        "",
        f"{'arm':<24}{'number':<20}{'n':>4}{'p50':>9}{'p95':>9}{'max':>9}",
    ]
    for name, arm in report["arms"].items():
        for number in ("paused_ms", "client_paused_ms", "server_paused_ms",
                       "committed_ms", "resumed_ms",
                       "client_pause_overhead_ms", "server_pause_overhead_ms",
                       "commit_overhead_ms"):
            stats = arm[number]
            if stats is None:
                continue
            lines.append(
                f"{name:<24}{number:<20}{stats['n']:>4}"
                f"{stats['p50']:>9.1f}{stats['p95']:>9.1f}{stats['max']:>9.1f}"
            )
        if arm["unresolved"]:
            lines.append(f"{name:<24}{'unresolved':<20}{arm['unresolved']:>4}")
        for label, key in (("clock error", "clock_error_ms"),
                           ("clock error net", "clock_error_net_ms")):
            error = arm[key]
            if error is not None:
                lines.append(
                    f"{name:<24}{label:<20}{error['n']:>4}"
                    f"{error['p50']:>9.1f}{error['p95']:>9.1f}{error['max']:>9.1f}"
                )
        lines.append("")
    lines.extend([
        "clock error is the session's own backdated start minus the frame the harness",
        "actually sent. Nothing in production can check that inference; this can. The net",
        "row takes out what the pacing above caused, which is the part this script owes",
        "rather than the part it found.",
    ])
    return "\n".join(lines)


def _pacing_verdict(pacing: dict) -> str:
    """Say whether the wall-clock columns are worth reading at all.

    A latency bench on a machine that cannot feed audio at real time reports a slow
    system, and every absolute number in this report scales with the number above. Said
    at the top rather than left for a reader to work out from an interval they were not
    looking for, because an unlabelled 84ms and a real 62ms are the same table.

    The overhead columns survive a slow run and the absolute ones do not, which is the
    whole reason both are printed.
    """
    if pacing["p50"] <= FRAME_MS * (1 + PACING_TOLERANCE):
        return "Within tolerance, so the absolute columns below are the system's own."
    return (
        f"Outside the {PACING_TOLERANCE:.0%} tolerance: this machine could not feed audio "
        f"at real time, so every absolute column below is inflated by roughly the same "
        f"proportion. Read the overhead rows, which are measured from the frame that "
        f"arrived rather than the frame that should have."
    )


def _frame(pcm: bytes, generation: int = 1) -> Incoming:
    return Incoming(frame=Frame(generation=generation, pcm=pcm))


def _rms(pcm: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm)
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


async def _answer(
    frames: AsyncIterator[bytes],
    still_current: Callable[[], bool],
    on_transcript=None,
    on_sentence=None,
) -> AsyncIterator[bytes]:
    """A reply with no provider behind it.

    Deliberately not a stub that returns instantly. What is being measured is how long
    the pipeline takes to stop, and a producer that had already finished would make
    every cancellation free.
    """
    async for _pcm in frames:
        pass
    if on_transcript is not None:
        await on_transcript("kya main is scheme ke liye eligible hoon")
    if on_sentence is not None:
        await on_sentence("Haan, aap eligible hain.")
    for index in range(ANSWER_CHUNKS):
        await asyncio.sleep(ANSWER_CHUNK_INTERVAL_S)
        yield f"chunk{index:04d}".encode() * 8


async def _filler() -> AsyncIterator[bytes]:
    yield b"ek minute"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--json", type=Path, default=None, help="write the raw report here")
    args = parser.parse_args()

    runs = await measure(args.runs)
    report = summarise(runs)

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
