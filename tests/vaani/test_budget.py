from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from vaani.budget import (
    FIRST_AUDIO_P50_MS,
    FIRST_AUDIO_P95_MS,
    TurnClock,
    remaining_ms,
    speak_within,
)


async def prompt_answer() -> AsyncIterator[bytes]:
    yield b"answer-1"
    yield b"answer-2"


def slow_answer(delay: float):
    async def produce() -> AsyncIterator[bytes]:
        await asyncio.sleep(delay)
        yield b"answer-1"
        yield b"answer-2"

    return produce


async def filler() -> AsyncIterator[bytes]:
    yield b"achha"


async def slow_silence() -> AsyncIterator[bytes]:
    """Late and then empty, which is the case that matters. An answer that is empty
    immediately beats the deadline and no filler is due."""
    await asyncio.sleep(0.05)
    return
    yield b""  # pragma: no cover


async def test_a_prompt_answer_needs_no_filler() -> None:
    clock = TurnClock()

    heard = [chunk async for chunk in speak_within(prompt_answer(), filler, clock)]

    assert heard == [b"answer-1", b"answer-2"]
    assert not clock.filler_spoken


async def test_a_prompt_answer_reports_the_same_number_twice() -> None:
    """So a turn that met the target honestly does not leave the headline field
    empty, which a reader would have to interpret."""
    clock = TurnClock()

    async for _chunk in speak_within(prompt_answer(), filler, clock):
        pass

    assert clock.first_audio_ms == clock.first_answer_audio_ms


async def test_a_late_answer_is_covered_by_filler_and_then_arrives() -> None:
    """The answer is never abandoned. A listener hears an acknowledgement and then
    the reply, which is what a person does when they need a moment. Cancelling would
    trade a slow answer for none."""
    clock = TurnClock()

    heard = [
        chunk
        async for chunk in speak_within(
            slow_answer(0.05)(), filler, clock, deadline_ms=10
        )
    ]

    assert heard == [b"achha", b"answer-1", b"answer-2"]
    assert clock.filler_spoken


async def test_filler_does_not_count_as_the_answer() -> None:
    """The integrity rule, as a test.

    Counting filler as time to first audio would let any system hit any target by
    learning to say "achha" quickly. The headline number is the answer's, and the
    two have to differ when filler was spoken or the accounting is not measuring
    what it claims.
    """
    clock = TurnClock()

    async for _chunk in speak_within(slow_answer(0.05)(), filler, clock, deadline_ms=10):
        pass

    assert clock.first_audio_ms is not None
    assert clock.first_answer_audio_ms is not None
    assert clock.first_audio_ms < clock.first_answer_audio_ms


async def test_the_target_is_judged_against_the_answer_not_the_filler() -> None:
    """A configuration that met p95 by talking over the gap has not met it."""
    clock = TurnClock()
    clock.first_audio_ms = 50.0
    clock.first_answer_audio_ms = 2_000.0
    clock.filler_spoken = True

    assert not clock.met_p50_target
    assert not clock.met_p95_floor


def test_the_targets_are_the_published_ones() -> None:
    """Pinned so a bench script and a regression assertion cannot drift from SPEC,
    and so moving a target is a visible change rather than an edit in one file."""
    assert FIRST_AUDIO_P50_MS == 500
    assert FIRST_AUDIO_P95_MS == 800


def test_a_turn_with_no_audio_meets_nothing() -> None:
    """An unmeasured turn must not read as a passing one. `None` compares as
    smaller than anything if the check is written carelessly."""
    clock = TurnClock()

    assert not clock.met_p50_target
    assert not clock.met_p95_floor


async def test_an_answer_that_produces_nothing_is_not_covered_by_the_filler() -> None:
    """Whatever the acknowledgement said, the turn has no reply in it. Letting the
    filler stand in for one is how a broken pipeline sounds healthy."""
    clock = TurnClock()

    heard = [chunk async for chunk in speak_within(slow_silence(), filler, clock, deadline_ms=10)]

    assert heard == [b"achha"]
    assert clock.first_answer_audio_ms is None
    assert not clock.met_p95_floor


async def test_the_pending_answer_is_not_cancelled_by_the_deadline() -> None:
    """Cancelling `anext` would close the generator mid-flight and throw away work
    already done, which is the opposite of what a deadline is for."""
    closed = False

    async def watched() -> AsyncIterator[bytes]:
        nonlocal closed
        try:
            await asyncio.sleep(0.05)
            yield b"answer-1"
        finally:
            closed = True

    clock = TurnClock()
    heard = [chunk async for chunk in speak_within(watched(), filler, clock, deadline_ms=10)]

    assert heard == [b"achha", b"answer-1"]
    assert closed


async def test_closing_speak_within_early_stops_the_answer_rather_than_outliving_it() -> None:
    """Regression for a bug introduced while fixing another one. `speak_within` used
    to drive `answer` through a dedicated task so its span opens and closes on the
    same task regardless of how the deadline race times the first chunk, but its own
    `finally` awaited that task unconditionally rather than cancelling it. A caller
    closing `speak_within` early, which is what a barge-in does, then blocked until
    the abandoned answer finished talking to itself instead of actually interrupting
    anything. `answer` here would hang for an hour if it were only awaited rather than
    cancelled, so this fails by timing out, not by a wrong value."""
    closed = False

    async def hangs_after_one_chunk() -> AsyncIterator[bytes]:
        nonlocal closed
        try:
            await asyncio.sleep(0.05)
            yield b"answer-1"
            await asyncio.sleep(3600)
            yield b"answer-2"  # pragma: no cover
        finally:
            closed = True

    clock = TurnClock()
    stream = speak_within(hangs_after_one_chunk(), filler, clock, deadline_ms=10)

    first = await anext(stream)
    assert first == b"achha"
    second = await anext(stream)
    assert second == b"answer-1"

    await asyncio.wait_for(stream.aclose(), timeout=1.0)
    assert closed


async def test_a_failing_answer_still_raises_after_filler() -> None:
    """The filler must not swallow a failure. A turn that says "achha" and then
    nothing is the silence SPEC's degradation rules exist to prevent."""

    async def broken() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.01)
        raise RuntimeError("llm died")
        yield b""  # pragma: no cover

    clock = TurnClock()

    with pytest.raises(RuntimeError):
        async for _chunk in speak_within(broken(), filler, clock, deadline_ms=1):
            pass


async def test_the_deadline_is_measured_from_the_listener_not_the_pipeline() -> None:
    """A turn can arrive already overdue, and then the filler is due immediately.

    The deadline is a promise to the person waiting, so it runs from when they stopped
    speaking. By the time an answer starts being produced the trailing silence has
    already been spent, several hundred milliseconds of it, and a deadline that
    restarted there promised 600ms and delivered 1300 in a real browser.
    """
    overdue = TurnClock(started_ns=time.monotonic_ns() - 2_000_000_000)

    assert remaining_ms(overdue, 600) == 0

    heard = [
        chunk async for chunk in speak_within(slow_answer(0.05)(), filler, overdue, 600)
    ]

    assert heard[0] == b"achha"
    assert overdue.filler_spoken


async def test_a_fresh_turn_still_gets_its_full_deadline() -> None:
    """The other side of it. A prompt turn must not be given filler it did not need,
    which is what a deadline of zero for everybody would do."""
    fresh = TurnClock()

    assert remaining_ms(fresh, 600) > 500

    heard = [chunk async for chunk in speak_within(prompt_answer(), filler, fresh, 600)]

    assert heard == [b"answer-1", b"answer-2"]
    assert not fresh.filler_spoken


async def test_a_backdated_clock_reports_the_wait_it_was_given() -> None:
    """The session backdates by the trailing silence, so the number a listener
    experiences includes the wait they actually sat through."""
    clock = TurnClock(started_ns=time.monotonic_ns() - 700_000_000)

    assert clock.elapsed_ms() >= 700


async def test_the_clock_measures_forward() -> None:
    """Guards every assertion above, which are all differences between two readings
    of it. A clock returning a constant would satisfy most of them."""
    clock = TurnClock()
    first = clock.elapsed_ms()
    await asyncio.sleep(0.01)

    assert clock.elapsed_ms() > first


def test_the_answer_is_judged_on_when_it_is_heard_not_when_it_is_sent() -> None:
    """The filler is yielded in full before the answer's first chunk, so the answer is
    queued behind however long the filler takes to play. Audio leaves the process far
    faster than real time, so the send time is a moment at which the listener is still
    hearing "ek minute"."""
    clock = TurnClock()
    clock.mark_audio(is_answer=False)
    clock.mark_audio(is_answer=True)

    # Two seconds of filler still to play in front of it.
    clock.mark_heard(queued_ahead_ms=2000)

    assert clock.first_answer_audio_ms < 100
    assert clock.first_answer_heard_ms >= 2000
    assert clock.target_ms == clock.first_answer_heard_ms
    assert not clock.met_p95_floor


def test_only_the_first_answer_chunk_sets_the_heard_time() -> None:
    """A later chunk is not when the reply started, and letting it overwrite would make
    the number grow with the length of the answer."""
    clock = TurnClock()
    clock.mark_audio(is_answer=True)
    clock.mark_heard(queued_ahead_ms=100)
    first = clock.first_answer_heard_ms

    clock.mark_heard(queued_ahead_ms=5000)

    assert clock.first_answer_heard_ms == first


def test_an_answer_with_nothing_queued_ahead_is_heard_when_it_is_sent() -> None:
    """The turn that met the target honestly: no filler, nothing in front, so the two
    numbers agree rather than one of them being absent."""
    clock = TurnClock()
    clock.mark_audio(is_answer=True)
    clock.mark_heard(queued_ahead_ms=0)

    assert clock.first_answer_heard_ms == pytest.approx(clock.first_answer_audio_ms, abs=5)
    assert clock.met_p50_target


def test_a_transport_that_cannot_estimate_playout_falls_back_to_the_send_time() -> None:
    """Known to be optimistic, and reported as the send time rather than silently
    compared as though it were the same number."""
    clock = TurnClock()
    clock.mark_audio(is_answer=True)

    assert clock.first_answer_heard_ms is None
    assert clock.target_ms == clock.first_answer_audio_ms
