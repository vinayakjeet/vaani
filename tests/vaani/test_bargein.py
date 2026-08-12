from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from vaani.barge_in import QUEUE_SIZE, AudioChunk, SpeakingTurn, is_current
from vaani.state import State, TurnState


class Synthesiser:
    """A producer that reports whether it was closed, and can be held open."""

    def __init__(self, chunks: int = 6, stall: bool = False) -> None:
        self.produced = 0
        self.closed = False
        self._chunks = chunks
        self._stall = stall

    async def __call__(self) -> AsyncIterator[bytes]:
        try:
            for index in range(self._chunks):
                if self._stall:
                    await asyncio.sleep(3600)
                self.produced += 1
                yield f"chunk-{index}".encode()
                await asyncio.sleep(0)
        finally:
            self.closed = True


async def test_audio_arrives_in_order_and_carries_its_generation() -> None:
    turn = SpeakingTurn(generation=7, produce=Synthesiser(chunks=3))

    chunks = [chunk async for chunk in turn.chunks()]

    assert [chunk.data for chunk in chunks] == [b"chunk-0", b"chunk-1", b"chunk-2"]
    assert {chunk.generation for chunk in chunks} == {7}


async def test_a_cancelled_turn_leaves_no_task_running() -> None:
    """The acceptance criterion for M2.1, and the reason `cancel` awaits the task.
    A cancel that returns early leaves a producer writing into a queue nobody
    reads, which is an orphan whether or not anyone calls it one."""
    turn = SpeakingTurn(generation=1, produce=Synthesiser(chunks=1000))
    turn.start()
    await asyncio.sleep(0)

    await turn.cancel()

    assert not turn.running


async def test_a_cancelled_turn_leaves_no_audio_queued() -> None:
    """The other half of the same criterion. Audio still queued is audio that will
    be played, and the user hears the agent answering a question they interrupted."""
    turn = SpeakingTurn(generation=1, produce=Synthesiser(chunks=1000), queue_size=4)
    turn.start()
    while turn.queued == 0:
        await asyncio.sleep(0)

    await turn.cancel()

    assert turn.queued == 0


class Suspended:
    """A producer that only ever suspends at its yield.

    No await inside the body, so a cancellation cannot propagate into it: the task
    is blocked putting to a full queue instead. That makes the way it was shut down
    unambiguous, which is what the test below needs. `aclose` throws `GeneratorExit`
    at the yield; being collected later does not.
    """

    def __init__(self) -> None:
        self.closed_by: str | None = None

    async def __call__(self) -> AsyncIterator[bytes]:
        try:
            for index in range(1000):
                yield f"chunk-{index}".encode()
        except GeneratorExit:
            self.closed_by = "aclose"
            raise
        except asyncio.CancelledError:
            self.closed_by = "cancel"
            raise


async def test_cancelling_closes_the_provider_stream_explicitly() -> None:
    """Cancelling the task alone leaves the synthesiser suspended at its yield, to
    be closed whenever the loop next collects it, which on a free tier is a
    connection held open against a quota somebody is counting.

    Asserting only that it ended up closed is not enough. That test passed with the
    `aclose` deleted, because refcounting finalised the generator in time on this
    machine, which is a race rather than a guarantee. Recording how it was closed
    distinguishes the two.
    """
    synthesiser = Suspended()
    turn = SpeakingTurn(generation=1, produce=synthesiser, queue_size=2)
    turn.start()
    while turn.queued < 2:
        await asyncio.sleep(0)

    await turn.cancel()

    assert synthesiser.closed_by == "aclose"


async def test_the_default_queue_is_bounded() -> None:
    """The bound is what makes an interruption cheap to abandon. Unbounded, a fast
    synthesiser runs whole sentences ahead of playback and an interruption has
    megabytes to throw away.

    Asserted as a ceiling on the queue after letting the producer run freely. An
    earlier version cancelled as soon as one chunk arrived and then checked that not
    everything had been produced, which was true whether or not the queue was
    bounded: the cancel simply landed early.
    """
    synthesiser = Synthesiser(chunks=500)
    turn = SpeakingTurn(generation=1, produce=synthesiser)
    turn.start()

    for _ in range(2000):
        await asyncio.sleep(0)

    queued = turn.queued
    await turn.cancel()

    assert queued <= QUEUE_SIZE
    assert synthesiser.produced < 500


async def test_cancelling_a_stalled_producer_still_returns() -> None:
    """A synthesiser that has stopped sending is the common real failure, and a
    barge-in that waits for it to finish is a barge-in that never happens."""
    turn = SpeakingTurn(generation=1, produce=Synthesiser(stall=True))
    turn.start()
    await asyncio.sleep(0)

    await asyncio.wait_for(turn.cancel(), timeout=1.0)

    assert not turn.running


async def test_cancelling_before_starting_is_harmless() -> None:
    await SpeakingTurn(generation=1, produce=Synthesiser()).cancel()


async def test_the_producer_stops_early_rather_than_running_to_completion() -> None:
    """Bounded queue plus cancellation means an interrupted turn stops paying for
    audio nobody will hear. An unbounded queue would let it run to the end."""
    synthesiser = Synthesiser(chunks=500)
    turn = SpeakingTurn(generation=1, produce=synthesiser, queue_size=4)
    turn.start()
    while turn.queued == 0:
        await asyncio.sleep(0)

    await turn.cancel()

    assert synthesiser.produced < 500


async def test_a_stale_chunk_is_rejected_on_arrival() -> None:
    """M2.2, written to fail against an implementation that only stops the producer.

    A chunk already on the wire when the user interrupts will arrive. Nothing
    upstream can unsend it, so the receiver has to check which turn it belongs to,
    and this is that check.
    """
    in_flight = AudioChunk(generation=3, data=b"already sent")

    assert is_current(in_flight, 3)
    assert not is_current(in_flight, 4)


async def test_the_race_a_producer_only_fix_would_lose() -> None:
    """The whole scenario end to end: a chunk is dequeued, the user interrupts, and
    the chunk arrives afterwards. Silence is the only correct outcome."""
    turn = SpeakingTurn(generation=3, produce=Synthesiser(chunks=100))
    stream = turn.chunks()
    in_flight = await anext(stream)

    state = TurnState(state=State.SPEAKING, generation=3)
    await turn.cancel()
    new_generation = state.begin()

    played = [chunk for chunk in [in_flight] if is_current(chunk, new_generation)]

    assert state.interrupted_previous
    assert played == []


async def test_a_producer_failure_surfaces_to_the_consumer() -> None:
    """Raised in the caller's context rather than on the producer's task, where it
    would arrive as an unretrieved task exception with nothing to attribute it to."""

    async def broken() -> AsyncIterator[bytes]:
        yield b"first"
        raise RuntimeError("synthesis died")

    turn = SpeakingTurn(generation=1, produce=broken)

    with pytest.raises(RuntimeError):
        async for _chunk in turn.chunks():
            pass


async def test_a_failure_after_some_audio_keeps_the_audio() -> None:
    """Half an answer already spoken cannot be un-said, so the chunks delivered
    before the failure stay delivered. M3.3 is where the rest of the sentence gets
    a fallback voice."""

    async def breaks_late() -> AsyncIterator[bytes]:
        yield b"pehla"
        yield b"doosra"
        raise RuntimeError("synthesis died")

    turn = SpeakingTurn(generation=1, produce=breaks_late)
    heard: list[bytes] = []

    with pytest.raises(RuntimeError):
        async for chunk in turn.chunks():
            heard.append(chunk.data)

    assert heard == [b"pehla", b"doosra"]
