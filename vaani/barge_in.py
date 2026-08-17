"""Producing a turn's audio in a way that can be stopped mid-sentence.

Barge-in has two halves and only one of them is stopping the producer. The other
is that audio already handed onward cannot be recalled: a chunk on the wire when
the user interrupts will arrive, and the only thing that stops it being played is
the receiver checking which turn it belongs to. So every chunk carries its
generation, and both ends drop what is stale rather than trusting it to have been
stopped in time.

The queue is bounded on purpose. An unbounded one lets a fast synthesiser run
ahead of playback by whole sentences, and then an interruption has megabytes of
already-generated audio to throw away, which is both wasted provider spend and a
longer window in which the wrong thing can be spoken. Backpressure keeps the
amount of future the pipeline has committed to small enough to abandon cheaply.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# How many chunks may sit ahead of playback. Small enough that cancelling wastes
# little, large enough that the producer is not stalled by ordinary scheduling.
QUEUE_SIZE = 8


@dataclass(frozen=True)
class AudioChunk:
    generation: int
    data: bytes


def is_current(chunk: AudioChunk, generation: int) -> bool:
    """Whether this chunk still belongs to the turn in progress.

    The check the receiver makes on arrival. An implementation that only stops the
    producer passes every test that measures the producer and still plays the
    chunk that was already in flight, which is the failure the user hears.
    """
    return chunk.generation == generation


class SpeakingTurn:
    """One turn's audio production, as something that can be abandoned.

    `produce` is called once and its iterator is closed on cancellation, so a
    provider connection is released rather than left to a garbage collector that
    runs whenever it likes.
    """

    def __init__(
        self,
        generation: int,
        produce: Callable[[], AsyncIterator[bytes]],
        queue_size: int = QUEUE_SIZE,
    ) -> None:
        self.generation = generation
        self._produce = produce
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def cancelled(self) -> bool:
        """Whether production was stopped rather than reaching its own end.

        `chunks()` cannot tell a caller this on its own: `_pump` delivers the
        same `None` sentinel either way, deliberately, so a consumer's `async
        for` sees an ordinary end of iteration whether production finished or
        was cut off. That is right for `chunks()` itself, which only promises
        audio in order; it is wrong for a caller that needs to know which one
        happened, and asking `chunks()`'s own iterator cannot answer it.
        """
        return self._task is not None and self._task.cancelled()

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        source = self._produce()
        cancelled = False
        try:
            async for data in source:
                await self._queue.put(AudioChunk(generation=self.generation, data=data))
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            # Kept rather than raised here. This runs on its own task, so raising
            # would surface as an unretrieved task exception long after the turn,
            # with nothing to attribute it to. The consumer re-raises it in the
            # caller's own context instead.
            self._failure = exc
        finally:
            # Closing the source is the resource release. Cancelling the task alone
            # leaves the provider's stream to be closed whenever the loop next
            # collects it, which on a free tier is a connection held open against a
            # quota somebody is counting.
            await source.aclose()

            # The sentinel releases a consumer waiting on `get`, and how it is
            # delivered has to differ by path. On the cancelled path the queue is
            # usually full, which is what backpressure means, and awaiting a put
            # there deadlocks: the only consumer is the one being interrupted, so
            # nothing will ever make room. Dropping the queued audio first is
            # correct anyway, since none of it will be played.
            if cancelled:
                while not self._queue.empty():
                    self._queue.get_nowait()
                self._queue.put_nowait(None)
            else:
                await self._queue.put(None)

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Audio in order, ending when production finishes or is cancelled."""
        if self._task is None:
            self.start()

        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

        if self._failure is not None:
            raise self._failure

    async def cancel(self) -> None:
        """Stop producing, drop what is queued, and wait for the task to be gone.

        Awaiting the task is the part that matters for the acceptance criterion. A
        cancel that returns before the producer has actually stopped leaves a task
        writing into a queue nobody reads, which is an orphan whether or not it is
        called one.
        """
        if self._task is None:
            return

        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

        dropped = 0
        while not self._queue.empty():
            self._queue.get_nowait()
            dropped += 1

        logger.info("bargein.cancelled", generation=self.generation, dropped_chunks=dropped)
