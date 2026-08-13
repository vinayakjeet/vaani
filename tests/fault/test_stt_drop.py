from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from vaani.protocol import SAMPLES_PER_FRAME
from vaani.stt import (
    Partial,
    Reason,
    RecoveringStt,
    SttError,
    Transcript,
    recoveries,
)

FRAME = b"\x01\x00" * SAMPLES_PER_FRAME


class Batch:
    """The whole-file endpoint the recovery falls back to."""

    name = "batch"
    streaming = False

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[int] = []
        self._fail = fail

    async def transcribe(self, pcm: bytes) -> Transcript:
        self.calls.append(len(pcm))
        if self._fail:
            raise SttError("batch path is down too")
        return Transcript(
            text="poora sawaal", language="hi", provider=self.name, streaming=False
        )


class Streaming:
    """A streaming recogniser that can die partway through, or simply stop."""

    name = "socket"
    streaming = True

    def __init__(
        self,
        partials: int = 2,
        drop_after: int | None = None,
        never_final: bool = False,
        dead_on_arrival: bool = False,
    ) -> None:
        self._partials = partials
        self._drop_after = drop_after
        self._never_final = never_final
        self._dead_on_arrival = dead_on_arrival
        self.frames_seen = 0

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Partial]:
        index = 0
        if self._dead_on_arrival:
            # Refused the connection outright, before a single frame was read. The
            # only shape where recovery genuinely has nothing to work with.
            raise SttError("socket: connection refused")
            yield  # pragma: no cover
        async for _frame in frames:
            self.frames_seen += 1
            if self.frames_seen % 5:
                continue
            if self._drop_after is not None and index >= self._drop_after:
                raise SttError("socket: connection reset mid-utterance")
            index += 1
            if index <= self._partials:
                yield Partial(text=f"aadha {index}", final=False, index=index)
        if not self._never_final:
            yield Partial(text="socket ka final", final=True, index=index + 1)


async def frames(count: int) -> AsyncIterator[bytes]:
    for _ in range(count):
        yield FRAME


@pytest.fixture(autouse=True)
def clear_counters():
    recoveries.clear()
    yield
    recoveries.clear()


async def collect(stack) -> list[Partial]:
    return [partial async for partial in stack.stream(frames(30))]


async def test_a_healthy_stream_never_touches_the_batch_path() -> None:
    """The control. A wrapper that always fell back would pass every test below and
    would also throw away the streaming the project is built on."""
    batch = Batch()
    stack = RecoveringStt(Streaming(), batch)

    seen = await collect(stack)

    assert seen[-1].text == "socket ka final"
    assert batch.calls == []
    assert not recoveries


async def test_a_dropped_stream_is_retried_through_the_batch_path() -> None:
    """SPEC S5. The audio is still in memory, so it is sent again through the
    endpoint that takes a whole file, which is slower and works."""
    batch = Batch()
    stack = RecoveringStt(Streaming(drop_after=1), batch)

    seen = await collect(stack)

    assert seen[-1].final
    assert seen[-1].text == "poora sawaal"
    assert len(batch.calls) == 1


async def test_the_retry_sends_the_whole_utterance_not_just_the_tail() -> None:
    """The words spoken before the drop are the ones most likely to carry the
    question. Sending only what arrived after it would transcribe the second half of
    a sentence and answer that."""
    batch = Batch()
    stack = RecoveringStt(Streaming(drop_after=1), batch)

    await collect(stack)

    assert batch.calls[0] == 30 * len(FRAME)


async def test_the_partials_heard_before_the_drop_are_still_delivered() -> None:
    """They may already be on screen. Withdrawing them would make the pane flicker
    for a recovery the user did not need to know about."""
    stack = RecoveringStt(Streaming(drop_after=2), Batch())

    seen = await collect(stack)

    assert [partial.text for partial in seen if not partial.final] == ["aadha 1", "aadha 2"]


async def test_a_recovery_is_counted_with_a_reason() -> None:
    """A stack whose recoveries all fire is indistinguishable from a healthy one
    without a counter, and the reason is a closed set because a label built from an
    exception message has unbounded cardinality."""
    stack = RecoveringStt(Streaming(drop_after=1), Batch())

    await collect(stack)

    assert recoveries[Reason.STREAM_FAILED] == 1


async def test_a_stream_that_never_says_final_is_recovered_too() -> None:
    """Treating a stream that simply stopped as a complete transcript answers half a
    question. It is a recovery rather than a success with a shrug."""
    batch = Batch()
    stack = RecoveringStt(Streaming(never_final=True), batch)

    seen = await collect(stack)

    assert seen[-1].final
    assert seen[-1].text == "poora sawaal"
    assert recoveries[Reason.NO_FINAL] == 1


async def test_a_drop_before_any_audio_is_refused_rather_than_answered() -> None:
    """Nothing was heard, so there is nothing to send again. SPEC S6 wants a silent
    microphone diagnosed, and inventing a transcript here would answer a question
    nobody asked."""
    stack = RecoveringStt(Streaming(dead_on_arrival=True), Batch())

    with pytest.raises(SttError):
        [partial async for partial in stack.stream(frames(0))]


async def test_both_paths_failing_still_raises() -> None:
    """Recovery is not a way to hide an outage. The user has to be told."""
    stack = RecoveringStt(Streaming(drop_after=1), Batch(fail=True))

    with pytest.raises(SttError):
        await collect(stack)


async def test_the_final_index_moves_past_the_partials() -> None:
    """So a caller counting requests does not see the recovered final collide with
    the partial that came before it."""
    stack = RecoveringStt(Streaming(drop_after=2), Batch())

    seen = await collect(stack)

    assert seen[-1].index > seen[-2].index


async def test_the_wrapper_reports_that_it_is_not_streaming() -> None:
    """It can fall back to a batch request, so the waterfall must not read it as a
    streaming stage. SPEC A4's whole point is that the two are not comparable."""
    assert RecoveringStt(Streaming(), Batch()).streaming is False
