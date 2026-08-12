from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from vaani.protocol import FRAME_MS, SAMPLES_PER_FRAME
from vaani.stt import ChunkedStt, Partial, StreamingStt, SttError, Transcript

FRAME = b"\x01\x00" * SAMPLES_PER_FRAME


class FakeTranscriber:
    """Transcribes by length, so a longer buffer gives a longer transcript.

    Deterministic and offline. It also makes the growing-cost property of the
    chunked stack visible: each call receives the whole utterance so far.
    """

    name = "fake"
    streaming = False

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.calls: list[int] = []
        self._fail_on = fail_on or set()

    async def transcribe(self, pcm: bytes) -> Transcript:
        self.calls.append(len(pcm))
        if len(self.calls) in self._fail_on:
            raise SttError("provider returned 503")
        words = max(1, len(pcm) // (SAMPLES_PER_FRAME * 2 * 10))
        return Transcript(
            text=" ".join(["shabd"] * words), language="hi", provider=self.name, streaming=False
        )


class FakeStreamingStt:
    """A genuine streaming recogniser, for the contract test's other side."""

    name = "fake-streaming"
    streaming = True

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Partial]:
        index = 0
        heard = 0
        async for _frame in frames:
            heard += 1
            if heard % 5 == 0:
                index += 1
                yield Partial(text="shabd " * index, final=False, index=index)
        yield Partial(text="poora vaakya", final=True, index=index + 1)


async def frames(count: int) -> AsyncIterator[bytes]:
    for _ in range(count):
        yield FRAME


def chunked(**kwargs) -> tuple[ChunkedStt, FakeTranscriber]:
    transcriber = FakeTranscriber(**kwargs)
    return ChunkedStt(transcriber, interval_ms=FRAME_MS * 5), transcriber


IMPLEMENTATIONS = ["chunked", "streaming"]


def implementation(kind: str) -> StreamingStt:
    return chunked()[0] if kind == "chunked" else FakeStreamingStt()


@pytest.mark.parametrize("kind", IMPLEMENTATIONS)
async def test_a_partial_arrives_before_the_utterance_ends(kind: str) -> None:
    """The acceptance criterion for M1.2, asserted against both stacks.

    Ordering, not content: an implementation that transcribes once at the end and
    labels it a partial would satisfy any assertion about the text.
    """
    stack = implementation(kind)
    seen: list[Partial] = []

    async for partial in stack.stream(frames(20)):
        seen.append(partial)
        if not partial.final:
            break

    assert seen
    assert not seen[0].final


@pytest.mark.parametrize("kind", IMPLEMENTATIONS)
async def test_the_last_partial_is_the_final_one(kind: str) -> None:
    """Callers key off `final` to start the model turn. A stream that never sets it
    leaves the pipeline waiting on a recogniser that has already finished."""
    stack = implementation(kind)

    seen = [partial async for partial in stack.stream(frames(20))]

    assert seen[-1].final
    assert sum(1 for partial in seen if partial.final) == 1


@pytest.mark.parametrize("kind", IMPLEMENTATIONS)
async def test_partial_indices_count_up_without_gaps(kind: str) -> None:
    stack = implementation(kind)

    seen = [partial async for partial in stack.stream(frames(20))]

    assert [partial.index for partial in seen] == list(range(1, len(seen) + 1))


@pytest.mark.parametrize("kind", IMPLEMENTATIONS)
async def test_an_utterance_with_no_audio_still_produces_a_final(kind: str) -> None:
    """A muted microphone produces frames of nothing, or none at all, and the
    pipeline still needs to be told the recogniser is done. SPEC S6 depends on it."""
    stack = implementation(kind)

    seen = [partial async for partial in stack.stream(frames(0))]

    assert seen[-1].final


async def test_the_chunked_stack_re_sends_the_whole_utterance_each_time() -> None:
    """SPEC A4's cost, made visible. Each partial is a request over everything said
    so far, so the requests grow, and this is the difference the two waterfalls are
    supposed to show rather than hide."""
    stack, transcriber = chunked()

    [partial async for partial in stack.stream(frames(20))]

    assert transcriber.calls == sorted(transcriber.calls)
    assert len(transcriber.calls) > 1
    assert transcriber.calls[-1] == 20 * len(FRAME)


async def test_a_failed_partial_is_skipped_and_the_stream_continues() -> None:
    """A partial was going to be replaced by the next one anyway. Failing the turn
    over a discarded intermediate result trades a real answer for a provisional
    one."""
    stack, _ = chunked(fail_on={1})

    seen = [partial async for partial in stack.stream(frames(20))]

    assert seen[-1].final
    assert all(partial.index != 1 for partial in seen)


async def test_a_failed_final_is_not_swallowed() -> None:
    """The opposite rule, and the reason the two are separate branches. Returning an
    empty transcript here would send a confident answer to a question nobody
    heard."""
    stack, transcriber = chunked()
    transcriber._fail_on = {5}

    with pytest.raises(SttError):
        [partial async for partial in stack.stream(frames(20))]


async def test_the_chunked_stack_says_it_is_not_streaming() -> None:
    """The whole reason the interface carries the flag. Calling both stacks
    "streaming STT" in the waterfall would be a lie by omission."""
    stack, _ = chunked()

    assert stack.streaming is False
    assert FakeStreamingStt().streaming is True
