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


def chunked(reuse_final_within_ms: int = 700, **kwargs) -> tuple[ChunkedStt, FakeTranscriber]:
    transcriber = FakeTranscriber(**kwargs)
    return (
        ChunkedStt(
            transcriber,
            interval_ms=FRAME_MS * 5,
            reuse_final_within_ms=reuse_final_within_ms,
        ),
        transcriber,
    )


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
async def test_partial_indices_never_go_backwards(kind: str) -> None:
    """Monotonic rather than contiguous, because a reused final repeats the index of
    the partial it reuses. That repeat is the information: it says no further request
    was made."""
    stack = implementation(kind)

    seen = [partial async for partial in stack.stream(frames(20))]
    indices = [partial.index for partial in seen]

    assert indices == sorted(indices)
    assert indices[0] == 1


@pytest.mark.parametrize("kind", IMPLEMENTATIONS)
async def test_an_utterance_with_no_audio_still_produces_a_final(kind: str) -> None:
    """A muted microphone produces frames of nothing, or none at all, and the
    pipeline still needs to be told the recogniser is done. SPEC S6 depends on it."""
    stack = implementation(kind)

    seen = [partial async for partial in stack.stream(frames(0))]

    assert seen[-1].final


def test_the_default_interval_is_the_measured_one() -> None:
    """Pinned so a future edit changes this number on purpose rather than by
    reverting a merge conflict back to a guess.

    Moved from 400 to 600 on 2026-08-16 against live Groq, not on reasoning alone:
    a four-second utterance cost 11 requests and 7.76s of wall time in the STT
    stage at 400ms, 7 requests and 4.05s at 600, 5 requests and 3.25s at 800. 600
    was chosen over 800 because the win past it is small and the cost is not: a
    wider interval delays how soon `note_partial` can see a completed sentence,
    and that delay is bounded by `trailing_silence_ms` regardless, so 800 would
    double the delay against 400 for a small further cut in requests.
    """
    assert ChunkedStt(FakeTranscriber())._interval_ms == 600


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
    heard.

    Reuse is switched off so a final request actually happens. With it on there is
    nothing to fail, which is the point of the optimisation and would make this test
    pass without testing anything.
    """
    # 22 frames leaves a two-frame tail after the last partial, so the reuse test
    # below is a real branch rather than an arithmetic coincidence: with a tail of
    # exactly zero, "within 0ms" is satisfied and reuse happens anyway.
    stack, transcriber = chunked(reuse_final_within_ms=0)
    transcriber._fail_on = {5}

    with pytest.raises(SttError):
        [partial async for partial in stack.stream(frames(22))]


async def test_a_fresh_partial_becomes_the_final_without_another_request() -> None:
    """The STT tail term in SPEC's latency budget, which allows 100ms for it.

    The frame stream ends at the endpoint, so the audio not yet transcribed is the
    trailing silence that ended the turn. The last partial already contains every
    word spoken, and transcribing again spends a whole request, roughly 400ms on the
    critical path, to arrive at the same string. An unconditional final request
    cannot meet the budget, and this is what removes it.
    """
    stack, transcriber = chunked()

    seen = [partial async for partial in stack.stream(frames(20))]
    requests = len(transcriber.calls)

    assert seen[-1].final
    assert seen[-1].text == seen[-2].text
    assert requests == 4
    assert seen[-1].index == seen[-2].index


async def test_a_stale_tail_still_gets_a_final_request() -> None:
    """The other branch, and it is not optional. A tail longer than the silence that
    ended the turn means partials fell behind or failed, so words at the end were
    never transcribed and skipping the request would drop them."""
    stack, transcriber = chunked(reuse_final_within_ms=0)

    seen = [partial async for partial in stack.stream(frames(22))]

    assert seen[-1].final
    assert len(transcriber.calls) == 5
    assert seen[-1].index > seen[-2].index


async def test_a_stream_with_no_successful_partial_still_transcribes() -> None:
    """Every partial failing leaves nothing to reuse, and the turn still needs an
    answer. Reuse must not turn a stack with broken partials into a silent one."""
    stack, transcriber = chunked(fail_on={1, 2, 3, 4})

    seen = [partial async for partial in stack.stream(frames(20))]

    assert seen[-1].final
    assert seen[-1].text
    assert len(transcriber.calls) == 5


async def test_the_chunked_stack_says_it_is_not_streaming() -> None:
    """The whole reason the interface carries the flag. Calling both stacks
    "streaming STT" in the waterfall would be a lie by omission."""
    stack, _ = chunked()

    assert stack.streaming is False
    assert FakeStreamingStt().streaming is True
