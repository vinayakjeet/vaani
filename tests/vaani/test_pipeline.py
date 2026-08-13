from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from vaani.endpoint import Endpointer
from vaani.grounding import REFUSED
from vaani.llm_turn import StreamedTurn
from vaani.pipeline import StreamingPipeline, frames_until_endpoint
from vaani.stt import Partial, SttError

from .test_endpoint import SILENCE, SPEECH
from .test_llm_turn import ScriptedClient, text
from .test_streamed_speech import RecordingTts


class ScriptedStt:
    """Emits scripted partials, then a final, recording the frames it was given."""

    name = "scripted"
    streaming = True

    def __init__(self, *texts: str, final: str | None = None) -> None:
        self._texts = texts
        self._final = final if final is not None else (texts[-1] if texts else "")
        self.frames_seen = 0

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Partial]:
        index = 0
        async for _frame in frames:
            self.frames_seen += 1
            if index < len(self._texts):
                yield Partial(text=self._texts[index], final=False, index=index + 1)
                index += 1
        yield Partial(text=self._final, final=True, index=index + 1)


async def frames(*sequence: bytes) -> AsyncIterator[bytes]:
    for frame in sequence:
        yield frame


def pipeline(stt, *rounds, endpointer=None) -> tuple[StreamingPipeline, RecordingTts]:
    tts = RecordingTts()
    turn = StreamedTurn(llm=ScriptedClient(*rounds), provider="scripted")
    return (
        StreamingPipeline(stt=stt, turn=turn, tts=tts, endpointer=endpointer),
        tts,
    )


async def test_the_frame_that_triggers_the_endpoint_is_still_transcribed() -> None:
    """Testing the endpoint before yielding would drop the last 20ms of every
    utterance, which is a clipped final word and reads as a bad recogniser."""
    endpointer = Endpointer()
    speech = [SPEECH] * 30
    silence = [SILENCE] * (endpointer.trailing_silence_ms // 20)

    passed = [frame async for frame in frames_until_endpoint(frames(*speech, *silence), endpointer)]

    assert len(passed) == len(speech) + len(silence)


async def test_frames_after_the_endpoint_are_not_forwarded() -> None:
    """Otherwise the recogniser keeps transcribing into the next turn's audio."""
    endpointer = Endpointer()
    speech = [SPEECH] * 30
    silence = [SILENCE] * (endpointer.trailing_silence_ms // 20)
    extra = [SPEECH] * 10

    passed = [
        frame
        async for frame in frames_until_endpoint(frames(*speech, *silence, *extra), endpointer)
    ]

    assert len(passed) == len(speech) + len(silence)


async def test_a_turn_runs_from_frames_to_audio() -> None:
    stt = ScriptedStt("kya main", final="kya main eligible hoon")
    streamed, tts = pipeline(stt, text("Haan, aap eligible hain."))

    audio = [chunk async for chunk in streamed.run(frames(SPEECH, SPEECH, SILENCE))]

    assert tts.said == ["Haan, aap eligible hain."]
    assert audio


async def test_partials_are_fed_to_the_endpointer() -> None:
    """The whole reason semantic endpointing can work at all: the decision to cut
    the wait short needs to know what has been said, and only the recogniser does."""
    endpointer = Endpointer(semantic=True)
    stt = ScriptedStt("mujhe ghar chahiye", final="mujhe ghar chahiye")
    streamed, _ = pipeline(stt, text("ok"), endpointer=endpointer)

    async for _chunk in streamed.run(frames(SPEECH, SPEECH, SILENCE)):
        pass

    assert endpointer.silence_needed_ms == endpointer.early_silence_ms


async def test_an_incomplete_partial_leaves_the_full_wait_in_place() -> None:
    endpointer = Endpointer(semantic=True)
    stt = ScriptedStt("mera income", final="mera income")
    streamed, _ = pipeline(stt, text("ok"), endpointer=endpointer)

    async for _chunk in streamed.run(frames(SPEECH, SPEECH, SILENCE)):
        pass

    assert endpointer.silence_needed_ms == endpointer.trailing_silence_ms


async def test_an_empty_transcript_is_refused_rather_than_answered() -> None:
    """A model handed nothing answers something, and the user hears a confident
    reply to a question they did not ask. SPEC S6 depends on this too: a muted
    microphone must be diagnosed, not answered."""
    streamed, tts = pipeline(ScriptedStt(final="   "), text("this must never be said"))

    with pytest.raises(SttError):
        async for _chunk in streamed.run(frames(SPEECH, SILENCE)):
            pass

    assert tts.said == []


async def test_synthesis_starts_before_the_reply_is_finished() -> None:
    """The overlap the whole path exists for. Asserting only the sentences would
    pass against an implementation that buffers the reply and splits it at the end,
    which is the baseline this is measured against."""
    stt = ScriptedStt(final="kya main eligible hoon")
    streamed, tts = pipeline(
        stt, text("Haan aap ", "eligible ", "hain. ", "Aapko ", "6000 ", "milega.")
    )

    async for _chunk in streamed.run(frames(SPEECH, SILENCE)):
        # The first audio chunk is out, and at that point only the first sentence
        # can have been synthesised.
        break

    assert tts.said == ["Haan aap eligible hain."]


async def test_the_transcript_reaches_the_model() -> None:
    """Not merely that a reply was produced. A pipeline that dropped the transcript
    would still produce one, and it would be invented."""
    stt = ScriptedStt(final="mujhe ghar chahiye")
    tts = RecordingTts()
    client = ScriptedClient(text("ok"))
    streamed = StreamingPipeline(
        stt=stt, turn=StreamedTurn(llm=client, provider="scripted"), tts=tts
    )

    async for _chunk in streamed.run(frames(SPEECH, SILENCE)):
        pass

    user_messages = [m for m in client.seen[0] if m.role == "user"]

    assert [m.content for m in user_messages] == ["mujhe ghar chahiye"]


async def test_the_recogniser_actually_receives_the_frames() -> None:
    """Guards the tests above, which all assert on what came out. A pipeline that
    fed the recogniser nothing would still return its scripted final."""
    stt = ScriptedStt(final="kya main eligible hoon")
    streamed, _ = pipeline(stt, text("ok"))

    async for _chunk in streamed.run(frames(SPEECH, SPEECH, SPEECH, SILENCE)):
        pass

    assert stt.frames_seen == 4


async def test_the_transcript_the_model_sees_has_its_numbers_as_digits() -> None:
    """The applicant's own figure is the input to an eligibility comparison, so it is
    normalised in code before the model reads it rather than by the model while it is
    also reasoning about it."""
    stt = ScriptedStt(final="meri aay pachaas hazaar hai")
    flow, _tts = pipeline(stt, text("Theek hai."))

    seen: list[str] = []

    async def note(transcript: str) -> None:
        seen.append(transcript)

    async for _chunk in flow.run(frames(SPEECH), on_transcript=note):
        pass

    assert seen == ["meri aay 50000 hai"]


async def test_a_figure_the_tools_never_returned_is_refused_before_synthesis() -> None:
    """The guardrail on the real path, not beside it. Audio cannot be un-said, so a
    check that runs on a finished reply runs after the wrong number was spoken."""
    stt = ScriptedStt(final="pm jay ki limit kya hai")
    flow, tts = pipeline(stt, text("Ayushman Bharat ki limit 800000 hai."))

    async for _chunk in flow.run(frames(SPEECH)):
        pass

    assert tts.said == [REFUSED]


async def test_a_figure_the_user_said_may_be_repeated_back() -> None:
    """Otherwise a confirmation sentence is impossible, and confirming a high-stakes
    figure is cheaper than one wrong eligibility answer."""
    stt = ScriptedStt(final="meri aay teen lakh hai")
    flow, tts = pipeline(stt, text("Aapki aay 300000 hai, theek hai."))

    async for _chunk in flow.run(frames(SPEECH)):
        pass

    assert tts.said == ["Aapki aay 300000 hai, theek hai."]
