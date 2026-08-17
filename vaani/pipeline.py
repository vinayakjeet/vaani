"""The overlapped path, composed. Capture to audio, with the stages overlapping.

The unstreamed baseline in `vaani/turn.py` stays exactly as it is and this is built
beside it, because `bench/ablation.py` runs both and "streaming bought 400ms" means
nothing without a measured before.

What overlaps, and what cannot:

- Transcription runs while the user is still speaking, so the final transcript is a
  string that already exists when the endpoint fires rather than a request made
  after it.
- Partials feed the endpointer, so the wait can be cut when what was said already
  sounds like a finished question. That is the same technique twice: a fresh partial
  shortens the endpoint wait and then serves as the final transcript.
- Synthesis of sentence one runs while the model writes sentence three.

The model cannot overlap with transcription, and no amount of design changes that.
It needs a transcript to answer, so the only way to move that boundary is to answer
a guess, which is what speculative prefill does and why it is a separate technique
with its own row in the ablation rather than part of this.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing

import structlog

from vaani.confirm import needs_confirming, question
from vaani.endpoint import Endpointer
from vaani.grounding import check, figures
from vaani.history import Conversation
from vaani.llm_turn import StreamedTurn
from vaani.numerals import normalise
from vaani.stt import StreamingStt, SttError
from vaani.tts import VOICE_HI, TtsProvider, speak_as_they_arrive

logger = structlog.get_logger(__name__)


async def frames_until_endpoint(
    frames: AsyncIterator[bytes], endpointer: Endpointer
) -> AsyncIterator[bytes]:
    """Pass frames through, stopping once the endpointer says the turn ended.

    The frame is yielded before the endpoint is tested, so the frame that triggered
    the endpoint is still transcribed. Testing first would drop the last 20ms of
    every utterance, which is a clipped final word and reads as a bad recogniser.
    """
    async for frame in frames:
        yield frame
        if endpointer.accept(frame):
            return


class StreamingPipeline:
    """One turn, overlapped, from audio frames to audio chunks."""

    def __init__(
        self,
        stt: StreamingStt,
        turn: StreamedTurn,
        tts: TtsProvider,
        endpointer: Endpointer | None = None,
        voice: str = VOICE_HI,
        confirm: bool = True,
        history: Conversation | None = None,
    ) -> None:
        self._stt = stt
        self._turn = turn
        self._tts = tts
        # Whether an unreadable figure costs a turn. On by default and switchable,
        # because it is an ablation arm rather than a setting: it spends latency to
        # avoid a wrong eligibility answer, and both sides of that trade get measured.
        self._confirm = confirm
        # Shared with the session, which owns the other half of the record: this stages
        # what the model produced and the session decides what was heard, because only
        # the transport knows which audio actually went out.
        self._history = history if history is not None else Conversation()
        # Semantic endpointing on by default here and off in the baseline, which is
        # what keeps the ablation comparing the technique against its absence rather
        # than against itself.
        self._endpointer = endpointer or Endpointer(semantic=True)
        self._voice = voice

    async def run(
        self,
        frames: AsyncIterator[bytes],
        still_current: Callable[[], bool] | None = None,
        on_transcript: Callable[[str], Awaitable[None]] | None = None,
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncIterator[bytes]:
        heard = await self._listen(frames)

        # Numbers become digits here, in code, before the model reads them. The
        # applicant's own figure is the input to an eligibility comparison, and a model
        # asked to both normalise and reason about it will occasionally do one of those
        # jobs to the other's cost.
        spoken = normalise(heard)
        if spoken.unresolved:
            logger.info("pipeline.numbers_unresolved", count=len(spoken.unresolved))

        if on_transcript is not None:
            await on_transcript(spoken.text)

        if self._confirm and needs_confirming(spoken):
            # The turn ends here. Handing an unreadable amount to the model means either
            # the model normalises it, which is the job `vaani.numerals` exists to take
            # away, or the figure is simply absent from the prompt and the reply is built
            # on an assumption nobody made deliberately.
            ask = question()
            if on_sentence is not None:
                await on_sentence(ask)
            async for chunk in self._tts.synthesize(ask, self._voice, index=1):
                yield chunk
            return

        # What the reply may state: what the user said, plus whatever the tools return
        # as the turn runs. The set is read per sentence rather than captured now,
        # because the tool round happens after this line and before the first sentence
        # that could quote it.
        said_by_user = figures(spoken.text)

        def grounded(sentence: str) -> str:
            return check(sentence, said_by_user | self._turn.sourced)

        # Read at the moment the turn starts rather than held from construction, because
        # the session has been writing into it: the previous turn's reply was committed,
        # truncated or dropped between then and now depending on what was heard.
        replies = self._turn.run(spoken.text, still_current, history=self._history.for_model())
        # `aclosing`, for the same reason as `_listen`'s: this whole method is the
        # `answer` that `vaani.budget.speak_within` can close early (a barge-in, or
        # a caller that simply stops reading), and `speak_as_they_arrive` holds a
        # `stage_span` open across its own `yield`s further down. Being idle at
        # `yield chunk` below when that happens is enough to abandon it: the
        # closure reaches this frame, but nothing then tells the generator this
        # frame is holding onto that it should close too.
        spoken_audio = speak_as_they_arrive(
            replies, self._tts, self._voice, on_sentence=on_sentence, validate=grounded
        )
        async with aclosing(spoken_audio):
            async for chunk in spoken_audio:
                yield chunk

    async def _listen(self, frames: AsyncIterator[bytes]) -> str:
        """Transcribe while the user talks, and return the final transcript.

        Every partial is handed to the endpointer, which is what makes semantic
        endpointing possible at all: the decision to cut the wait short needs to know
        what has been said so far, and only the recogniser knows that.
        """
        final: str | None = None
        partials = 0

        # `aclosing` rather than a bare `async for`, because the loop below breaks
        # the instant the final partial arrives instead of exhausting the generator.
        # `ChunkedStt.stream` holds its `stage_span` open across every `yield`, so a
        # generator abandoned mid-span never runs that span's `__exit__` in the task
        # that opened it; it runs whenever the garbage collector gets to it, in
        # whatever context that happens to be, and OpenTelemetry's context token
        # detach then raises "was created in a different Context". Explicitly
        # closing here throws `GeneratorExit` at the suspended yield immediately,
        # in this task, so the span closes where it opened.
        stream = self._stt.stream(frames_until_endpoint(frames, self._endpointer))
        async with aclosing(stream):
            async for partial in stream:
                if partial.final:
                    final = partial.text
                    break
                partials += 1
                self._endpointer.note_partial(partial.text)

        if not final or not final.strip():
            # Never an empty string onward. A model handed nothing answers something,
            # and the user hears a confident reply to a question they did not ask.
            raise SttError("empty transcript")

        # Counts, never the text. The transcript is user speech about a real person's
        # eligibility and does not leave the process.
        logger.info("pipeline.heard", partials=partials, chars=len(final))
        return final
