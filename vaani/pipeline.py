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

import structlog

from vaani.confirm import needs_confirming, question
from vaani.endpoint import Endpointer
from vaani.grounding import check, figures
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
    ) -> None:
        self._stt = stt
        self._turn = turn
        self._tts = tts
        # Whether an unreadable figure costs a turn. On by default and switchable,
        # because it is an ablation arm rather than a setting: it spends latency to
        # avoid a wrong eligibility answer, and both sides of that trade get measured.
        self._confirm = confirm
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

        replies = self._turn.run(spoken.text, still_current)
        async for chunk in speak_as_they_arrive(
            replies, self._tts, self._voice, on_sentence=on_sentence, validate=grounded
        ):
            yield chunk

    async def _listen(self, frames: AsyncIterator[bytes]) -> str:
        """Transcribe while the user talks, and return the final transcript.

        Every partial is handed to the endpointer, which is what makes semantic
        endpointing possible at all: the decision to cut the wait short needs to know
        what has been said so far, and only the recogniser knows that.
        """
        final: str | None = None
        partials = 0

        async for partial in self._stt.stream(
            frames_until_endpoint(frames, self._endpointer)
        ):
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
