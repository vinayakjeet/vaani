"""Speech to text, behind one interface with two stacks underneath it.

The interface exists so `bench/waterfall.py` can swap the whole stack without
the pipeline knowing, and so the difference SPEC A4 names is visible at the seam
rather than buried: Groq takes a file and returns a transcript, while Sarvam
Saaras is a genuine streaming socket. Calling both of them "streaming STT" in
the waterfall would make the comparison a lie by omission, so the interface
reports which one it is.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx
import structlog

from vaani.audio import to_wav
from vaani.protocol import FRAME_MS
from vaani.spans import STT_REQUEST, STT_STREAM, stage_span

logger = structlog.get_logger(__name__)

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Whisper large v3 turbo. The non-turbo model is more accurate on Hindi and
# roughly three times slower, and this project's headline number is latency, so
# the tradeoff is recorded here and measured in the ablation rather than assumed.
GROQ_MODEL = "whisper-large-v3-turbo"

# Long enough for a 30 second utterance on a bad connection, short enough that a
# hung provider does not hold a turn open forever. The user is waiting with a
# microphone open, which is a harsher deadline than a web request.
TIMEOUT_SECONDS = 20.0


class SttError(Exception):
    """Transcription did not produce a usable transcript."""


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    provider: str
    # False for a provider that transcribes a finished recording, True for one
    # that emits partials while the user is still speaking. The waterfall reports
    # this beside the latency, because the two are not comparable stages.
    streaming: bool


class SttProvider(Protocol):
    name: str
    streaming: bool

    async def transcribe(self, pcm: bytes) -> Transcript: ...


@dataclass(frozen=True)
class Partial:
    """A transcript of everything heard so far, and whether it can still change."""

    text: str
    final: bool
    # Which request produced it, counting from one. On the chunked stack this is
    # also the number of provider calls the partial cost, which is the price SPEC
    # A4 says must not be glossed over.
    index: int


class StreamingStt(Protocol):
    """Anything that can emit transcripts while the user is still speaking.

    `streaming` says whether it genuinely is one. A chunked implementation
    satisfies this interface and is not a streaming recogniser, and the waterfall
    reports the difference rather than letting the shared interface imply
    equivalence.
    """

    name: str
    streaming: bool

    def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Partial]: ...


class ChunkedStt:
    """Partials by re-transcribing the audio so far, repeatedly. Not streaming.

    This is SPEC A4 made concrete. Groq's endpoint takes a finished file, so a
    partial on the free stack costs a whole request over the whole utterance to
    date, and the requests get more expensive as the utterance grows. That is
    likely the single largest difference between the two waterfalls, so it is a
    named class rather than a flag on the streaming one.

    A failed partial is skipped rather than raised. It is a guess that was going to
    be replaced by the next one, and failing the turn over a discarded intermediate
    result would trade a real answer for a provisional one. The final transcript is
    the one allowed to fail loudly.
    """

    streaming = False

    def __init__(self, transcriber: SttProvider, interval_ms: int = 600) -> None:
        self._transcriber = transcriber
        self._interval_ms = interval_ms
        self.name = f"chunked:{transcriber.name}"

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Partial]:
        # Wraps the recogniser only. The endpoint wait belongs to `vad.endpoint`,
        # and counting it here as well would put the same milliseconds in two
        # columns and make the waterfall add up to more than the turn.
        with stage_span(STT_STREAM, **{"vaani.stt.streaming": self.streaming}) as stage:
            buffer = bytearray()
            since_last = 0
            index = 0

            async for frame in frames:
                buffer += frame
                since_last += FRAME_MS
                if since_last < self._interval_ms:
                    continue

                since_last = 0
                index += 1
                try:
                    interim = await self._transcriber.transcribe(bytes(buffer))
                except SttError:
                    # Logged without the audio or the text. A partial that failed
                    # is worth counting, because a stack whose partials all fail is
                    # indistinguishable from one with no partials at all.
                    logger.warning("stt.partial_failed", provider=self.name, index=index)
                    continue
                yield Partial(text=interim.text, final=False, index=index)

            index += 1
            final = await self._transcriber.transcribe(bytes(buffer))
            stage.record(
                **{
                    "vaani.stt.partials": index - 1,
                    "vaani.stt.final_chars": len(final.text),
                }
            )
            logger.info("stt.stream_done", provider=self.name, requests=index)
            yield Partial(text=final.text, final=True, index=index)


class GroqWhisper:
    """Chunked transcription, not streaming. See SPEC A4.

    Groq's transcription endpoint takes a complete audio file. Partial results on
    this stack come from sending the audio so far, repeatedly, which is a
    different thing from a streaming recogniser and costs a request per partial.
    M1.2 builds that; this is the whole-utterance form M0 needs.
    """

    name = "groq-whisper"
    streaming = False

    def __init__(self, api_key: str | None = None, model: str = GROQ_MODEL) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._model = model

    async def transcribe(self, pcm: bytes) -> Transcript:
        if not self._api_key:
            raise SttError("GROQ_API_KEY is not set")

        with stage_span(
            STT_REQUEST,
            **{
                "gen_ai.system": "groq",
                "vaani.stt.audio_ms": round(len(pcm) / 32),
            },
        ) as stage:
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        GROQ_TRANSCRIBE_URL,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        files={"file": ("utterance.wav", to_wav(pcm), "audio/wav")},
                        data={"model": self._model, "response_format": "verbose_json"},
                    )
            except httpx.HTTPError as exc:
                raise SttError(f"{type(exc).__name__}") from exc

            if response.status_code != 200:
                # The status, never the body. A provider error body can quote the
                # audio's transcript back, and that is user speech going into a
                # log line this project promises never to write.
                raise SttError(f"provider returned {response.status_code}")

            payload = response.json()
            text = (payload.get("text") or "").strip()
            stage.record(
                **{
                    "vaani.stt.final_chars": len(text),
                    "gen_ai.response.model": payload.get("model", self._model),
                }
            )

        if not text:
            raise SttError("empty transcript")

        # The transcript itself is never logged and never put on a span. It is
        # user speech, and SPEC's threat model says it does not leave the process.
        logger.info("stt.done", provider=self.name, chars=len(text))
        return Transcript(
            text=text,
            language=payload.get("language"),
            provider=self.name,
            streaming=self.streaming,
        )
