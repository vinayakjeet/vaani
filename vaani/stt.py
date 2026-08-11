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
from dataclasses import dataclass
from typing import Protocol

import httpx
import spanlight
import structlog

from vaani.audio import to_wav

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

        with spanlight.model_span(provider="groq", operation="transcribe") as span:
            span.set_attribute("vaani.stt.streaming", self.streaming)
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
            span.set_attribute("vaani.stt.final_chars", len(text))
            span.set_attribute("gen_ai.response.model", payload.get("model", self._model))

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
