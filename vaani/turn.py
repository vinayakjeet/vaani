"""One turn, unstreamed. The baseline the ablation measures against.

Wait for the whole utterance, transcribe all of it, wait for the whole reply,
synthesise all of it, then play. Every stage waits for the one before it to
finish, which is the slowest correct arrangement and exactly why it is worth
building first: "streaming bought 400ms" is meaningless without a measured
before.

This module is not replaced in M1. It stays, and the streamed path is added
beside it, because `bench/ablation.py` needs to run both on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from llm import ChatClient, ChatMessage
from vaani.stt import SttProvider, Transcript
from vaani.tts import VOICE_HI, TtsProvider

logger = structlog.get_logger(__name__)

# Hinglish is the default because it is what the eval set is half made of, and
# because a model told to answer in Hindi will transliterate English scheme names
# into Devanagari, which reads as wrong to the people who use them.
SYSTEM_PROMPT = """You answer questions about Indian government welfare schemes.

Reply in the language the user spoke. If they mixed Hindi and English, mix them
back the same way: keep scheme names, numbers and technical terms in the script
the user used them in.

Answer in at most three short sentences. This is spoken aloud, so a listener
cannot re-read it, and a long answer is one they will interrupt.

If you do not know whether someone is eligible, say so and say what would decide
it. Never invent a scheme, an income limit, or a deadline."""


@dataclass(frozen=True)
class TurnResult:
    transcript: Transcript
    reply: str
    audio: bytes
    audio_mime: str


class Turn:
    """A single question and answer, with no overlap between the stages."""

    def __init__(
        self,
        stt: SttProvider,
        tts: TtsProvider,
        llm: ChatClient | None = None,
        provider: str = "groq",
        voice: str = VOICE_HI,
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._llm = llm or ChatClient()
        self._provider = provider
        self._voice = voice

    async def run(self, pcm: bytes) -> TurnResult:
        transcript = await self._stt.transcribe(pcm)
        reply = await self._answer(transcript.text)

        audio = bytearray()
        async for chunk in self._tts.synthesize(reply, self._voice):
            audio.extend(chunk)

        return TurnResult(
            transcript=transcript,
            reply=reply,
            audio=bytes(audio),
            audio_mime=self._tts.mime,
        )

    async def _answer(self, question: str) -> str:
        """The model call, through the chassis client against Tollgate's contract.

        No `model_span` here. `ChatClient.complete` opens its own, and wrapping it
        again would nest two model spans per call and record the usage twice,
        which the cost detector reads as double the spend.
        """
        response = await self._llm.complete(
            self._provider,
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=question),
            ],
        )
        reply = response.text.strip()
        if not reply:
            raise ValueError("model returned an empty reply")
        # Length, never the text. The reply is about a real person's eligibility.
        logger.info("turn.answered", chars=len(reply))
        return reply
