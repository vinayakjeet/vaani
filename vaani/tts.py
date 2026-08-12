"""Text to speech, streamed as chunks so playback can start early.

Chunked from the start even in M0, where nothing overlaps yet, because the
interface is what M1.4's first-sentence flush and M3.3's mid-utterance failover
are built on. An interface that returns one finished audio blob would have to be
replaced rather than extended, and the fallback path would never have been
exercised until the day it was needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

import spanlight
import structlog

from vaani.sentences import from_stream

logger = structlog.get_logger(__name__)

# Hindi voices. Neural voices from this provider handle Devanagari and read
# Latin-script Hinglish acceptably, which matters because half the eval set is
# code-mixed and a voice that spells out English words ruins the demo.
VOICE_HI = "hi-IN-SwaraNeural"
VOICE_EN_IN = "en-IN-NeerjaNeural"

# Output is MP3 from this provider. The browser decodes it; the alternative is
# raw PCM at three times the bytes over a socket that is also carrying the user's
# microphone in the other direction.
AUDIO_MIME = "audio/mpeg"


class TtsError(Exception):
    """Synthesis failed or produced nothing playable."""


async def speak_as_they_arrive(
    tokens: AsyncIterator[str], tts: TtsProvider, voice: str = VOICE_HI
) -> AsyncIterator[bytes]:
    """Audio for each sentence as soon as the token stream completes that sentence.

    The overlap the milestone is named for: sentence one is being synthesised, and
    then played, while the model is still writing sentence three. The unstreamed
    baseline in `vaani/turn.py` waits for the whole reply before it starts, and the
    gap between the two is a row in the ablation.

    Each sentence is its own synthesis request, so the audio arrives as a sequence
    of separate encoded streams rather than one. Browsers play them back to back,
    and whether the joins are audible is an open question the M1.8 note already
    raises. It is a quality cost paid for milliseconds, so the ablation reports it
    rather than only the milliseconds.
    """
    sentences = 0
    async for sentence in from_stream(tokens):
        sentences += 1
        async for chunk in tts.synthesize(sentence, voice):
            yield chunk

    logger.info("tts.stream_done", provider=tts.name, sentences=sentences)


class TtsProvider(Protocol):
    name: str
    mime: str

    def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]: ...


class EdgeTts:
    """The free stack's voice. No key, no quota, and no service guarantee.

    Being unofficial is the point of M3.3: this is the provider most likely to
    fail mid-utterance in the real world, so the failover path it forces is not
    a hypothetical.
    """

    name = "edge-tts"
    mime = AUDIO_MIME

    async def synthesize(self, text: str, voice: str = VOICE_HI) -> AsyncIterator[bytes]:
        import edge_tts

        with spanlight.model_span(provider=self.name, operation="synthesize") as span:
            span.set_attribute("vaani.tts.voice", voice)
            span.set_attribute("vaani.tts.chars", len(text))

            first = True
            chunks = 0
            try:
                async for chunk in edge_tts.Communicate(text, voice).stream():
                    if chunk["type"] != "audio":
                        continue
                    if first:
                        # The number the whole project is about. Time to first
                        # audio is what a listener experiences as responsiveness,
                        # and it is not the same as time to a finished answer.
                        span.add_event("vaani.tts.first_chunk")
                        first = False
                    chunks += 1
                    yield chunk["data"]
            except Exception as exc:
                raise TtsError(f"{type(exc).__name__}") from exc

            span.set_attribute("vaani.tts.chunks", chunks)

        if chunks == 0:
            raise TtsError("no audio produced")
        logger.info("tts.done", provider=self.name, voice=voice, chunks=chunks)
