"""Sarvam Saaras and Bulbul, the second waterfall column. See SPEC A4, M4.4.

Bench-only. Never imported by `app/routers/voice.py`: the deployed free stack is
Groq and EdgeTts, and this exists so `bench/waterfall.py` can run the same corpus
through a genuinely streaming STT socket and a paid, quota-tracked TTS voice,
making SPEC A4's chunked-versus-streaming difference a measured comparison rather
than an assertion. QUOTAS.md's own rule stands: nothing here runs except from a
bench script asking for it by name, and the balance is read before and after any
real call against it.

The vendor's own SDK (`sarvamai`) rather than a hand-rolled WebSocket client. The
streaming STT socket's exact wire framing is documented across two overlapping,
partially-inconsistent pages at the time this was written, and getting a framing
detail wrong against a 100-credit balance is not a mistake worth risking twice to
save a dependency.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import time
from collections.abc import AsyncIterator

import structlog

from vaani.audio import to_wav
from vaani.protocol import FRAME_MS, SAMPLE_RATE
from vaani.spans import STT_STREAM, TTS_SYNTHESIZE, stage_span
from vaani.stt import Partial, SttError
from vaani.tts import AUDIO_MIME, TtsError

logger = structlog.get_logger(__name__)

SAARAS_MODEL = "saaras:v3"
BULBUL_MODEL = "bulbul:v2"
VOICE_HI_BULBUL = "anushka"

# 64kbps MP3, chosen from Sarvam's own fixed bitrate set (32/64/96/128/192k); the
# closest to EdgeTts's 48kbps without going under it, so a turn compared across
# stacks is never timed against a lower-fidelity encode on this side by accident.
BULBUL_BITRATE = "64k"
BULBUL_BYTES_PER_SECOND = 64_000 // 8

# How much new audio accumulates between sends. Same reasoning M1.2's interval
# tuning already recorded for the chunked stack: too short wastes requests on
# audio that has not changed enough to move the transcript, too long delays how
# soon a finished sentence can be confirmed. Unmeasured against Sarvam specifically
# because that measurement costs real credits; started at the free stack's own
# tuned value rather than a fresh guess, and is a candidate to revisit once a
# waterfall run against this stack exists to tune it against.
INTERVAL_MS = 600

# How long to keep listening for a trailing response once every frame has been
# sent and the stream has been flushed, and how long to wait for the very first
# one. Real, not provisional-and-forgotten: Saaras runs its own VAD server-side
# (QUOTAS.md's own note on this) and decides on its own schedule when a
# transcript is worth emitting, which this project's request does not control
# and cannot assume is one response per message sent. The grace window resets
# every time a new response arrives, so it bounds silence, not the true length
# of a reply.
RESPONSE_GRACE_S = 8.0


def _transcript_of(message: object) -> str | None:
    """Pull the transcript out of a streaming response, or None for anything
    that is not a transcript: an `events` frame (VAD signal) or an `error`."""
    kind = getattr(message, "type", None)
    if kind != "data":
        return None
    data = getattr(message, "data", None)
    return getattr(data, "transcript", None)


class SarvamSaaras:
    """Genuinely streaming STT, unlike `ChunkedStt`: one persistent socket per
    turn, fed incremental audio rather than the whole buffer on every request.

    `connect` is the injection seam for tests: a callable returning the async
    context manager `AsyncSarvamAI.speech_to_text_streaming.connect(...)` yields.
    Real construction happens lazily inside `stream`, not in `__init__`, so
    importing this module never requires the SDK's own network-touching client
    to exist yet and a missing API key is reported as `SttError` like every
    other provider here, not an import-time crash.
    """

    name = "sarvam-saaras"
    streaming = True

    def __init__(
        self,
        api_key: str | None = None,
        language_code: str = "hi-IN",
        interval_ms: int = INTERVAL_MS,
        connect=None,
    ) -> None:
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self._language_code = language_code
        self._interval_ms = interval_ms
        self._connect = connect

    def _connection(self):
        if self._connect is not None:
            return self._connect()
        from sarvamai import AsyncSarvamAI

        client = AsyncSarvamAI(api_subscription_key=self._api_key)
        return client.speech_to_text_streaming.connect(
            language_code=self._language_code,
            model=SAARAS_MODEL,
            mode="transcribe",
        )

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Partial]:
        """Two concurrent halves, not a request-then-its-response loop.

        The first version of this sent one message and awaited one response per
        message, which hung against the real socket: Saaras runs its own VAD and
        decides on its own schedule when a transcript is worth emitting, not once
        per message received, so a caller that waits for "the" response to a send
        can wait for a response that was never coming from that send at all. The
        sender pushes audio without waiting on anything; a second task pumps
        every response that arrives, in the order it arrives, onto a queue this
        generator reads from and yields as partials. `RESPONSE_GRACE_S` bounds
        how long this waits for one more response once sending is finished,
        resetting on every response actually received, so it is a silence bound
        rather than a guess at the true length of a reply.
        """
        if not self._api_key:
            raise SttError("SARVAM_API_KEY is not set")

        with stage_span(STT_STREAM, **{"vaani.stt.streaming": True}) as stage:
            index = 0
            last_text: str | None = None

            try:
                async with self._connection() as ws:
                    queue: asyncio.Queue[object] = asyncio.Queue()

                    async def pump() -> None:
                        try:
                            async for message in ws:
                                await queue.put(message)
                        except Exception as exc:  # pragma: no branch
                            await queue.put(exc)

                    pump_task = asyncio.create_task(pump())
                    sender_task = asyncio.create_task(self._send_all(ws, frames))
                    try:
                        await sender_task

                        deadline = time.monotonic() + RESPONSE_GRACE_S
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            try:
                                item = await asyncio.wait_for(queue.get(), timeout=remaining)
                            except TimeoutError:
                                break
                            if isinstance(item, Exception):
                                raise item
                            text = _transcript_of(item)
                            if text is None:
                                continue
                            index += 1
                            last_text = text
                            deadline = time.monotonic() + RESPONSE_GRACE_S
                            yield Partial(text=text, final=False, index=index)
                    finally:
                        pump_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await pump_task
            except SttError:
                raise
            except Exception as exc:
                raise SttError(f"sarvam-saaras: {type(exc).__name__}: {exc}") from exc

            index += 1
            final_text = last_text or ""
            stage.record(
                **{"vaani.stt.partials": index - 1, "vaani.stt.final_chars": len(final_text)}
            )
            logger.info("stt.stream_done", provider=self.name, requests=index)
            yield Partial(text=final_text, final=True, index=index)

    async def _send_all(self, ws, frames: AsyncIterator[bytes]) -> None:
        buffer = bytearray()
        since_last = 0
        async for frame in frames:
            buffer += frame
            since_last += FRAME_MS
            if since_last < self._interval_ms:
                continue
            since_last = 0
            try:
                await self._transmit(ws, bytes(buffer))
            except Exception:
                logger.warning("stt.partial_failed", provider=self.name)

        # Whatever is left, even if it never crossed the interval, has to reach
        # the server before `flush` asks it to finalize. Without this an
        # utterance shorter than one interval, most backchannels, "haan",
        # "theek hai", never reaches the server at all: flush finalizes zero
        # seconds of audio. Found live, against the real socket, on exactly
        # that shape of utterance.
        if since_last > 0:
            try:
                await self._transmit(ws, bytes(buffer))
            except Exception:
                logger.warning("stt.partial_failed", provider=self.name)

        await ws.flush()

    async def _transmit(self, ws, pcm: bytes) -> None:
        wav = to_wav(pcm, sample_rate=SAMPLE_RATE)
        await ws.transcribe(
            audio=base64.b64encode(wav).decode("ascii"),
            encoding="audio/wav",
            sample_rate=SAMPLE_RATE,
        )


class SarvamBulbul:
    """Bulbul TTS, streamed. Same shape as `EdgeTts`, so it drops into
    `FailingOverTts`/`speak_as_they_arrive` without either knowing which stack
    produced the audio."""

    name = "sarvam-bulbul"
    mime = AUDIO_MIME
    bytes_per_second = BULBUL_BYTES_PER_SECOND

    def __init__(
        self,
        api_key: str | None = None,
        speaker: str = VOICE_HI_BULBUL,
        client=None,
    ) -> None:
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self._speaker = speaker
        self._client = client

    def _sdk_client(self):
        if self._client is not None:
            return self._client
        from sarvamai import AsyncSarvamAI

        return AsyncSarvamAI(api_subscription_key=self._api_key)

    async def synthesize(
        self, text: str, voice: str = "hi-IN", index: int = 0
    ) -> AsyncIterator[bytes]:
        if not self._api_key:
            raise TtsError("SARVAM_API_KEY is not set")

        with stage_span(
            TTS_SYNTHESIZE,
            **{
                "gen_ai.system": self.name,
                "vaani.tts.voice": self._speaker,
                "vaani.tts.chars": len(text),
                "vaani.tts.sentence_index": index,
            },
        ) as stage:
            chunks = 0
            client = self._sdk_client()
            try:
                async for chunk in client.text_to_speech.convert_stream(
                    text=text,
                    language_code=voice if voice != "hi-IN" else "hi-IN",
                    speaker=self._speaker,
                    model=BULBUL_MODEL,
                    output_audio_codec="mp3",
                    output_audio_bitrate=BULBUL_BITRATE,
                ):
                    if chunks == 0:
                        stage.mark("vaani.tts.first_chunk")
                    chunks += 1
                    yield chunk
            except Exception as exc:
                raise TtsError(f"{type(exc).__name__}") from exc

            stage.record(**{"vaani.tts.chunks": chunks})

        if chunks == 0:
            raise TtsError("no audio produced")
        logger.info("tts.done", provider=self.name, voice=self._speaker, chunks=chunks)
