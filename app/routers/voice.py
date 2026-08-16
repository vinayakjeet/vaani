"""The WebSocket the browser talks to.

Deliberately thin, and thinner than it was. It owns the socket, the framing, and
the single-session rule. The conversation itself is `vaani.session`, which is
written against a transport interface rather than against Starlette so it can be
tested without a socket or a microphone.

The overlapped pipeline is what serves a turn now. The unstreamed `vaani.turn`
stays in the tree untouched, because `bench/ablation.py` runs it as the baseline
and "streaming bought 400ms" means nothing without a measured before.

One session per socket. SPEC A10 makes concurrency a refusal rather than a claim,
and a second connection is turned away with a reason instead of being queued into
a worse experience nobody can debug.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import httpx
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from llm import ChatClient
from vaani.endpoint import Endpointer
from vaani.fillers import FillerBank, Purpose, read_chunks
from vaani.history import Conversation
from vaani.llm_turn import StreamedTurn
from vaani.pipeline import StreamingPipeline
from vaani.protocol import (
    ClientMessage,
    ProtocolError,
    ServerMessage,
    decode,
)
from vaani.session import Incoming, VoiceSession
from vaani.stt import TIMEOUT_SECONDS, ChunkedStt, GroqWhisper, RecoveringStt
from vaani.tts import VOICE_HI, EdgeTts, FailingOverTts
from vaani.turn_taking import TurnTaking

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["voice"])

# One at a time. Not a scale decision so much as an honesty one: a free instance
# that accepts four sessions serves all four badly, and the user hears the
# degradation as the agent being slow rather than as the server being full.
_in_session = asyncio.Lock()

# Short, and in Hinglish because the reply will be. It covers the gap when an
# answer misses the deadline, and it is never counted as the answer's own audio:
# `TurnClock` keeps that number separately so a filler cannot flatter the target.
# One phrase per purpose, since "Ek minute" and "Ji, boliye" are different speech
# acts and the on-demand fallback must not say the wrong one.
FILLER_TEXT: dict[Purpose, str] = {
    Purpose.THINKING: "Ek minute, dekh raha hoon.",
    Purpose.RESUMING: "Ji, boliye.",
}

# Synthesised once and committed, rather than spoken on demand.
#
# Measured in a browser against the live backend, an on-demand filler arrived 1796ms
# after speech ended. It exists to protect an 800ms floor and it could not, because
# saying it meant a network round trip to a synthesiser on the one path that is
# already late. The phrase never changes, so it is the one piece of audio in this
# system that can be a file.
#
# This is M1.8's argument in miniature: the tail belongs to the network, and the fix
# is not to make the request faster but to stop making it.
FILLER_AUDIO = Path(__file__).resolve().parent.parent.parent / "vaani" / "assets" / "filler-hi.mp3"

# Roughly a frame of MP3 at this bitrate. Chunked rather than sent whole so playback
# starts on the first slice, the same shape the synthesiser's own output has.
FILLER_CHUNK_BYTES = 720


class SocketTransport:
    """Starlette's WebSocket, in the shape `VoiceSession` expects.

    Everything Starlette-specific lives here. A closed tab arrives as a message
    rather than an exception, and translating it into one field means the session
    loop has no reason to know what a WebSocket is.
    """

    def __init__(self, socket: WebSocket) -> None:
        self._socket = socket

    async def receive(self) -> Incoming:
        message = await self._socket.receive()

        if message["type"] == "websocket.disconnect":
            return Incoming(disconnected=True)

        if "text" in message:
            return Incoming(control=message["text"])

        if "bytes" not in message:
            # Neither text nor audio. Ignored as an empty control rather than
            # looped on, because calling `receive` again on a socket that sent
            # something unreadable is how a handler spins.
            return Incoming(control="")

        try:
            return Incoming(frame=decode(message["bytes"]))
        except ProtocolError as exc:
            # The client is misframing. Saying so is more useful than dropping
            # frames until the transcript comes back as noise.
            await self.send_json({"type": ServerMessage.ERROR, "reason": str(exc)})
            return Incoming(control="")

    async def send_json(self, payload: dict) -> None:
        await self._socket.send_json(payload)

    async def send_bytes(self, data: bytes) -> None:
        await self._socket.send_bytes(data)


def build_answer(
    transcriber: GroqWhisper, tts: FailingOverTts, history: Conversation
) -> Callable[[AsyncIterator[bytes], Callable[[], bool]], AsyncIterator[bytes]]:
    """The overlapped pipeline, as the one callable a session needs.

    Built per session rather than per process so a turn's state cannot leak into
    the next visitor's, and so the endpointer belongs to one conversation.
    """
    pipeline = StreamingPipeline(
        # The batch path behind the chunked one. A dropped stream is retried through
        # the endpoint that takes a whole file, which is slower and works, and it is
        # the other mechanism rather than a retry of the one that just failed.
        stt=RecoveringStt(ChunkedStt(transcriber), transcriber),
        turn=StreamedTurn(llm=ChatClient()),
        tts=tts,
        history=history,
    )
    return pipeline.run


def build_backchannel_check(
    transcriber: GroqWhisper,
) -> Callable[[bytes], Awaitable[bool]]:
    """Whether a short interruption was an acknowledgement rather than a question.

    One transcription of the interrupting audio, on the path where the agent has already
    stopped talking, so the round trip costs a resume rather than a reply. The alternative
    is what Bolna can afford and this stack cannot: a persistent recogniser socket handing
    back interim transcripts to count words in, which on a chunked provider would be a
    request every few hundred milliseconds for the whole of every reply.
    """
    turns = TurnTaking()

    async def is_backchannel(pcm: bytes) -> bool:
        transcript = await transcriber.transcribe(pcm)
        # Counts and the verdict, never the words. This is user speech.
        verdict = turns.is_backchannel(transcript.text, agent_speaking=True)
        logger.info("voice.barge_checked", backchannel=verdict, chars=len(transcript.text))
        return verdict

    return is_backchannel


# One bank per process. The clips do not change between sessions and reading twelve
# files per turn would put a disk seek on the path that exists to avoid a network one.
# It holds the last clip played, which is per process rather than per session and is the
# right scope: a visitor who hears the same phrase the previous visitor heard has not
# noticed anything, and one who hears it twice running has.
_fillers = FillerBank()

# One connection to Groq's transcription endpoint, held for the process rather than
# opened fresh per request. `ChunkedStt` sends five to eleven of these for one utterance
# by design (SPEC A4), and each one used to pay its own TCP and TLS handshake to a
# provider that is not local. Safe to share across sessions because only one session
# runs at a time (`_in_session` above), so the pool is never contended.
_stt_client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)


async def speak_filler(purpose: Purpose = Purpose.THINKING) -> AsyncIterator[bytes]:
    """An acknowledgement from the bank, from disk, with no provider on the path.

    Never the same one twice running. The deadline is shorter than the endpoint wait, so
    this fires on most turns, and one clip repeating is how an IVR sounds.

    `purpose` picks the speech act: `THINKING` for an ordinary turn, `RESUMING` for one
    that began by resolving a verified interruption, where "ek minute" would be a stall
    on an answer the listener has already been told is coming.

    Falls back to synthesising if the bank is empty, so a fork that has not run
    `scripts/build_fillers.py` still speaks rather than going silent. That path is slow
    and says so in the log, because a filler that is late is worse than useless: it
    spends the deadline it was meant to cover.
    """
    clip = _fillers.pick(purpose)
    if clip is None:
        logger.warning("filler.bank_missing", purpose=str(purpose), assets=str(FILLER_AUDIO.parent))
        async for chunk in EdgeTts().synthesize(FILLER_TEXT[purpose], VOICE_HI):
            yield chunk
        return

    for chunk in read_chunks(clip, FILLER_CHUNK_BYTES):
        yield chunk


@router.websocket("/ws/voice")
async def voice(socket: WebSocket) -> None:
    await socket.accept()

    if _in_session.locked():
        await socket.send_json(
            {
                "type": ServerMessage.ERROR,
                "reason": "busy",
                "detail": "One session at a time on the free tier. Try again in a moment.",
            }
        )
        await socket.close()
        return

    async with _in_session:
        transcriber = GroqWhisper(client=_stt_client)
        # A second voice behind the first. The primary is unofficial with no uptime
        # promise, so this path runs whenever it breaks rather than only on the day
        # somebody remembers to test it.
        tts = FailingOverTts(EdgeTts(), EdgeTts())
        # One record, two writers of different things. The pipeline reads it to build the
        # prompt; the session decides what goes into it, because only the transport knows
        # which sentences the listener actually heard.
        history = Conversation()
        session = VoiceSession(
            transport=SocketTransport(socket),
            answer=build_answer(transcriber, tts, history),
            filler=speak_filler,
            endpointer=Endpointer(semantic=True),
            # Read off the synthesiser rather than assumed, so the playout estimate is a
            # byte count divided by the rate the audio was actually encoded at.
            bytes_per_second=tts.bytes_per_second,
            backchannel=build_backchannel_check(transcriber),
            history=history,
        )
        try:
            await session.run()
        except WebSocketDisconnect:
            # Expected. A browser tab closing is not an error, and logging it as
            # one buries the disconnects that are.
            logger.info("voice.disconnected")


__all__ = ["router", "ClientMessage"]
