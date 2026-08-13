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
from collections.abc import AsyncIterator, Callable

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from llm import ChatClient
from vaani.endpoint import Endpointer
from vaani.llm_turn import StreamedTurn
from vaani.pipeline import StreamingPipeline
from vaani.protocol import (
    ClientMessage,
    ProtocolError,
    ServerMessage,
    decode,
)
from vaani.session import Incoming, VoiceSession
from vaani.stt import ChunkedStt, GroqWhisper, RecoveringStt
from vaani.tts import VOICE_HI, EdgeTts, FailingOverTts

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["voice"])

# One at a time. Not a scale decision so much as an honesty one: a free instance
# that accepts four sessions serves all four badly, and the user hears the
# degradation as the agent being slow rather than as the server being full.
_in_session = asyncio.Lock()

# Short, and in Hinglish because the reply will be. It covers the gap when an
# answer misses the deadline, and it is never counted as the answer's own audio:
# `TurnClock` keeps that number separately so a filler cannot flatter the target.
FILLER_TEXT = "Ek minute, dekh raha hoon."


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


def build_answer() -> Callable[[AsyncIterator[bytes], Callable[[], bool]], AsyncIterator[bytes]]:
    """The overlapped pipeline, as the one callable a session needs.

    Built per session rather than per process so a turn's state cannot leak into
    the next visitor's, and so the endpointer belongs to one conversation.
    """
    transcriber = GroqWhisper()
    pipeline = StreamingPipeline(
        # The batch path behind the chunked one. A dropped stream is retried through
        # the endpoint that takes a whole file, which is slower and works, and it is
        # the other mechanism rather than a retry of the one that just failed.
        stt=RecoveringStt(ChunkedStt(transcriber), transcriber),
        turn=StreamedTurn(llm=ChatClient()),
        # A second voice behind the first. The primary is unofficial with no uptime
        # promise, so this path runs whenever it breaks rather than only on the day
        # somebody remembers to test it.
        tts=FailingOverTts(EdgeTts(), EdgeTts()),
    )
    return pipeline.run


async def speak_filler() -> AsyncIterator[bytes]:
    async for chunk in EdgeTts().synthesize(FILLER_TEXT, VOICE_HI):
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
        session = VoiceSession(
            transport=SocketTransport(socket),
            answer=build_answer(),
            filler=speak_filler,
            endpointer=Endpointer(semantic=True),
        )
        try:
            await session.run()
        except WebSocketDisconnect:
            # Expected. A browser tab closing is not an error, and logging it as
            # one buries the disconnects that are.
            logger.info("voice.disconnected")


__all__ = ["router", "ClientMessage"]
