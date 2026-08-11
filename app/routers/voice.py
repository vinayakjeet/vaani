"""The WebSocket the browser talks to.

Deliberately thin. It owns the socket and the framing and nothing else: the
endpoint decision is `vaani.endpoint`, the pipeline is `vaani.turn`, and every
concurrency bug this project will have belongs in the state machine M1.5 adds
rather than in here.

One session per socket. SPEC A10 makes concurrency a refusal rather than a
claim, and a second connection is turned away with a reason instead of being
queued into a worse experience nobody can debug.
"""

from __future__ import annotations

import asyncio

import spanlight
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vaani.endpoint import Endpointer
from vaani.protocol import (
    MAX_UTTERANCE_FRAMES,
    ClientMessage,
    ProtocolError,
    ServerMessage,
    decode,
    duration_ms,
)
from vaani.stt import GroqWhisper, SttError
from vaani.tts import EdgeTts, TtsError
from vaani.turn import Turn

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["voice"])

# One at a time. Not a scale decision so much as an honesty one: a free instance
# that accepts four sessions serves all four badly, and the user hears the
# degradation as the agent being slow rather than as the server being full.
_in_session = asyncio.Lock()


def build_turn() -> Turn:
    return Turn(stt=GroqWhisper(), tts=EdgeTts())


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
        try:
            await _serve(socket)
        except WebSocketDisconnect:
            # Expected. A browser tab closing is not an error, and logging it as
            # one buries the disconnects that are.
            logger.info("voice.disconnected")


async def _serve(socket: WebSocket) -> None:
    turn = build_turn()
    endpointer = Endpointer()
    buffer = bytearray()
    frames = 0
    generation = 0

    await socket.send_json({"type": ServerMessage.READY})

    while True:
        message = await socket.receive()

        # A closed tab arrives as a message, not an exception. Falling through to
        # the checks below would find neither text nor bytes, loop, and call
        # `receive` again on a dead socket, which raises out of the handler and
        # holds the single-session lock until the process restarts.
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))

        if "text" in message:
            action = message["text"]
            if action == ClientMessage.STOP:
                break
            if action == ClientMessage.START:
                generation += 1
                endpointer.reset()
                buffer.clear()
                frames = 0
            continue

        if "bytes" not in message:
            continue

        try:
            frame = decode(message["bytes"])
        except ProtocolError as exc:
            # The client is misframing. Saying so is more useful than dropping
            # frames until the transcript comes back as noise.
            await socket.send_json({"type": ServerMessage.ERROR, "reason": str(exc)})
            continue

        if frame.generation != generation:
            # Audio from a turn that has already been abandoned. Dropped on
            # arrival rather than trusted to have been stopped in time.
            continue

        buffer.extend(frame.pcm)
        frames += 1

        if frames >= MAX_UTTERANCE_FRAMES:
            await socket.send_json(
                {"type": ServerMessage.ERROR, "reason": "utterance too long"}
            )
            endpointer.reset()
            buffer.clear()
            frames = 0
            continue

        if not endpointer.accept(frame.pcm):
            continue

        spoken = bytes(buffer)
        buffer.clear()
        frames = 0
        endpointer.reset()
        await _answer(socket, turn, spoken, generation)


async def _answer(socket: WebSocket, turn: Turn, pcm: bytes, generation: int) -> None:
    """Run one turn and ship the audio back.

    The session span wraps the whole turn including the failure paths, so a turn
    that ended in an apology is still one session in the trace rather than a gap
    where a session should be.
    """
    with spanlight.session(name="vaani.turn") as session_id:
        try:
            result = await turn.run(pcm)
        except SttError as exc:
            await socket.send_json(
                {
                    "type": ServerMessage.ERROR,
                    "reason": "stt",
                    "detail": type(exc).__name__,
                }
            )
            return
        except TtsError as exc:
            await socket.send_json(
                {
                    "type": ServerMessage.ERROR,
                    "reason": "tts",
                    "detail": type(exc).__name__,
                }
            )
            return

        await socket.send_json(
            {
                "type": ServerMessage.TRANSCRIPT,
                "text": result.transcript.text,
                "session_id": session_id,
                "heard_ms": round(duration_ms(len(pcm) // 640)),
            }
        )
        await socket.send_json({"type": ServerMessage.REPLY, "text": result.reply})
        await socket.send_json(
            {
                "type": ServerMessage.AUDIO_START,
                "mime": result.audio_mime,
                "generation": generation,
            }
        )
        await socket.send_bytes(result.audio)
        await socket.send_json({"type": ServerMessage.AUDIO_END})
