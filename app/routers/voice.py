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
import time

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
from vaani.spans import (
    PLAYBACK_FIRST_AUDIO,
    TURN,
    VAD_ENDPOINT,
    stage_span,
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
    # When this turn's first speech frame arrived. `vad.endpoint` starts there
    # rather than at socket open, because a user who takes three seconds to begin
    # would otherwise have that silence recorded as detection work.
    speech_began_at: int | None = None

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
                speech_began_at = None
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
        if speech_began_at is None and endpointer.started:
            speech_began_at = time.time_ns()

        if frames >= MAX_UTTERANCE_FRAMES:
            await socket.send_json(
                {"type": ServerMessage.ERROR, "reason": "utterance too long"}
            )
            endpointer.reset()
            buffer.clear()
            frames = 0
            speech_began_at = None
            continue

        if not endpointer.accept(frame.pcm):
            continue

        # Closed the instant the endpointer fires, before any request is built, and
        # backdated to the first frame of speech. The trailing silence is inside it
        # on purpose: it is the detector's own cost, so the aggressiveness knob
        # visibly moves this number, which is the point of measuring it.
        with stage_span(
            VAD_ENDPOINT,
            start_time=speech_began_at,
            **{
                "vaani.vad.speech_ms": endpointer.speech_ms,
                "vaani.vad.trailing_silence_ms": endpointer.trailing_silence_ms,
                "vaani.vad.aggressiveness": endpointer.aggressiveness
                if endpointer.aggressiveness is not None
                else -1,
            },
        ):
            pass

        spoken = bytes(buffer)
        buffer.clear()
        frames = 0
        endpointer.reset()
        speech_began_at = None
        await _answer(socket, turn, spoken, generation)


async def _answer(socket: WebSocket, turn: Turn, pcm: bytes, generation: int) -> None:
    """Run one turn and ship the audio back.

    The session span wraps the whole turn including the failure paths, so a turn
    that ended in an apology is still one session in the trace rather than a gap
    where a session should be.

    One session per turn rather than per socket, which is a deviation from SPEC's
    tree and a deliberate one. Detector state in Spanlight is per session, so a
    session spanning a whole conversation would read the repeated tool calls of
    three separate questions as a loop and fire on a healthy dialogue. ShipGate hit
    the same thing and went to per-item sessions for the same reason. The `turn`
    span still exists inside it so the attributes SPEC names have somewhere to live.
    """
    with spanlight.session(name="vaani.turn") as session_id, stage_span(
        TURN,
        **{"vaani.turn.index": generation, "vaani.turn.interrupted": False},
    ):
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

        # Starts when the first audio goes onto the wire. It should end when the
        # browser reports playback has begun, and the browser does not report yet,
        # so it closes immediately and says so rather than pretending. A span that
        # silently never ends is worse than a short one that admits what it is
        # missing, and `reported=false` is queryable, which an absent span is not.
        queued_at = time.monotonic()
        await socket.send_bytes(result.audio)
        with stage_span(
            PLAYBACK_FIRST_AUDIO,
            **{
                "vaani.playback.queued_ms": (time.monotonic() - queued_at) * 1000,
                "vaani.playback.reported": False,
            },
        ):
            pass

        await socket.send_json({"type": ServerMessage.AUDIO_END})
