"""One socket's worth of conversation, with playback that does not block receiving.

The defect this exists to fix: a handler that awaits its own sends cannot read an
interrupt. The previous shape sent a whole answer as one payload after synthesising
all of it, so a user talking over the agent waited for the entire reply before
anything noticed, roughly two seconds. No amount of cancellation machinery
downstream helps, because the message asking for it had not been read yet.

So receiving and speaking are separate tasks. The receive loop only ever reads, and
audio leaves on a playback task it can cancel. That bounds the worst case at one
chunk rather than one answer, which is the difference between barge-in working and
barge-in existing in a module.

Transport is an interface rather than a WebSocket, because a test that needs a real
socket and a real microphone is a test that never runs in CI. The router adapts
Starlette to it and owns nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import structlog

from llm.types import ProviderError
from vaani.barge_in import AudioChunk, SpeakingTurn
from vaani.budget import TurnClock, speak_within
from vaani.endpoint import Endpointer, MicState
from vaani.protocol import ClientMessage, Frame, ServerMessage
from vaani.state import State, TurnState
from vaani.stt import SttError
from vaani.tts import AUDIO_MIME, TtsError

logger = structlog.get_logger(__name__)

# Exceptions whose messages this project writes itself, so they name a reason rather
# than echoing a provider. Anything else is logged by class only.
SAFE_TO_LOG = (ProviderError, SttError, TtsError)


@dataclass(frozen=True)
class Incoming:
    """One thing the client sent, already framed."""

    control: str | None = None
    frame: Frame | None = None
    disconnected: bool = False


class Transport(Protocol):
    async def receive(self) -> Incoming: ...
    async def send_json(self, payload: dict) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...


class Answer(Protocol):
    """One turn's audio, with hooks for the text the client displays.

    The hooks exist because the browser shows what it heard and what was said, and
    only the pipeline knows those. Passing them in rather than returning them keeps
    the transcript pane updating as the words appear instead of after the audio for
    them has been synthesised.
    """

    def __call__(
        self,
        frames: AsyncIterator[bytes],
        still_current: Callable[[], bool],
        on_transcript: Callable[[str], Awaitable[None]],
        on_sentence: Callable[[str], Awaitable[None]],
    ) -> AsyncIterator[bytes]: ...


class VoiceSession:
    """The receive loop, and the playback task it can interrupt."""

    def __init__(
        self,
        transport: Transport,
        answer: Answer,
        filler: Callable[[], AsyncIterator[bytes]],
        endpointer: Endpointer | None = None,
    ) -> None:
        self._transport = transport
        self._answer = answer
        self._filler = filler
        self._endpointer = endpointer or Endpointer(semantic=True)
        self._state = TurnState()

        self._speaking: SpeakingTurn | None = None
        self._playback: asyncio.Task[None] | None = None
        self._frames: asyncio.Queue[bytes | None] | None = None
        # Both tasks write to one transport. Only playback sends audio and only the
        # receive loop sends control JSON, but they can be in flight at the same
        # instant, and a WebSocket is not safe for concurrent writes.
        self._send = asyncio.Lock()
        # Which generation has had its AUDIO_START sent. The client uses that message
        # to open a playback buffer, so sending it twice for one turn splices the
        # answer and sending it never leaves the audio unplayed.
        self._announced: int | None = None
        # Which microphone complaint has already been made this turn, so it is said
        # once rather than fifty times a second.
        self._microphone_reported: MicState | None = None
        self.clock: TurnClock | None = None

    async def run(self) -> None:
        await self._say({"type": ServerMessage.READY})
        try:
            while True:
                message = await self._transport.receive()
                if message.disconnected:
                    return
                if message.control is not None:
                    if await self._on_control(message.control):
                        return
                elif message.frame is not None:
                    await self._on_frame(message.frame)
        finally:
            # A half-finished turn is abandoned rather than left running. SPEC's
            # disconnect rule, and also what keeps a dropped tab from holding a
            # provider stream open on a free instance.
            await self._stop_speaking()

    async def _on_control(self, control: str) -> bool:
        """Returns whether the session should end."""
        if control == ClientMessage.STOP:
            return True
        if control == ClientMessage.START:
            await self._stop_speaking()
            self._begin_listening()
        elif control == ClientMessage.INTERRUPT:
            # The whole point of the split. This is read while audio is still going
            # out, so it is acted on now rather than after the answer finishes.
            await self._interrupt()
        return False

    async def _on_frame(self, frame: Frame) -> None:
        if not self._state.owns(frame.generation):
            # Audio from a turn already abandoned. Dropped on arrival rather than
            # trusted to have been stopped in time.
            return

        if self._state.state is State.SPEAKING:
            # Speech while the agent is talking is a barge-in without the client
            # having to say so. A client that sends INTERRUPT gets there sooner; one
            # that does not still gets interrupted.
            #
            # `started` and not `accept`, and only after the endpointer was reset for
            # this purpose. `accept` answers "has the turn ended", which is the wrong
            # question here, and reading a stale `started` left over from the
            # utterance that just finished made the first silence frame of playback
            # interrupt the answer it had only just started.
            #
            # `started` also requires `min_speech_ms` of sustained speech, so a cough
            # or a door does not cut the agent off mid-sentence, which is the same
            # rule that keeps a cough from starting a turn.
            self._endpointer.accept(frame.pcm)
            if self._endpointer.started:
                await self._interrupt()
            return

        if self._state.state is not State.LISTENING:
            return

        assert self._frames is not None
        await self._frames.put(frame.pcm)

        if self._endpointer.accept(frame.pcm):
            await self._frames.put(None)
            await self._begin_answering()
            return

        await self._check_microphone()

    async def _check_microphone(self) -> None:
        """Say so when no speech is arriving, rather than waiting for an endpoint.

        Reported once per turn. Repeating it every 20ms would bury the message under
        itself, and the session keeps listening afterwards because the usual fix is
        the user unmuting and speaking again.
        """
        state = self._endpointer.diagnose()
        if state is MicState.OK or state is self._microphone_reported:
            return

        self._microphone_reported = state
        logger.info("session.microphone", state=str(state))
        await self._say(
            {
                "type": ServerMessage.ERROR,
                "reason": "microphone",
                "detail": str(state),
            }
        )

    def _begin_listening(self) -> None:
        self._state.begin()
        self._endpointer.reset()
        self._microphone_reported = None
        self._frames = asyncio.Queue()

    async def _begin_answering(self) -> None:
        self._state.to(State.THINKING)
        generation = self._state.generation
        # Read before the reset, and that order is the whole measurement.
        #
        # The clock is backdated to the last frame of speech, which is `silence_ms`
        # ago: the endpoint fires only once the trailing silence has been waited out.
        # Starting it here instead would remove that wait from every number, several
        # hundred milliseconds of it, in the flattering direction. SPEC picks the
        # harder clock on purpose, because most published voice latency starts after
        # endpointing and two systems quoting the same figure can differ by a factor of
        # two in the only thing a listener perceives.
        #
        # The first version of this read `silence_ms` after the reset below had already
        # zeroed it, so the backdating was always nothing and the flattering clock
        # shipped behind a comment saying it had not. Where you start the measurement
        # is the measurement, and so is when you read it.
        spent_waiting = self._endpointer.silence_ms * 1_000_000
        self.clock = TurnClock(started_ns=time.monotonic_ns() - spent_waiting)

        # Reset after, because the endpointer changes job here: it stops answering
        # "has the turn ended" and starts answering "is the user talking over us".
        # Carrying the finished utterance's state into that question makes the answer
        # yes immediately.
        self._endpointer.reset()

        self._speaking = SpeakingTurn(
            generation=generation, produce=lambda: self._audio(generation)
        )
        self._speaking.start()
        self._state.to(State.SPEAKING)
        self._playback = asyncio.create_task(self._play(self._speaking))

    def _audio(self, generation: int) -> AsyncIterator[bytes]:
        assert self._frames is not None
        assert self.clock is not None
        answer = self._answer(
            _drain(self._frames),
            lambda: self._state.owns(generation),
            lambda text: self._report(ServerMessage.TRANSCRIPT, text, generation),
            lambda text: self._report(ServerMessage.REPLY, text, generation),
        )
        return speak_within(answer, self._filler, self.clock)

    async def _report(self, kind: str, text: str, generation: int) -> None:
        """Text for the client's transcript pane, dropped if the turn is stale.

        Without the generation check an interrupted turn keeps writing into the pane
        after the user has moved on, so the screen shows an answer to the previous
        question while the agent listens to the next one.
        """
        if not self._state.owns(generation):
            return
        await self._say({"type": kind, "text": text})

    async def _play(self, speaking: SpeakingTurn) -> None:
        try:
            async for chunk in speaking.chunks():
                await self._send_audio(chunk)
        except Exception as exc:
            # Reported rather than raised. This is a task, so raising here surfaces
            # as an unretrieved exception with nothing to attribute it to, and the
            # user would hear silence with no reason given.
            # The class alone was not diagnosable: a live turn failed with
            # "ProviderError" and nothing in the logs could say which of four raises
            # it was. Our own exception messages carry a reason and no provider body,
            # so they are safe to log; `ProviderClientError` is excluded because it
            # quotes the response, which can quote a transcript back.
            detail = str(exc) if isinstance(exc, SAFE_TO_LOG) else ""
            logger.warning(
                "session.playback_failed", error=type(exc).__name__, detail=detail
            )
            await self._say(
                {
                    "type": ServerMessage.ERROR,
                    "reason": "playback",
                    "detail": type(exc).__name__,
                }
            )
        finally:
            if self._state.state is State.SPEAKING:
                self._state.to(State.LISTENING)
                self._endpointer.reset()
            await self._say({"type": ServerMessage.AUDIO_END})

    async def _interrupt(self) -> None:
        # The new turn is begun before the old one is stopped, and the order is the
        # whole correctness of this method.
        #
        # Playback's own teardown returns the machine to LISTENING when it ends, and
        # stopping first lets that run before the interruption is recorded, so the
        # turn came back marked as finished rather than cut off and
        # `vaani.turn.interrupted` was false on every barge-in.
        #
        # Bumping the generation first also makes every chunk already dequeued stale
        # immediately, so the transport drops it without depending on how quickly the
        # producer stops.
        self._begin_listening()
        await self._stop_speaking()
        await self._say({"type": ServerMessage.READY})

    async def _stop_speaking(self) -> None:
        if self._speaking is not None:
            await self._speaking.cancel()
            self._speaking = None
        if self._playback is not None:
            self._playback.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._playback
            self._playback = None

    async def _send_audio(self, chunk: AudioChunk) -> None:
        if not self._state.owns(chunk.generation):
            # Produced before the interruption and dequeued after it. Stopping the
            # producer does not unsend what it already handed on, so the last chance
            # to drop it is here.
            return

        if self._announced != chunk.generation:
            self._announced = chunk.generation
            await self._say(
                {
                    "type": ServerMessage.AUDIO_START,
                    "mime": AUDIO_MIME,
                    "generation": chunk.generation,
                }
            )

        async with self._send:
            await self._transport.send_bytes(chunk.data)

    async def _say(self, payload: dict) -> None:
        async with self._send:
            await self._transport.send_json(payload)


async def _drain(queue: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
    """Frames the receive loop has accepted, ending when it says the turn ended."""
    while True:
        frame = await queue.get()
        if frame is None:
            return
        yield frame
