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
from vaani.budget import BargeClock, TurnClock, speak_within
from vaani.endpoint import Endpointer, MicState
from vaani.fillers import Purpose
from vaani.history import Conversation
from vaani.protocol import FRAME_BYTES, ClientMessage, Frame, ServerMessage
from vaani.quality import Interactivity, Interruption
from vaani.spans import TURN, VAD_ENDPOINT, stage_span
from vaani.state import State, TurnState
from vaani.stt import SttError
from vaani.tts import AUDIO_MIME, TtsError, playout_seconds
from vaani.turn_taking import AudioStatus, Barge, PendingBarge, TurnTaking

logger = structlog.get_logger(__name__)

# Exceptions whose messages this project writes itself, so they name a reason rather
# than echoing a provider. Anything else is logged by class only.
SAFE_TO_LOG = (ProviderError, SttError, TtsError)

# How long a hold may go on with no further evidence that the user is still talking before
# the flag is treated as stuck and cleared.
#
# The first version bounded the hold and then sent the audio anyway, which unblocks the turn
# and leaves the wrong state in place, so the next chunk waits the full three seconds again.
# Bolna's version is better and it is the one here: the release is keyed on staleness rather
# than on a timer, and it repairs the flag instead of bypassing the gate. Their constant is
# also 3.0 seconds, arrived at independently.
STUCK_HOLD_S = 3.0

# How long a connected client may go without sending anything at all before the session
# gives up on it and releases the one-at-a-time lock in `app/routers/voice.py`.
#
# Found live: a connection that never sends a clean close, whether a dropped wifi link, a
# frozen tab, or a laptop put to sleep, leaves `receive()` awaiting a socket that will not
# produce another message for a long time, bounded by the OS's own TCP timeout rather than
# anything this process controls. `_in_session` is then held until the process is restarted
# by hand, and every visitor after the first one is told the service is busy, forever.
#
# Twenty seconds is safe rather than tight, because a listening client is never legitimately
# silent this long: the AudioWorklet forwards a frame every 20ms for as long as the tab is
# capturing, mid-answer included, since barge-in depends on that same stream. A frame every
# 20ms going quiet for twenty full seconds is not a pause, it is an absence.
IDLE_RECEIVE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class Incoming:
    """One thing the client sent, already framed."""

    control: str | None = None
    frame: Frame | None = None
    disconnected: bool = False


@dataclass(frozen=True)
class PendingTurnSpan:
    """What `turn`'s span needs, captured where it is known and read where it ends.

    `_begin_answering` knows the start; `_send_audio` knows when playback actually
    begins, which is the span's end per `bench/stages.md`. Carrying it as one small
    object between the two is less to get wrong than three separate attributes on
    `self` that have to be kept in step by hand.
    """

    generation: int
    began_ns: int | None
    index: int
    interrupted_previous: bool


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
        filler: Callable[[Purpose], AsyncIterator[bytes]],
        endpointer: Endpointer | None = None,
        bytes_per_second: int = 0,
        backchannel: Callable[[bytes], Awaitable[bool]] | None = None,
        history: Conversation | None = None,
    ) -> None:
        self._transport = transport
        self._answer = answer
        self._filler = filler
        self._endpointer = endpointer or Endpointer(semantic=True)
        self._state = TurnState()
        # How a chunk's byte count becomes a duration. Zero means the caller could not
        # say, and then the playout estimate stays at zero rather than being invented:
        # a confidently wrong duration is worse than an absent one, because the gate's
        # grace period trusts it.
        self._bytes_per_second = bytes_per_second
        # Whether a short interruption was a backchannel. None means no verifier, so a
        # suspected interruption is committed on duration alone, which is the previous
        # behaviour and the ablation's baseline arm for this technique.
        self._backchannel = backchannel

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
        # The turn-taking decisions read from Bolna: the three-state gate, the playout
        # estimate, and the backchannel rule.
        self._turns = TurnTaking()
        # The interruption currently under suspicion, if any. Held rather than acted on
        # until it is either long enough to be certain or identified by the recogniser.
        self._barge: PendingBarge | None = None
        # That suspicion's clock, started at the frame the speech began on rather than
        # at the frame we noticed it. Separate from `_barge` because it outlives the
        # decision: the record is written when the suspicion is resolved, and resolving
        # it is what clears the suspicion.
        self._barge_clock: BargeClock | None = None
        # Every resolved interruption this session, for M2.4. Kept rather than only
        # logged, because a bench that had to parse its own log lines back would be
        # measuring the logging.
        self.barges: list[BargeClock] = []
        # Counted from one, per session, for `vaani.turn.index`. A waterfall reading
        # spans out of order still knows which turn of the conversation each one was.
        self._turn_index = 0
        # The `turn` span's start, captured in `_begin_answering` and consumed in
        # `_send_audio` once playback actually begins, which is where the span ends.
        self._pending_turn_span: PendingTurnSpan | None = None
        # What the latency numbers cannot see. Per session rather than per turn, because
        # every ratio in it is about a conversation rather than an utterance.
        self.quality = Interactivity()
        # Shared with the pipeline, which reads it to build the prompt. Every write is
        # here, because a write needs the generation and the transport is the only thing
        # that knows whether a sentence's audio actually went out.
        self.history = history if history is not None else Conversation()
        self.clock: TurnClock | None = None

    async def run(self) -> None:
        await self._say({"type": ServerMessage.READY})
        try:
            while True:
                message = await self._receive_or_give_up()
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

    async def _receive_or_give_up(self) -> Incoming:
        """Wait for the transport, but not forever.

        A clean disconnect arrives as a message; a dead one, whether a dropped link,
        a frozen tab, or a sleeping laptop, arrives as nothing at all, and awaiting a
        socket that will not produce another message is indistinguishable from a
        client that is merely quiet. So the wait itself is bounded, and going quiet
        is treated the same as saying goodbye: `run`'s own `finally` already abandons
        a half-finished turn and releases the one-at-a-time lock either way.
        """
        try:
            return await asyncio.wait_for(
                self._transport.receive(), timeout=IDLE_RECEIVE_TIMEOUT_S
            )
        except TimeoutError:
            logger.info("session.idle_timeout", seconds=IDLE_RECEIVE_TIMEOUT_S)
            return Incoming(disconnected=True)

    async def _on_control(self, control: str) -> bool:
        """Returns whether the session should end."""
        if control == ClientMessage.STOP:
            return True
        if control == ClientMessage.START:
            if self._state.state in (State.THINKING, State.SPEAKING):
                # A client asking to start over while the agent is still answering is
                # asking for the same thing an explicit interrupt is, and the reply in
                # flight needs the same accounting `_interrupt` already gets right:
                # truncated to what was actually heard, not silently orphaned in
                # `Conversation._staged` and not committed as complete for having been
                # cut off by a new START rather than an INTERRUPT.
                #
                # Found writing a test for the turn span, not from a report: calling
                # `_stop_speaking` before `_begin_listening` here, the reverse of
                # `_interrupt`'s order, let `_play`'s own teardown see the state still
                # SPEAKING and commit the turn as though it had finished, and computed
                # `interrupted_previous` against the state `_begin_listening` had
                # already changed by the time it ran. Not currently reachable by the
                # real client, which sends START only once at socket-open, but the
                # state machine's own docstring calls this a legal barge-in and it
                # was not behaving like the other one.
                await self._interrupt()
            else:
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
            await self._on_frame_while_speaking(frame)
            return

        if self._state.state is not State.LISTENING:
            return

        assert self._frames is not None
        await self._frames.put(frame.pcm)

        if self._endpointer.accept(frame.pcm):
            self._span_vad_endpoint()
            await self._frames.put(None)
            await self._begin_answering()
            return

        await self._check_microphone()

    async def _on_frame_while_speaking(self, frame: Frame) -> None:
        """Speech over the agent: pause first, decide second.

        Interrupting on any sustained speech is what made the agent feel neurotic, because
        a Hinglish speaker says "haan" and "achha" over a reply constantly and every one of
        them killed playback. Enumerating those was the wrong fix: they are an open set.

        So there are two thresholds on one signal. Long speech commits immediately, since
        nobody backchannels for `commit_ms`, and making a real interruption wait for a
        transcription round trip is the barge-in latency this project is bounding. Short
        speech only pauses: the audio is held rather than discarded, the utterance is
        identified, and then the turn is either cancelled or resumed. That is X-Talk's
        pause-verify-resume loop, and it is possible here only because the gate already had
        a WAIT state to hold audio in.

        `started` rather than `accept`'s return value, and only after the endpointer was
        reset for this purpose. `accept` answers "has the turn ended", which is the wrong
        question, and a stale `started` from the utterance that just finished made the first
        silence frame of playback interrupt the answer it had only just started.
        """
        ended = self._endpointer.accept(frame.pcm)

        if not self._endpointer.started:
            return

        # Held audio needs to know the user has started, which is what turns the gate to
        # WAIT rather than letting the agent talk over them.
        self._turns.note_user_speaking()
        self.quality.user_started_speaking()

        if self._barge is None:
            self._barge = PendingBarge()
            # Backdated to where the speech started, which is `min_speech_ms` before
            # this frame. Detection cannot fire any sooner than that by construction, so
            # a clock started here would report the whole mechanism as free.
            self._barge_clock = BargeClock.from_speech(self._endpointer.speech_run_ms)

        match self._barge.note(frame.pcm, self._endpointer.speech_ms):
            case Barge.COMMIT:
                # Still mid-utterance, so the new turn keeps listening from here with the
                # opening syllables restored.
                await self._commit_barge("duration", complete=False)
                return
            case Barge.PAUSE if not self._barge.paused:
                self._barge.paused = True
                logger.info("session.barge_pending")
                await self._say({"type": ServerMessage.PAUSE})
                # After the send rather than before it. The client cannot act on a
                # message still queued behind an audio chunk, and the lock this send
                # takes is the queue.
                if self._barge_clock is not None:
                    self._barge_clock.mark_paused()
            case _:
                pass

        if ended and self._barge is not None:
            # The interrupting utterance finished under the commit threshold, so it is
            # short enough to be a backchannel and has to be identified before a turn is
            # thrown away for it.
            await self._resolve_barge()

    async def _resolve_barge(self) -> None:
        assert self._barge is not None
        audio = bytes(self._barge.audio)

        if self._backchannel is None:
            await self._commit_barge("unverified", complete=True)
            return

        try:
            is_backchannel = await self._backchannel(audio)
        except SttError as exc:
            # A recogniser that could not say is not evidence of a backchannel. Committing
            # is the safe direction: the cost is a turn the user has to repeat, where
            # resuming over somebody who really did interrupt is the failure this exists to
            # fix.
            logger.warning("session.barge_unverified", detail=str(exc))
            await self._commit_barge("verify_failed", complete=True)
            return

        if not is_backchannel:
            await self._commit_barge("transcript", complete=True)
            return

        logger.info("session.backchannel_ignored", speech_ms=self._barge.speech_ms)
        # Counted, because the other reading of a low interruption count is that the
        # detector never fires, and the two are identical from the latency numbers.
        self.quality.backchannel_ignored()
        self._barge = None
        # The user stopped, so the gate leaves WAIT and the held audio goes out. The turn
        # is not counted, because a backchannel is not a turn and counting it would start
        # the grace period that delays the rest of this very reply.
        self._turns.note_user_stopped(count_turn=False)
        self.quality.user_stopped_speaking()
        self._endpointer.reset()
        await self._say({"type": ServerMessage.RESUME})
        self._retire_barge_clock(lambda clock: clock.mark_resumed())

    async def _commit_barge(self, reason: str, *, complete: bool) -> None:
        """Turn a suspicion into an interruption, and start the turn it belongs to.

        `complete` is the difference between the two paths and it decides whether the new
        turn is still listening. On the duration path the user is mid-sentence, so the turn
        continues from where they are. On the verified path their utterance already ended,
        and the endpoint that will never fire again is read off the endpointer before it is
        reset: without that the clock would not be backdated and the turn would wait for a
        second utterance that is not coming.
        """
        speech_ms = self._barge.speech_ms if self._barge is not None else 0
        heard = bytes(self._barge.audio) if self._barge is not None else b""
        silence_ms = self._endpointer.silence_ms
        self._barge = None
        logger.info("session.interrupted", reason=reason, speech_ms=speech_ms)
        self.quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)

        await self._interrupt(heard=heard, outcome=reason)

        if complete and self._frames is not None:
            self._frames.put_nowait(None)
            # Only this path ever reaches here with `complete=True`: the duration
            # path passes `complete=False`, since the user is still mid-sentence and
            # the turn keeps listening rather than re-answering immediately. So a
            # turn beginning here has always resolved a verified interruption, never
            # talked over someone still speaking, which is the one hazard that made
            # this deliberately unbuilt for a session before this one.
            await self._begin_answering(silence_ms=silence_ms, acknowledge=True)

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

        # A working mic that has heard nothing is a different situation from a broken
        # one, and currently reads identically from the far end: silence either way.
        # SILENT keeps only the diagnostic above, since asking "are you there" of
        # someone who cannot be heard at all is the wrong message; TOO_QUIET means the
        # input looks fine and the honest reading is that the person may simply have
        # stepped away, which the diagnostic alone does not say.
        #
        # `speech_ms == 0` on top of the state check, found writing the test for this
        # rather than assumed: `diagnose` reads `peak_rms`, a running max updated on
        # every frame, while `started` needs `min_speech_ms` of sustained speech.
        # Real speech beginning right after a long silence is loud on its very first
        # frame and started for none of the next several, so for up to that many
        # frames the state genuinely reads TOO_QUIET while somebody is mid-word. This
        # excludes exactly that window rather than greeting the start of an answer
        # with "are you still there".
        if state is MicState.TOO_QUIET and self._endpointer.speech_ms == 0:
            await self._say({"type": ServerMessage.CHECKING_IN})

    def _span_vad_endpoint(self) -> None:
        """The `vad.endpoint` span, emitted the instant the endpointer fires.

        Both boundaries are already known at this one point: the start is the first
        frame that cleared the threshold, backdated on the endpointer itself, and the
        end is now. A context manager entered and left in the same statement is the
        honest shape for a span whose whole extent is already in the past.
        """
        began = self._endpointer.speech_began_ns
        if began is None:
            # Cannot happen on the path that calls this: `accept` only returns true
            # once `_started`, which requires speech to have begun. Guarded anyway,
            # because a span with a fabricated start is worse than a missing one.
            logger.error("session.vad_span_missing_start")
            return
        attributes: dict[str, object] = {
            "vaani.vad.speech_ms": self._endpointer.speech_ms,
            "vaani.vad.trailing_silence_ms": self._endpointer.trailing_silence_ms,
        }
        if self._endpointer.aggressiveness is not None:
            attributes["vaani.vad.aggressiveness"] = self._endpointer.aggressiveness
        with stage_span(VAD_ENDPOINT, start_time=began, **attributes):
            pass

    def _span_turn(self, generation: int) -> None:
        """The `turn` span: first frame of speech to playback starting, filler or not.

        Matches `first_audio_ms` in `bench/stages.md`'s three-number table rather than
        `first_answer_audio_ms`: `turn` wraps every stage including
        `playback.first_audio`, whose own start is the same "first chunk of any kind"
        moment this already is.
        """
        pending = self._pending_turn_span
        self._pending_turn_span = None
        if pending is None or pending.generation != generation:
            # Stale or absent. A generation mismatch means the turn this chunk belongs
            # to was never the one `_begin_answering` opened this pending span for,
            # which should not happen given generations only move forward, but a
            # missing span beats a mislabelled one.
            return
        if pending.began_ns is None:
            logger.info("session.turn_span_skipped", generation=generation)
            return
        with stage_span(
            TURN,
            start_time=pending.began_ns,
            **{
                "vaani.turn.index": pending.index,
                "vaani.turn.interrupted": pending.interrupted_previous,
            },
        ):
            pass

    def _begin_listening(self) -> None:
        self._state.begin()
        self._endpointer.reset()
        self._microphone_reported = None
        self._frames = asyncio.Queue()

    async def _begin_answering(
        self, silence_ms: int | None = None, *, acknowledge: bool = False
    ) -> None:
        self._state.to(State.THINKING)
        generation = self._state.generation
        # Found alongside the turn span: `_announced` was compared against
        # `chunk.generation` alone, on the assumption every new turn has a new
        # generation. That is only true across an interrupt; an ordinary second
        # question, asked after the first answer finished without ever being cut
        # off, keeps the same generation the whole session, since nothing but an
        # interrupt or a fresh START advances it. So a second, uninterrupted turn's
        # first chunk compared equal to the value the first turn already left here,
        # AUDIO_START was never sent again, and the client had no signal to open a
        # playback buffer for a reply the server was already sending. Reset once per
        # turn, which `_begin_answering` runs exactly once for regardless of whether
        # the generation actually moved, is what makes the check correct again.
        self._announced = None
        self.quality.turn_started()
        self._turn_index += 1
        # `speech_began_ns` before the reset below clears it, same rule as the clock
        # two lines down. None on the path that resolves a verified interruption: the
        # interrupting audio is written straight into the frame queue in `_interrupt`
        # without passing through `accept`, so the endpointer never saw it and has
        # nothing to backdate from. The turn span is skipped rather than started from
        # a fabricated time; `_send_audio` is where that is decided.
        self._pending_turn_span = PendingTurnSpan(
            generation=generation,
            began_ns=self._endpointer.speech_began_ns,
            index=self._turn_index,
            interrupted_previous=self._state.interrupted_previous,
        )
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
        # `silence_ms` is passed when the endpointer has already been reset for a new turn,
        # which happens on a verified interruption: the utterance that ended is the one being
        # answered, and its trailing silence is no longer readable here.
        waited = self._endpointer.silence_ms if silence_ms is None else silence_ms
        self._turns.note_user_stopped()
        self.quality.user_stopped_speaking()
        spent_waiting = waited * 1_000_000
        self.clock = TurnClock(started_ns=time.monotonic_ns() - spent_waiting)

        # Reset after, because the endpointer changes job here: it stops answering
        # "has the turn ended" and starts answering "is the user talking over us".
        # Carrying the finished utterance's state into that question makes the answer
        # yes immediately.
        self._endpointer.reset()

        self._speaking = SpeakingTurn(
            generation=generation,
            produce=lambda: self._audio(generation, acknowledge=acknowledge),
        )
        self._speaking.start()
        self._state.to(State.SPEAKING)
        self._playback = asyncio.create_task(self._play(self._speaking))

    def _audio(self, generation: int, *, acknowledge: bool = False) -> AsyncIterator[bytes]:
        assert self._frames is not None
        assert self.clock is not None
        answer = self._answer(
            _drain(self._frames),
            lambda: self._state.owns(generation),
            lambda text: self._report(ServerMessage.TRANSCRIPT, text, generation),
            lambda text: self._report(ServerMessage.REPLY, text, generation),
        )
        # "Ek minute" says the answer is coming; "Ji, boliye" says the floor is
        # theirs. Only a resolved interruption gets the second one, since it is the
        # only case that knows the turn it is covering for was caused by somebody
        # being cut off rather than a question asked from silence.
        purpose = Purpose.RESUMING if acknowledge else Purpose.THINKING
        return speak_within(answer, lambda: self._filler(purpose), self.clock)

    async def _report(self, kind: str, text: str, generation: int) -> None:
        """Text for the client's transcript pane, dropped if the turn is stale.

        Without the generation check an interrupted turn keeps writing into the pane
        after the user has moved on, so the screen shows an answer to the previous
        question while the agent listens to the next one.
        """
        if not self._state.owns(generation):
            return
        if kind == ServerMessage.REPLY:
            # Counted here rather than at synthesis, because this is the text the listener
            # is about to hear and it is already past the grounding check.
            self.quality.sentence_spoken()
            # Staged, not committed. Whether the listener hears it is decided by the
            # transport, which is the whole point of the record's shape: text whose audio
            # is blocked or abandoned never enters the conversation at all.
            self.history.stage(generation, text)
        elif kind == ServerMessage.TRANSCRIPT:
            self.history.user_said(text)
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
            if not speaking.cancelled and self._state.state is State.SPEAKING:
                # `speaking.cancelled` first, because `chunks()` delivers the same
                # `None` sentinel whether production reached its own end or was cut
                # off, so this loop finishing is not by itself evidence of which one
                # happened. The state check stays too, since it is what actually
                # distinguishes an interruption, which has already moved the machine
                # out of SPEAKING by the time this runs. A disconnect cancels
                # `speaking` directly without touching the state first, which used to
                # read here as "reached the end", and a turn cut off after one chunk
                # of four was committed to history as the whole reply, delivered.
                self.quality.response_delivered()
                # The turn ran to the end, so all of its text was heard and enters the
                # record whole. Reached only here: an interruption has already moved the
                # machine out of SPEAKING, so it cannot arrive at this branch.
                self.history.commit(speaking.generation)
                self._state.to(State.LISTENING)
                self._endpointer.reset()
                # A suspicion the reply outlived, which the stuck-hold release makes
                # reachable. Left in place it carries its accumulated audio and its
                # `paused` flag into the next turn, so the next real interruption finds
                # one already open and never sends its PAUSE. Found while giving the
                # suspicion a clock: the clock made the leak a number rather than a
                # state nothing reported.
                self._barge = None
                self._retire_barge_clock(lambda clock: clock.mark_unresolved())
            else:
                self.quality.agent_stopped_speaking()
            await self._say({"type": ServerMessage.AUDIO_END})
            logger.info("session.quality", **self.quality.summary())

    async def _interrupt(self, heard: bytes = b"", outcome: str = "client") -> None:
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
        # Read before the playout estimate is dropped, and that order is the measurement.
        # What is still queued is what the listener never heard, so it decides where the
        # reply is cut, and `drop_playout` two lines down zeroes it.
        interrupted = self._state.generation
        self.history.truncate(interrupted, self._turns.playing_ms_remaining())

        self._turns.drop_playout()
        self._barge = None
        self._begin_listening()
        await self._stop_speaking()

        # Here, because this is the line `bench/stages.md` names: the generation has
        # moved and the in-flight generation and synthesis are stopped rather than
        # merely asked to stop. Recording it at the request instead would report the
        # interruption as complete while a synthesiser was still writing.
        self._retire_barge_clock(lambda clock: clock.mark_committed(outcome))

        # The audio that proved this was an interruption, handed to the turn it started.
        # Without it the new turn begins from whatever arrives after the decision, so the
        # agent hears an interruption from its second word onward and answers a fragment.
        if heard and self._frames is not None:
            for start in range(0, len(heard), FRAME_BYTES):
                self._frames.put_nowait(heard[start : start + FRAME_BYTES])

        await self._say({"type": ServerMessage.READY})

    def _retire_barge_clock(self, close: Callable[[BargeClock], None]) -> None:
        """Close out a suspicion and keep what it cost.

        One place, because there are three ways an interruption ends and a clock left
        open by any of them is a barge-in that never appears in the numbers. That is the
        empty-panel failure in miniature: the median improves because the slow cases
        stopped being counted.
        """
        clock = self._barge_clock
        if clock is None:
            return
        self._barge_clock = None
        close(clock)
        self.barges.append(clock)
        logger.info(
            "session.barge_latency",
            outcome=clock.outcome,
            server_paused_ms=_rounded(clock.server_paused_ms),
            committed_ms=_rounded(clock.committed_ms),
            resumed_ms=_rounded(clock.resumed_ms),
        )

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
        # Three states, not two. BLOCK is stale audio, WAIT is audio that is still the right
        # answer at the wrong moment: the user has started talking, so hold it rather than
        # speaking over them or throwing it away. Read from Bolna's `get_audio_send_status`.
        status = self._turns.status(chunk.generation, live={self._state.generation})
        if status is AudioStatus.WAIT:
            # Held until they stop, and bounded, because an unbounded poll on a flag is how a
            # turn hangs in silence. This exact loop spun forever the first time, because
            # nothing cleared `user_speaking` when the endpoint fired.
            #
            # The release repairs the flag rather than bypassing the gate, which is the
            # correction. Forcing this chunk through leaves the stuck state in place, so the
            # next chunk waits the full three seconds again and the whole answer arrives one
            # slice every three seconds. Clearing it once fixes every chunk after it.
            held = time.monotonic()
            while (
                self._turns.status(chunk.generation, live={self._state.generation})
                is AudioStatus.WAIT
            ):
                if time.monotonic() - held > STUCK_HOLD_S:
                    logger.warning("session.hold_released", generation=chunk.generation)
                    self._turns.note_user_stopped(count_turn=False)
                    break
                await asyncio.sleep(0.02)
            status = self._turns.status(chunk.generation, live={self._state.generation})

        if status is AudioStatus.BLOCK:
            return

        if not self._state.owns(chunk.generation):
            # Produced before the interruption and dequeued after it. Stopping the
            # producer does not unsend what it already handed on, so the last chance
            # to drop it is here.
            return

        if self._announced != chunk.generation:
            self._announced = chunk.generation
            # Here rather than where the chunk was produced. The agent starts speaking
            # when audio reaches the listener, and a chunk that was synthesised and then
            # blocked was never spoken.
            self.quality.agent_started_speaking()
            await self._say(
                {
                    "type": ServerMessage.AUDIO_START,
                    "mime": AUDIO_MIME,
                    "generation": chunk.generation,
                }
            )
            self._span_turn(chunk.generation)

        # Playout advances by the audio's own duration, not by how long the send took, because
        # a chunk starts playing when the previous one ends.
        #
        # The rate comes from the synthesiser rather than from a constant here. The constant
        # was 30000 bytes a second, five times the 6000 that 48 kbit/s actually is, so the
        # playout clock ran five times fast and the 900ms grace period expired after 180ms of
        # real speech. A number with no derivation in the code that reads it is a number
        # nobody can check.
        played = playout_seconds(chunk.data, self._bytes_per_second)

        # Read before this chunk is added, because what delays the answer is the audio
        # already queued in front of it, not its own length.
        if self.clock is not None and self.clock.first_answer_audio_ms is not None:
            self.clock.mark_heard(self._turns.playing_ms_remaining())
            # Answer audio only, so the record's estimate of what was heard is not
            # shifted by a filler clip the listener heard and that is not the reply.
            self.history.note_audio(chunk.generation, played)

        self._turns.note_audio_queued(played)

        # Filler counts toward the playout estimate and not toward the answer's numbers,
        # and the split is deliberate. For deciding whether to hold a chunk, filler is
        # audio the listener is hearing. For asking whether synthesis kept up with
        # playback, filler is not the reply, and counting it would flatter the ratio using
        # a file read off disk. `TurnClock` exists to keep exactly this distinction.
        if self.clock is not None and self.clock.first_answer_audio_ms is not None:
            self.quality.note_answer_audio(played)

        async with self._send:
            await self._transport.send_bytes(chunk.data)

    async def _say(self, payload: dict) -> None:
        async with self._send:
            await self._transport.send_json(payload)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


async def _drain(queue: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
    """Frames the receive loop has accepted, ending when it says the turn ended."""
    while True:
        frame = await queue.get()
        if frame is None:
            return
        yield frame
