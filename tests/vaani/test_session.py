from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable

import pytest

from vaani import session as session_module
from vaani.endpoint import DEFAULT_TRAILING_SILENCE_MS
from vaani.fillers import Purpose
from vaani.protocol import FRAME_MS, ClientMessage, Frame, ServerMessage
from vaani.session import Incoming, VoiceSession
from vaani.stt import SttError

from .test_endpoint import SILENCE, SPEECH


class FakeTransport:
    """A scripted client. No socket, no microphone, so it runs in CI.

    Incoming messages are handed out in order; once they run out it blocks, which is
    what a real client does between utterances and what lets a test observe playback
    happening while the loop is still receiving.
    """

    def __init__(self, *incoming: Incoming | str) -> None:
        self._incoming = list(incoming)
        self.sent_json: list[dict] = []
        self.sent_audio: list[bytes] = []
        self.audio_gate = asyncio.Event()

    async def receive(self) -> Incoming:
        # AFTER_AUDIO models the only interruption that exists in reality: a person
        # who has heard something and talks over it. Delivering an interrupt before
        # any audio has been produced tests a race nobody can perform, and it was
        # what made these tests fail against correct code.
        while self._incoming and self._incoming[0] == AFTER_AUDIO:
            self._incoming.pop(0)
            await self.audio_gate.wait()
        if self._incoming:
            return self._incoming.pop(0)
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")  # pragma: no cover

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_audio.append(data)
        self.audio_gate.set()

    def kinds(self) -> list[str]:
        return [str(message.get("type")) for message in self.sent_json]


AFTER_AUDIO = "after-audio"


def frame(pcm: bytes, generation: int = 1) -> Incoming:
    return Incoming(frame=Frame(generation=generation, pcm=pcm))


def control(action: str) -> Incoming:
    return Incoming(control=action)


def speech_then_silence(generation: int = 1) -> list[Incoming]:
    """Enough speech to start a turn, then enough silence to end it."""
    speaking = [frame(SPEECH, generation) for _ in range(30)]
    quiet = [frame(SILENCE, generation) for _ in range(60)]
    return speaking + quiet


def answering(chunks: list[bytes], delay: float = 0.0):
    async def answer(
        frames: AsyncIterator[bytes],
        still_current: Callable[[], bool],
        on_transcript=None,
        on_sentence=None,
    ) -> AsyncIterator[bytes]:
        # Drain the frames the way the real pipeline does, so the queue does not
        # fill and the endpoint path is exercised.
        async for _pcm in frames:
            pass
        if on_transcript is not None:
            await on_transcript("kya main eligible hoon")
        if on_sentence is not None:
            await on_sentence("Haan, aap eligible hain.")
        for chunk in chunks:
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    return answer


async def filler(purpose=None) -> AsyncIterator[bytes]:
    yield b"achha"


def recording_filler():
    """A filler that remembers which purpose it was asked for each time.

    The default `filler` fixture ignores its argument, which is right for every test
    that does not care what speech act was chosen. This is for the one that does.
    """
    calls: list = []

    async def record(purpose=None) -> AsyncIterator[bytes]:
        calls.append(purpose)
        yield b"achha"

    return record, calls


def session(
    *incoming: Incoming | str,
    chunks=None,
    delay: float = 0.0,
    backchannel: bool | Exception | None = None,
    filler=filler,
):
    """A session over a fake transport.

    `backchannel` is the verifier's verdict, so a test can say what the interrupting
    utterance turned out to be without a recogniser. None means no verifier at all, which
    is the unverified arm: a suspected interruption commits on duration alone.
    """
    transport = FakeTransport(control(ClientMessage.START), *incoming)

    async def verdict(_pcm: bytes) -> bool:
        if isinstance(backchannel, Exception):
            raise backchannel
        return bool(backchannel)

    return (
        VoiceSession(
            transport=transport,
            answer=answering(chunks if chunks is not None else [b"a1", b"a2"], delay),
            filler=filler,
            bytes_per_second=6000,
            backchannel=None if backchannel is None else verdict,
        ),
        transport,
    )


async def run_until_audio(voice: VoiceSession, transport: FakeTransport) -> asyncio.Task:
    task = asyncio.create_task(voice.run())
    await asyncio.wait_for(transport.audio_gate.wait(), timeout=2.0)
    return task


async def stop(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_the_client_gets_the_transcript_the_reply_and_an_audio_start() -> None:
    """The browser shows what it heard and what was said, and opens its playback
    buffer on `audio_start`. Dropping any of the three leaves the pane blank or the
    audio unplayed, and `web/index.html` handles exactly these."""
    voice, transport = session(*speech_then_silence())

    task = await run_until_audio(voice, transport)
    await stop(task)

    kinds = transport.kinds()

    assert kinds.index(ServerMessage.TRANSCRIPT) < kinds.index(ServerMessage.REPLY)
    assert kinds.index(ServerMessage.REPLY) < kinds.index(ServerMessage.AUDIO_START)
    assert transport.sent_json[kinds.index(ServerMessage.AUDIO_START)]["mime"]


async def test_audio_start_is_announced_once_per_turn() -> None:
    """Twice splices the answer in the client's buffer; never leaves it unplayed."""
    voice, transport = session(
        *speech_then_silence(), chunks=[b"a1", b"a2", b"a3"], delay=0.01
    )

    task = await run_until_audio(voice, transport)
    await asyncio.sleep(0.1)
    await stop(task)

    assert transport.kinds().count(ServerMessage.AUDIO_START) == 1


async def test_text_from_an_interrupted_turn_stops_reaching_the_pane() -> None:
    """Otherwise the screen shows an answer to the previous question while the agent
    is already listening to the next one."""
    voice, transport = session(*speech_then_silence())
    task = await run_until_audio(voice, transport)

    voice._state.begin()
    before = len(transport.sent_json)
    await voice._report(ServerMessage.REPLY, "stale", voice._state.generation - 1)
    await stop(task)

    assert len(transport.sent_json) == before


async def test_a_turn_produces_audio() -> None:
    voice, transport = session(*speech_then_silence())

    task = await run_until_audio(voice, transport)
    await stop(task)

    assert transport.sent_audio[0] == b"a1"
    assert ServerMessage.READY in transport.kinds()


async def test_an_interrupt_is_read_while_audio_is_still_going_out() -> None:
    """The defect M2.5 exists to fix, as a test.

    The old handler awaited its own sends, so an interrupt sat unread until the
    whole answer had gone. With a slow answer this test would hang rather than fail
    if receiving were still blocked behind playback.
    """
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        control(ClientMessage.INTERRUPT),
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: ServerMessage.READY in transport.kinds()[1:]), 2.0)
    await stop(task)

    assert transport.sent_audio
    assert len(transport.sent_audio) < 4


async def test_an_interrupt_leaves_the_session_listening_and_clean() -> None:
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        control(ClientMessage.INTERRUPT),
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await stop(task)

    assert voice._state.interrupted_previous
    assert voice._speaking is None


async def test_sustained_speech_during_playback_interrupts_without_a_control_message() -> None:
    """A client that never sends INTERRUPT still gets interrupted. The one that does
    gets there sooner, which is the only difference.

    Long enough to clear the commit threshold, because that is the point of the
    threshold: nobody backchannels for that long, so this needs no transcript to be
    certain and does not pay a round trip to find out."""
    # Still generation 1: a client that simply talks over the agent has not started
    # a new turn of its own, so its frames carry the turn it is interrupting.
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await stop(task)

    assert len(transport.sent_audio) < 4


async def test_a_short_backchannel_pauses_playback_and_then_resumes_it() -> None:
    """"haan" over a reply must cost nothing. Killing playback on every acknowledgement
    is what made the agent feel neurotic, and enumerating the acknowledgements is a
    losing game in Hinglish, so the short ones are held and identified instead."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        # Twenty frames of speech is 400ms: past the pause threshold, under the commit
        # one. Then silence, so the utterance ends and gets identified.
        *[frame(SPEECH, 1) for _ in range(20)],
        *[frame(SILENCE, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
        backchannel=True,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: ServerMessage.RESUME in transport.kinds()), 2.0)
    await asyncio.sleep(0.2)
    await stop(task)

    kinds = transport.kinds()
    assert kinds.index(ServerMessage.PAUSE) < kinds.index(ServerMessage.RESUME)
    # The turn survived. A backchannel that abandons the reply has cost the user the
    # answer they were agreeing with.
    assert not voice._state.interrupted_previous
    # Every chunk of the answer, not merely most of them. The held audio is delivered
    # after the resume rather than dropped, which is the difference between WAIT and BLOCK.
    answer_chunks = [chunk for chunk in transport.sent_audio if chunk != b"achha"]
    assert answer_chunks == [b"a1", b"a2", b"a3", b"a4"]


async def test_a_short_utterance_that_was_not_a_backchannel_still_interrupts() -> None:
    """The other half of the same decision. Short and content-bearing is a real
    interruption, and the verifier is what tells the two apart."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(20)],
        *[frame(SILENCE, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
        backchannel=False,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await stop(task)

    assert ServerMessage.RESUME not in transport.kinds()


async def test_a_resolved_interruption_acknowledges_rather_than_stalls() -> None:
    """M2.8. Once the interrupting utterance has ended and the verifier says it was a
    real question, the turn that answers it opens with "ji, boliye" rather than "ek
    minute": the listener has already been told the floor is theirs, and repeating the
    thinking phrase would read as the agent not having heard that its own turn began."""
    record, calls = recording_filler()
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(20)],
        *[frame(SILENCE, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
        backchannel=False,
        filler=record,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await asyncio.wait_for(_until(lambda: len(calls) >= 2), 2.0)
    await stop(task)

    # The first turn, asked from silence, gets the ordinary one; the second, which
    # exists because somebody was cut off and re-identified, gets the other.
    assert calls[0] is Purpose.THINKING
    assert calls[1] is Purpose.RESUMING


async def test_a_duration_interruption_does_not_acknowledge() -> None:
    """The hazard M2.8 was built around. On the duration path the user is still
    mid-sentence when the turn is abandoned, so an acknowledgement here would be the
    agent talking over them: exactly the `agent_interrupted_user` failure M4.12 counts.
    The turn keeps listening rather than re-answering immediately, so there is no
    second filler call to mislabel in the first place."""
    record, calls = recording_filler()
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
        filler=record,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await asyncio.sleep(0.1)
    await stop(task)

    assert Purpose.RESUMING not in calls


async def test_a_recogniser_that_cannot_say_commits_the_interruption() -> None:
    """A failed check is not evidence of a backchannel. Committing costs a turn the user
    repeats; resuming over somebody who really did interrupt is the failure the whole
    mechanism exists to prevent."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(20)],
        *[frame(SILENCE, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
        backchannel=SttError("provider returned 503"),
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await stop(task)

    assert ServerMessage.RESUME not in transport.kinds()


async def test_the_interrupting_audio_starts_the_turn_it_caused() -> None:
    """Without this the new turn begins from whatever arrives after the decision, so the
    agent hears an interruption from its second word onward and answers a fragment."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    def carried() -> bool:
        return voice._frames is not None and voice._frames.qsize() > 0

    task = await run_until_audio(voice, transport)
    # Waited for directly, not sampled once after a weaker precondition.
    # `_interrupt` sets `interrupted_previous` before it awaits `_stop_speaking`, and
    # the frames are only queued after that await completes, so a check gated on the
    # flag alone can land in the real gap between the two and read an empty queue that
    # was never going to stay empty. This flaked on a machine busy enough to widen
    # that gap, and it was the test racing its own assertion, not the code being wrong.
    await asyncio.wait_for(_until(carried), 2.0)
    frames_carried = voice._frames.qsize() if voice._frames is not None else 0
    await stop(task)

    assert voice._state.interrupted_previous
    assert frames_carried > 0


async def test_audio_from_an_abandoned_turn_is_never_sent() -> None:
    """Produced before the interruption and dequeued after it. Stopping the producer
    does not unsend what it already handed on, so the last chance to drop it is at
    the transport."""
    # Slow and multi-chunk on purpose. With a fast two-chunk answer the producer has
    # already finished by the time the generation moves, so there is nothing in
    # flight and the test passes whether or not the check exists.
    voice, transport = session(
        *speech_then_silence(), chunks=[b"a1", b"a2", b"a3", b"a4", b"a5"], delay=0.02
    )
    task = await run_until_audio(voice, transport)

    voice._state.begin()
    before = len(transport.sent_audio)
    await asyncio.sleep(0.15)
    await stop(task)

    assert before < 5
    assert len(transport.sent_audio) == before


async def test_frames_from_a_stale_generation_are_ignored() -> None:
    voice, transport = session(frame(SPEECH, 99), frame(SPEECH, 99))
    task = asyncio.create_task(voice.run())
    await asyncio.sleep(0.05)
    await stop(task)

    assert transport.sent_audio == []


def spans_named(exported, name: str) -> list:
    return [span for span in exported.get_finished_spans() if span.name == name]


async def test_a_turn_emits_its_vad_endpoint_span(exported) -> None:
    """M1.6's actual gap: the span was declared in the contract and defined in
    `bench/stages.md`, and nothing anywhere ever created one."""
    voice, transport = session(*speech_then_silence())

    task = await run_until_audio(voice, transport)
    await stop(task)

    spans = spans_named(exported, "vad.endpoint")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["vaani.vad.speech_ms"] > 0
    assert attrs["vaani.vad.trailing_silence_ms"] == voice._endpointer.trailing_silence_ms


async def test_a_turn_emits_its_turn_span_with_a_positive_duration(exported) -> None:
    """Backdated to the first frame of speech, not created at request time, so its
    duration includes the endpoint wait the same way `TurnClock` already does."""
    voice, transport = session(*speech_then_silence())

    task = await run_until_audio(voice, transport)
    await stop(task)

    spans = spans_named(exported, "turn")
    assert len(spans) == 1
    assert spans[0].attributes["vaani.turn.index"] == 1
    assert spans[0].attributes["vaani.turn.interrupted"] is False
    assert spans[0].end_time > spans[0].start_time


class QueueTransport:
    """Like `FakeTransport`, except more can be fed in after a wait.

    `FakeTransport`'s script is a plain list: once exhausted, `receive` sleeps for an
    hour, and appending to the list after that does not wake it, since nothing is
    watching the list. This one blocks on an `asyncio.Queue`, so a `feed` from the
    test after confirming something else has genuinely finished is delivered rather
    than raced against a queue whose length the test cannot see.
    """

    def __init__(self, *incoming: Incoming | str) -> None:
        self._incoming: asyncio.Queue[Incoming] = asyncio.Queue()
        self.feed(incoming)
        self.sent_json: list[dict] = []
        self.sent_audio: list[bytes] = []
        self.audio_gate = asyncio.Event()

    def feed(self, items) -> None:
        for item in items:
            self._incoming.put_nowait(item)

    async def receive(self) -> Incoming:
        return await self._incoming.get()

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_audio.append(data)
        self.audio_gate.set()

    def kinds(self) -> list[str]:
        return [str(message.get("type")) for message in self.sent_json]


async def test_a_second_turn_increments_the_turn_index(exported) -> None:
    """The session's own generation counter does not move between two ordinary,
    uninterrupted turns; only `begin()` advances it, on a barge-in or a fresh START.
    Both turns here are tagged with the one generation the session actually owns
    throughout, which is the property `_state.owns` checks incoming frames against.

    Turn 2's frames are fed only once turn 1's `AUDIO_END` has actually been sent.
    The first version of this test used `AFTER_AUDIO`, a gate already satisfied by
    turn 1's first chunk, so the scripted frames behind it arrived while the agent
    was still SPEAKING and were read as a barge-in on turn 1 rather than a second,
    ordinary one.

    Turn 1's own silence tail is trimmed to just past `trailing_silence_ms` rather
    than reusing `speech_then_silence`'s generous 60 frames: whatever is left
    unconsumed when the endpoint fires is still ahead of turn 2's frames in the same
    queue, and 60 frames of backlog was enough to make this test wait on real
    wall-clock time it did not need to spend.
    """
    trailing_frames = DEFAULT_TRAILING_SILENCE_MS // FRAME_MS + 2
    turn = [frame(SPEECH, 1) for _ in range(30)]
    turn += [frame(SILENCE, 1) for _ in range(trailing_frames)]
    transport = QueueTransport(control(ClientMessage.START), *turn)
    voice = VoiceSession(
        transport=transport, answer=answering([b"a1", b"a2"]), filler=filler,
        bytes_per_second=6000,
    )

    task = asyncio.create_task(voice.run())
    await asyncio.wait_for(transport.audio_gate.wait(), timeout=2.0)
    await asyncio.wait_for(_until(lambda: ServerMessage.AUDIO_END in transport.kinds()), 2.0)
    transport.feed(turn)
    await asyncio.wait_for(_until(lambda: len(spans_named(exported, "turn")) >= 2), 5.0)
    await stop(task)

    indices = sorted(span.attributes["vaani.turn.index"] for span in spans_named(exported, "turn"))
    assert indices == [1, 2]


async def test_a_turn_restarted_by_start_marks_interrupted_previous(exported) -> None:
    """The endpointer runs normally on the new turn's own speech here, unlike the
    verified-interruption path, so this is the case that actually exercises
    `vaani.turn.interrupted=True` on a span with a real start time."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        control(ClientMessage.START),
        *speech_then_silence(generation=2),
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: len(spans_named(exported, "turn")) >= 2), 2.0)
    await stop(task)

    by_index = sorted(spans_named(exported, "turn"), key=lambda s: s.attributes["vaani.turn.index"])
    assert by_index[1].attributes["vaani.turn.interrupted"] is True


async def test_a_resolved_interruptions_turn_span_is_skipped_not_fabricated(exported) -> None:
    """The interrupting audio is written straight into the frame queue in `_interrupt`,
    bypassing `accept`, so the endpointer has no backdated start to give this turn. A
    span with a start time invented for the occasion would be worse than none."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(20)],
        *[frame(SILENCE, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
        backchannel=False,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await asyncio.sleep(0.1)
    await stop(task)

    # One turn span: the first, ordinary one. The second, born from the resolved
    # interruption, is the one with nothing to backdate from.
    assert len(spans_named(exported, "turn")) == 1


async def test_stop_ends_the_session() -> None:
    voice, transport = session(control(ClientMessage.STOP))

    await asyncio.wait_for(voice.run(), timeout=2.0)

    assert transport.kinds() == [ServerMessage.READY]


async def test_an_idle_connection_eventually_gives_up(monkeypatch) -> None:
    """A connection can go quiet without ever sending a clean disconnect: a dropped
    link, a frozen tab, a laptop put to sleep. Left alone, `receive()` would await a
    socket that is never producing another message, indistinguishable from a client
    who is merely quiet, and the one-at-a-time lock in `app/routers/voice.py` would
    stay held until the process is restarted by hand. Found live: exactly this held a
    demo unusable for every visitor after the first stuck one.

    The bound has to be on the wait itself, not on a per-turn timer, because a session
    between utterances is legitimately waiting on `receive()` for as long as the
    visitor takes to speak again."""
    monkeypatch.setattr(session_module, "IDLE_RECEIVE_TIMEOUT_S", 0.05)
    voice, transport = session()

    await asyncio.wait_for(voice.run(), timeout=2.0)

    assert ServerMessage.READY in transport.kinds()
    assert voice._speaking is None


async def test_a_disconnect_abandons_the_turn_rather_than_finishing_it() -> None:
    """A half-finished turn left running holds a provider stream open on a free
    instance, which is a quota somebody is counting."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        Incoming(disconnected=True),
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    await asyncio.wait_for(voice.run(), timeout=2.0)

    assert voice._speaking is None
    assert len(transport.sent_audio) < 4


async def test_a_playback_failure_is_reported_not_swallowed() -> None:
    """A turn that goes silent with no reason given is the failure SPEC's degradation
    rules exist to prevent."""

    async def broken(
        frames, still_current, on_transcript=None, on_sentence=None
    ) -> AsyncIterator[bytes]:
        async for _pcm in frames:
            pass
        yield b"a1"
        raise RuntimeError("tts died")

    transport = FakeTransport(control(ClientMessage.START), *speech_then_silence())
    voice = VoiceSession(transport=transport, answer=broken, filler=filler)

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(
        _until(lambda: ServerMessage.ERROR in transport.kinds()), 2.0
    )
    await stop(task)

    assert ServerMessage.ERROR in transport.kinds()


async def test_the_clock_starts_when_the_endpoint_fires() -> None:
    """From the last frame of user speech, which is when the listener started
    waiting. Not at request dispatch, which is the clock that flatters."""
    voice, transport = session(*speech_then_silence())

    task = await run_until_audio(voice, transport)
    await stop(task)

    assert voice.clock is not None
    assert voice.clock.first_answer_audio_ms is not None


async def test_the_clock_includes_the_silence_the_user_sat_through() -> None:
    """The project's headline number, and the easiest one to flatter.

    The endpoint fires only after the trailing silence has been waited out, so a clock
    started there removes that wait from every figure: several hundred milliseconds,
    in our favour. SPEC picks the harder clock deliberately, because most published
    voice latency starts after endpointing and two systems quoting the same number can
    differ by a factor of two in the only thing a listener perceives.

    So first audio can never be reported as arriving sooner than the silence that
    ended the turn.
    """
    voice, transport = session(*speech_then_silence())

    task = await run_until_audio(voice, transport)
    await stop(task)

    assert voice.clock is not None
    assert voice.clock.first_answer_audio_ms is not None
    assert voice.clock.first_answer_audio_ms >= voice._endpointer.trailing_silence_ms - 40


async def test_the_transport_is_actually_driven() -> None:
    """Guards the tests above, which all read what the transport received. A session
    that never sent anything would satisfy several of them."""
    voice, transport = session(*speech_then_silence())

    task = await run_until_audio(voice, transport)
    await stop(task)

    assert transport.sent_json
    assert transport.sent_audio


async def _until(predicate, interval: float = 0.005) -> None:
    while not predicate():
        await asyncio.sleep(interval)


async def test_a_released_hold_does_not_delay_every_later_chunk(monkeypatch) -> None:
    """The property the old bound missed. Forcing one chunk through unblocks that chunk and
    leaves the flag set, so the next one waits the full timeout again and a forty-chunk
    reply arrives one slice every three seconds, with every log line blaming the hold
    rather than the flag nothing cleared."""
    monkeypatch.setattr(session_module, "STUCK_HOLD_S", 0.05)

    voice, transport = session(
        *speech_then_silence(), chunks=[b"a1", b"a2", b"a3"], delay=0.0
    )
    # A flag set by a path nothing clears, which is exactly how this arose: speech was
    # noted during playback and the endpoint that would have cleared it never fired.
    voice._turns.note_user_speaking()

    task = asyncio.create_task(voice.run())
    await asyncio.wait_for(_until(lambda: len(transport.sent_audio) >= 3), 2.0)
    await stop(task)

    # One release, not one per chunk: the flag is repaired rather than stepped over.
    assert not voice._turns.user_speaking


async def test_a_real_barge_in_shows_up_in_the_interactivity_numbers() -> None:
    """Wired to the events, not only unit-tested beside them. A metrics object nothing
    calls reports a healthy conversation forever, which is the shape of check this
    project keeps paying for."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await stop(task)

    assert voice.quality.user_interrupted_agent == 1
    # The reply was cut off and never finished, so it is not a recovery.
    assert voice.quality.recovery_rate == 0.0
    assert voice.quality.longest_agent_monologue_ms > 0


async def test_an_ignored_backchannel_is_counted_and_the_reply_still_counts_as_delivered() -> None:
    """A backchannel must not spend the recovery denominator. The user did not interrupt,
    so a session full of "haan" would otherwise read as a session full of barge-ins the
    agent never recovered from."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(20)],
        *[frame(SILENCE, 1) for _ in range(60)],
        chunks=[b"a1", b"a2"],
        delay=0.02,
        backchannel=True,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: ServerMessage.RESUME in transport.kinds()), 2.0)
    await asyncio.sleep(0.2)
    await stop(task)

    assert voice.quality.backchannels_ignored == 1
    assert voice.quality.user_interrupted_agent == 0
    assert voice.quality.recovery_rate is None


async def test_filler_audio_is_kept_out_of_the_answers_speed_ratio() -> None:
    """The filler is a file read off disk, so counting it as synthesised audio would
    flatter the one number that says whether synthesis kept up with playback."""
    # Slow enough that the answer misses the deadline, so the filler definitely speaks.
    # With an instant answer it does not, and the test would pass without exercising the
    # exclusion at all.
    voice, transport = session(*speech_then_silence(), chunks=[b"a1", b"a2"], delay=0.02)

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: len(transport.sent_audio) >= 3), 2.0)
    await stop(task)

    assert b"achha" in transport.sent_audio
    assert voice.quality.answer_audio_s > 0
    assert voice.quality.answer_audio_s == pytest.approx(len(b"a1" + b"a2") / 6000)


async def test_the_answers_heard_time_includes_the_filler_queued_in_front_of_it() -> None:
    """The measurement bug the filler bank exposed. `speak_within` yields the whole
    filler before the answer's first chunk, and audio leaves the process far faster than
    real time, so the send time is a moment at which the listener is still hearing "ek
    minute". Judging the target on it flatters every turn the filler fired on, which is
    most of them, because the deadline is shorter than the endpoint wait."""
    voice, transport = session(*speech_then_silence(), chunks=[b"a1", b"a2"], delay=0.02)

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: len(transport.sent_audio) >= 3), 2.0)
    await stop(task)

    clock = voice.clock
    assert clock.filler_spoken
    # The filler here is five bytes, so at 6000 bytes a second it is under a millisecond
    # of playout. The property under test is that the heard time is never earlier than
    # the send time, and carries the queue in front of it.
    assert clock.first_answer_heard_ms >= clock.first_answer_audio_ms
    assert clock.target_ms == clock.first_answer_heard_ms


async def test_a_real_filler_clip_pushes_the_answer_past_the_target() -> None:
    """With the committed bank rather than a five byte stub. The shortest clip is 1.78
    seconds of playout, so an answer queued behind one cannot meet a 500ms target however
    fast the pipeline was, and the honest number says so."""
    from vaani.fillers import FillerBank, Purpose, read_chunks

    clip = FillerBank().pick(Purpose.THINKING)
    assert clip is not None
    chunks = read_chunks(clip, 720)

    async def real_filler(purpose=None) -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    transport = FakeTransport(control(ClientMessage.START), *speech_then_silence())
    voice = VoiceSession(
        transport=transport,
        answer=answering([b"a1", b"a2"], delay=0.02),
        filler=real_filler,
        bytes_per_second=6000,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: b"a2" in transport.sent_audio), 2.0)
    await stop(task)

    assert voice.clock.first_answer_heard_ms > 1500
    assert not voice.clock.met_p50_target


async def test_a_completed_reply_enters_the_conversation_record() -> None:
    """The control for the test below. A record that never commits would pass that one
    and would also mean the agent remembers nothing it said."""
    voice, transport = session(*speech_then_silence())

    task = await run_until_audio(voice, transport)
    await asyncio.sleep(0.1)
    await stop(task)

    roles = [m.role for m in voice.history.messages]
    assert roles == ["user", "assistant"]
    assert voice.history.messages[-1].content == "Haan, aap eligible hain."


async def test_an_interrupted_reply_is_not_remembered_as_fully_spoken() -> None:
    """The failure this record was shaped around. Text staged when the model produced it
    would leave the agent believing it said a sentence nobody heard, and coherence fails
    two turns later when it refers back to a figure it never spoke."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(60)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await stop(task)

    said = [m.content for m in voice.history.messages if m.role == "assistant"]
    # Either nothing was heard, or a prefix of it was. What must never appear is the
    # whole sentence, because the audio for its tail was thrown away.
    assert said in ([], [""]) or not said[0].endswith("eligible hain.")
