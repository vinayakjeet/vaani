from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable

from vaani.protocol import ClientMessage, Frame, ServerMessage
from vaani.session import Incoming, VoiceSession

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


async def filler() -> AsyncIterator[bytes]:
    yield b"achha"


def session(*incoming: Incoming | str, chunks=None, delay: float = 0.0):
    transport = FakeTransport(control(ClientMessage.START), *incoming)
    return (
        VoiceSession(
            transport=transport,
            answer=answering(chunks if chunks is not None else [b"a1", b"a2"], delay),
            filler=filler,
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


async def test_speech_during_playback_interrupts_without_a_control_message() -> None:
    """A client that never sends INTERRUPT still gets interrupted. The one that does
    gets there sooner, which is the only difference."""
    # Still generation 1: a client that simply talks over the agent has not started
    # a new turn of its own, so its frames carry the turn it is interrupting.
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        *[frame(SPEECH, 1) for _ in range(30)],
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    task = await run_until_audio(voice, transport)
    await asyncio.wait_for(_until(lambda: voice._state.interrupted_previous), 2.0)
    await stop(task)

    assert len(transport.sent_audio) < 4


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


async def test_stop_ends_the_session() -> None:
    voice, transport = session(control(ClientMessage.STOP))

    await asyncio.wait_for(voice.run(), timeout=2.0)

    assert transport.kinds() == [ServerMessage.READY]


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
