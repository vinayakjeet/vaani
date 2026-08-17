from __future__ import annotations

import asyncio

import vaani.session as session_module
from vaani.protocol import ServerMessage
from vaani.session import Incoming

from ..vaani.test_session import AFTER_AUDIO, session, speech_then_silence, stop


async def run_to_audio_end(voice, transport) -> asyncio.Task:
    """`voice.run()` as a background task, stopped once `_play` has actually
    finished a turn rather than only started one. `FakeTransport.receive()` blocks
    once its script runs out, which every turn-completion test here relies on: a
    turn nothing interrupts still leaves `run()` waiting for the next utterance,
    not returned, so waiting for `run()` itself to finish is the wrong signal.
    `AUDIO_END` is `_play`'s own last act before that wait begins."""
    task = asyncio.create_task(voice.run())
    for _ in range(200):
        if ServerMessage.AUDIO_END in transport.kinds():
            return task
        await asyncio.sleep(0.01)
    raise AssertionError("AUDIO_END never arrived")  # pragma: no cover


async def test_a_disconnect_after_partial_playback_does_not_commit_the_reply_as_delivered() -> None:
    """A disconnect after one chunk of four used to read, at `_play`'s own finally,
    exactly like playback reaching its natural end: `speaking.chunks()` delivers the
    same `None` sentinel either way, and the session's own state was still SPEAKING
    because nothing had moved it, which is the same signal an ordinary completion
    leaves. The reply entered history as fully heard. `SpeakingTurn.cancelled` is
    the fix: the one signal that actually distinguishes reaching the end from being
    cut off partway through it."""
    voice, transport = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        Incoming(disconnected=True),
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )

    await asyncio.wait_for(voice.run(), timeout=2.0)

    assert len(transport.sent_audio) < 4
    assert not any(m.role == "assistant" for m in voice.history.messages)


async def test_a_disconnect_before_any_audio_leaves_nothing_staged() -> None:
    """The other end of the same gap: a disconnect that lands before the answer has
    produced a single chunk must not commit an empty assistant turn either."""
    voice, transport = session(
        *speech_then_silence(),
        Incoming(disconnected=True),
        chunks=[b"a1", b"a2"],
        delay=5.0,
    )

    await asyncio.wait_for(voice.run(), timeout=2.0)

    assert not any(m.role == "assistant" for m in voice.history.messages)


async def test_a_turn_that_actually_finishes_still_commits() -> None:
    """The regression this file exists to prevent is `_play` never committing again,
    not only never committing on a disconnect. A turn nothing interrupts must still
    enter history whole, or the fix for one bug becomes another."""
    voice, transport = session(*speech_then_silence(), chunks=[b"a1", b"a2"])

    task = await asyncio.wait_for(run_to_audio_end(voice, transport), timeout=2.0)
    await stop(task)

    assert any(m.role == "assistant" for m in voice.history.messages)


async def test_reconnecting_gets_a_session_with_nothing_carried_over() -> None:
    """M3.6's own acceptance: reconnecting starts a clean session. `VoiceSession`
    builds a fresh `Conversation` unless one is passed in, which is what
    `app/routers/voice.py` relies on for every new socket; a disconnected
    session's own history, whatever it holds, must never be the same object a
    session built after it reads or writes."""
    first, _ = session(
        *speech_then_silence(),
        AFTER_AUDIO,
        Incoming(disconnected=True),
        chunks=[b"a1", b"a2", b"a3", b"a4"],
        delay=0.02,
    )
    await asyncio.wait_for(first.run(), timeout=2.0)

    second, second_transport = session(*speech_then_silence(), chunks=[b"a1", b"a2"])
    second_task = await asyncio.wait_for(run_to_audio_end(second, second_transport), timeout=2.0)
    await stop(second_task)

    assert second.history is not first.history
    assert not any(m.role == "assistant" for m in first.history.messages)
    assert any(m.role == "assistant" for m in second.history.messages)


async def test_an_idle_disconnect_also_ends_the_socket_cleanly(monkeypatch) -> None:
    """`app/routers/voice.py` releases its single-session lock as soon as `run`
    returns, via the `async with` around it; nothing here can verify that lock
    directly without the router, but `run` returning at all, rather than hanging
    or raising, is the property the lock depends on. A connection that never sends
    anything after START, not even a clean disconnect, is the case `run`'s own idle
    timeout exists for."""
    monkeypatch.setattr(session_module, "IDLE_RECEIVE_TIMEOUT_S", 0.05)
    voice, transport = session()

    await asyncio.wait_for(voice.run(), timeout=2.0)

    assert ServerMessage.READY in transport.kinds()
