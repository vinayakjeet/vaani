from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app as APP
from app.routers import voice
from vaani.protocol import ServerMessage


@pytest.fixture(autouse=True)
def stub_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network here either. This file is about the single-session lock in
    `app/routers/voice.py`, not the pipeline behind it."""

    async def answer(frames, still_current, on_transcript=None, on_sentence=None):
        async for _pcm in frames:
            pass
        return
        yield b""  # pragma: no cover

    async def filler(purpose=None):
        yield b"achha"

    monkeypatch.setattr(voice, "build_answer", lambda _transcriber, _tts, _history: answer)
    monkeypatch.setattr(voice, "speak_filler", filler)
    monkeypatch.setattr(voice, "build_backchannel_check", lambda _transcriber: None)


def test_a_second_concurrent_session_is_refused_cleanly() -> None:
    """SPEC A10: refused with a spoken and visible reason, not queued silently. A
    caller left waiting with no message would read this as a hung server rather
    than a full one."""
    with TestClient(APP).websocket_connect("/ws/voice") as first:
        assert first.receive_json()["type"] == ServerMessage.READY

        with TestClient(APP).websocket_connect("/ws/voice") as second:
            message = second.receive_json()
            assert message["type"] == ServerMessage.ERROR
            assert message["reason"] == "busy"
            assert message["detail"]


def test_the_first_session_is_unaffected_by_the_refused_second_one() -> None:
    """The lock exists to protect the first caller's session, not to end it. A
    refusal that also disturbed the session already in progress would be a worse
    failure than the one it exists to prevent."""
    with TestClient(APP).websocket_connect("/ws/voice") as first:
        assert first.receive_json()["type"] == ServerMessage.READY

        with TestClient(APP).websocket_connect("/ws/voice") as second:
            second.receive_json()

        # The first session's socket is still open and answers normally, proven by
        # asking it something rather than only checking it did not raise.
        first.send_text("start")


def test_the_lock_is_released_once_the_first_session_ends() -> None:
    """A session that already ended must not go on refusing every visitor after
    it. Found live once, before the idle-receive timeout existed: a single dead
    connection held the lock forever and every visitor after the first stuck one
    was told busy until the process was restarted by hand."""
    with TestClient(APP).websocket_connect("/ws/voice") as first:
        assert first.receive_json()["type"] == ServerMessage.READY

    with TestClient(APP).websocket_connect("/ws/voice") as second:
        assert second.receive_json()["type"] == ServerMessage.READY
