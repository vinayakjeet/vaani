from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.main import app as APP
from app.routers import voice
from vaani.protocol import FRAME_BYTES, Frame, ServerMessage, encode
from vaani.stt import Transcript
from vaani.turn import Turn

SILENCE = b"\x00" * FRAME_BYTES
# Loud enough to clear the endpointer's threshold, as PCM16 sample pairs.
SPEECH = (b"\x00\x40" * (FRAME_BYTES // 2))


class StubStt:
    name = "stub"
    streaming = False

    async def transcribe(self, pcm: bytes) -> Transcript:
        return Transcript(text="kya main eligible hoon", language="hi",
                          provider=self.name, streaming=False)


class StubTts:
    name = "stub"
    mime = "audio/mpeg"

    async def synthesize(self, text: str, voice: str = "hi") -> AsyncIterator[bytes]:
        yield b"audio-bytes"


@pytest.fixture(autouse=True)
def stub_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network in the socket tests. A test that needs a paid key is a test
    that runs on one machine, and the framing is what this file is about."""
    from llm import ChatClient

    monkeypatch.setattr(
        voice,
        "build_turn",
        lambda: Turn(stt=StubStt(), tts=StubTts(), llm=ChatClient(), provider="mock"),
    )


def speak(socket, frames: int, generation: int = 1) -> None:
    for _ in range(frames):
        socket.send_bytes(encode(Frame(generation=generation, pcm=SPEECH)))


def go_quiet(socket, frames: int, generation: int = 1) -> None:
    for _ in range(frames):
        socket.send_bytes(encode(Frame(generation=generation, pcm=SILENCE)))


def test_the_socket_announces_itself_before_any_audio() -> None:
    """A client that starts streaming before the server is ready loses the front
    of the utterance, which is the part carrying the question word."""
    with TestClient(APP).websocket_connect("/ws/voice") as socket:
        assert socket.receive_json()["type"] == ServerMessage.READY


def test_speech_then_silence_produces_a_transcript_and_audio() -> None:
    """M0.4 end to end over the real socket: frames in, spoken answer out."""
    with TestClient(APP).websocket_connect("/ws/voice") as socket:
        assert socket.receive_json()["type"] == ServerMessage.READY
        socket.send_text("start")
        speak(socket, 30)
        go_quiet(socket, 40)

        transcript = socket.receive_json()
        reply = socket.receive_json()
        audio_start = socket.receive_json()
        audio = socket.receive_bytes()

        assert transcript["type"] == ServerMessage.TRANSCRIPT
        assert transcript["session_id"]
        assert reply["type"] == ServerMessage.REPLY
        assert audio_start["mime"] == "audio/mpeg"
        assert audio == b"audio-bytes"


def test_silence_alone_never_produces_a_turn() -> None:
    """An open microphone in a quiet room must not transcribe the room. Without
    this the agent answers questions nobody asked, every few seconds, forever."""
    with TestClient(APP).websocket_connect("/ws/voice") as socket:
        socket.receive_json()
        socket.send_text("start")
        go_quiet(socket, 200)
        socket.send_text("stop")


def test_a_frame_from_an_abandoned_turn_is_dropped() -> None:
    """The generation check. Audio already on the wire when the user restarted
    must not be transcribed as part of the new question."""
    with TestClient(APP).websocket_connect("/ws/voice") as socket:
        socket.receive_json()
        socket.send_text("start")
        speak(socket, 30, generation=1)
        socket.send_text("start")
        speak(socket, 5, generation=1)
        go_quiet(socket, 40, generation=2)
        socket.send_text("stop")


def test_a_misframed_payload_is_reported_not_swallowed() -> None:
    """Dropping ragged frames silently means the transcript degrades and the
    model gets blamed for a framing bug in the client."""
    with TestClient(APP).websocket_connect("/ws/voice") as socket:
        socket.receive_json()
        socket.send_bytes(b"\x01\x00" + SILENCE[:-4])

        error = socket.receive_json()
        assert error["type"] == ServerMessage.ERROR
        assert "bytes" in error["reason"]
