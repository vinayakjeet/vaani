from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest

import vaani.sarvam as sarvam_module
from vaani.sarvam import SarvamBulbul, SarvamSaaras
from vaani.stt import SttError
from vaani.tts import TtsError


@pytest.fixture(autouse=True)
def short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real grace window is sized for a live network round trip. Every test
    here scripts its own responses up front, so nothing is ever actually
    pending; shortening this is what keeps the suite fast rather than paying
    the real default on every test that reaches the end of a stream."""
    monkeypatch.setattr(sarvam_module, "RESPONSE_GRACE_S", 0.1)


@dataclass
class _TranscriptData:
    transcript: str | None


@dataclass
class _Message:
    """The two response shapes `_transcript_of` has to tell apart: a real
    transcript (`type="data"`) and anything else, a VAD signal or an error
    frame, which carries no transcript at all."""

    type: str
    data: _TranscriptData | None = None


def data_message(transcript: str) -> _Message:
    return _Message(type="data", data=_TranscriptData(transcript=transcript))


def events_message() -> _Message:
    return _Message(type="events")


@dataclass
class FakeSocket:
    """The server's own responses arrive on the server's own schedule, not one
    per `transcribe()` call, which is the real protocol shape found live. Push
    scripted responses in with `push`; `__aiter__` is what `pump` actually reads,
    matching the real SDK socket client's own interface."""

    sent: list[bytes] = field(default_factory=list)
    flushed: bool = False
    closed: bool = False
    fail_send: bool = False
    _inbox: asyncio.Queue = field(default_factory=asyncio.Queue)

    def push(self, message: _Message) -> None:
        self._inbox.put_nowait(message)

    async def transcribe(self, *, audio: str, encoding: str, sample_rate: int) -> None:
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append(base64.b64decode(audio))

    async def flush(self) -> None:
        self.flushed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._inbox.get()


def connect_to(socket: FakeSocket):
    @asynccontextmanager
    async def _connect():
        try:
            yield socket
        finally:
            socket.closed = True

    return _connect


async def frames(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


FRAME = b"\x00\x40" * 160  # one 20ms frame


async def drain(stream: AsyncIterator) -> list:
    return [item async for item in stream]


async def test_partials_arrive_as_pushed_by_the_server() -> None:
    """Not paired to sends: every response the server pushes becomes a partial,
    in arrival order, regardless of how many `transcribe()` calls preceded it."""
    socket = FakeSocket()
    socket.push(data_message("kisi"))
    socket.push(events_message())
    socket.push(data_message("kisi yojana ke baare"))
    saaras = SarvamSaaras(api_key="k", interval_ms=20, connect=connect_to(socket))

    partials = await drain(saaras.stream(frames(FRAME, FRAME)))

    assert [p.text for p in partials] == ["kisi", "kisi yojana ke baare", "kisi yojana ke baare"]
    assert [p.final for p in partials] == [False, False, True]
    assert socket.flushed
    assert socket.closed


async def test_only_new_audio_is_sent_each_time() -> None:
    """SPEC A4's whole point: a genuinely streaming stack sends incremental
    audio, not the whole buffer again on every interval."""
    socket = FakeSocket()
    saaras = SarvamSaaras(api_key="k", interval_ms=20, connect=connect_to(socket))

    await drain(saaras.stream(frames(FRAME, FRAME)))

    assert len(socket.sent) == 2
    assert len(socket.sent[1]) > len(socket.sent[0])
    assert len(socket.sent[1]) - len(socket.sent[0]) < len(socket.sent[0])


async def test_a_failed_send_does_not_end_the_stream() -> None:
    """Same rule as the free stack: a partial send failing is not fatal, since
    the next interval's send carries the same audio forward."""

    @dataclass
    class FlakyOnceSocket(FakeSocket):
        async def transcribe(self, *, audio: str, encoding: str, sample_rate: int) -> None:
            if not self.sent:
                self.sent.append(b"marker")
                raise RuntimeError("first send fails")
            await super().transcribe(audio=audio, encoding=encoding, sample_rate=sample_rate)

    flaky = FlakyOnceSocket()
    flaky.push(data_message("final text"))
    saaras = SarvamSaaras(api_key="k", interval_ms=20, connect=connect_to(flaky))

    partials = await drain(saaras.stream(frames(FRAME)))

    assert [p.text for p in partials] == ["final text", "final text"]
    assert partials[-1].final


async def test_missing_api_key_raises_before_connecting() -> None:
    saaras = SarvamSaaras(api_key=None, connect=lambda: (_ for _ in ()).throw(AssertionError()))

    with pytest.raises(SttError):
        await drain(saaras.stream(frames(FRAME)))


async def test_final_is_empty_when_nothing_was_ever_transcribed() -> None:
    socket = FakeSocket()
    saaras = SarvamSaaras(api_key="k", interval_ms=20, connect=connect_to(socket))

    partials = await drain(saaras.stream(frames(FRAME)))

    assert partials[-1].text == ""
    assert partials[-1].final


async def test_bulbul_streams_audio_chunks() -> None:
    class FakeSdkClient:
        class text_to_speech:
            @staticmethod
            async def convert_stream(**kwargs):
                for chunk in (b"a", b"b", b"c"):
                    yield chunk

    bulbul = SarvamBulbul(api_key="k", client=FakeSdkClient())

    chunks = [c async for c in bulbul.synthesize("Namaste", "hi-IN")]

    assert chunks == [b"a", b"b", b"c"]


async def test_bulbul_with_no_audio_raises() -> None:
    class EmptySdkClient:
        class text_to_speech:
            @staticmethod
            async def convert_stream(**kwargs):
                return
                yield b""  # pragma: no cover

    bulbul = SarvamBulbul(api_key="k", client=EmptySdkClient())

    with pytest.raises(TtsError):
        async for _chunk in bulbul.synthesize("Namaste", "hi-IN"):
            pass


async def test_bulbul_missing_api_key_raises() -> None:
    bulbul = SarvamBulbul(api_key=None)

    with pytest.raises(TtsError):
        async for _chunk in bulbul.synthesize("Namaste", "hi-IN"):
            pass
