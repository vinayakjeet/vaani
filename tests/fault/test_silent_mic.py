from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from vaani.endpoint import Endpointer, MicState
from vaani.protocol import FRAME_MS, ClientMessage, Frame, ServerMessage
from vaani.session import Incoming, VoiceSession

from ..vaani.test_endpoint import SILENCE, SPEECH, feed, tone

# A microphone that is working but too far away, or a room with a fan in it. Above
# digital zero and below the speech threshold, which is the whole point: this is the
# reading that must not be reported as a muted input.
ROOM_NOISE = tone(200)


def frames_of(pcm: bytes, ms: int) -> list[Incoming]:
    return [Incoming(frame=Frame(generation=1, pcm=pcm)) for _ in range(ms // FRAME_MS)]


class Client:
    def __init__(self, *incoming: Incoming) -> None:
        self._incoming = list(incoming)
        self.json: list[dict] = []
        self.audio: list[bytes] = []

    async def receive(self) -> Incoming:
        if self._incoming:
            return self._incoming.pop(0)
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")  # pragma: no cover

    async def send_json(self, payload: dict) -> None:
        self.json.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)

    def complaints(self) -> list[dict]:
        return [m for m in self.json if m.get("reason") == "microphone"]


async def answer(
    frames, still_current, on_transcript=None, on_sentence=None
) -> AsyncIterator[bytes]:
    async for _pcm in frames:
        pass
    yield b"audio"


async def filler(purpose=None) -> AsyncIterator[bytes]:
    yield b"achha"


async def drive(*incoming: Incoming, until, timeout: float = 3.0) -> Client:
    client = Client(Incoming(control=ClientMessage.START), *incoming)
    voice = VoiceSession(transport=client, answer=answer, filler=filler)
    task = asyncio.create_task(voice.run())
    try:
        await asyncio.wait_for(_until(lambda: until(client)), timeout=timeout)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return client


async def _until(predicate, interval: float = 0.002) -> None:
    while not predicate():
        await asyncio.sleep(interval)


def test_digital_silence_is_diagnosed_as_a_muted_input() -> None:
    """The reading a muted microphone or a denied permission produces. It is not a
    quiet room, it is no input at all, and the fix is different."""
    endpointer = Endpointer()
    feed(endpointer, SILENCE, endpointer.silence_timeout_ms)

    assert endpointer.diagnose() is MicState.SILENT


def test_room_noise_is_diagnosed_as_too_quiet_instead() -> None:
    """The microphone works and the speaker is too far away or too quiet. Reporting
    this as muted sends somebody to the wrong setting, which is why the two are
    separate states rather than one "no audio" error."""
    endpointer = Endpointer()
    feed(endpointer, ROOM_NOISE, endpointer.silence_timeout_ms)

    assert endpointer.diagnose() is MicState.TOO_QUIET


def test_nothing_is_diagnosed_before_the_timeout() -> None:
    """Somebody taking a moment to think must not be told their microphone is
    broken. That false complaint is the cost this threshold trades against."""
    endpointer = Endpointer()
    feed(endpointer, SILENCE, endpointer.silence_timeout_ms - FRAME_MS)

    assert endpointer.diagnose() is MicState.OK


def test_a_working_microphone_is_never_complained_about() -> None:
    """The control. A check that always fired would satisfy the tests above and make
    the demo unusable."""
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 600)
    feed(endpointer, SILENCE, endpointer.silence_timeout_ms)

    assert endpointer.diagnose() is MicState.OK


def test_a_pause_mid_turn_is_not_a_microphone_fault() -> None:
    """Once a turn is under way the trailing-silence timer owns the decision.
    Counting quiet against the microphone then would diagnose a pause as a fault."""
    endpointer = Endpointer()
    feed(endpointer, SPEECH, 600)

    assert feed(endpointer, SILENCE, endpointer.trailing_silence_ms)
    assert endpointer.diagnose() is MicState.OK


def test_the_diagnosis_resets_with_the_turn() -> None:
    endpointer = Endpointer()
    feed(endpointer, SILENCE, endpointer.silence_timeout_ms)
    assert endpointer.diagnose() is MicState.SILENT

    endpointer.reset()

    assert endpointer.diagnose() is MicState.OK


async def test_the_session_tells_the_client_the_microphone_is_muted() -> None:
    """SPEC S6, over the wire. The old behaviour discarded leading silence forever, so
    a muted microphone produced no turn and no complaint and the demo looked dead."""
    client = await drive(
        *frames_of(SILENCE, 6000), until=lambda c: c.complaints()
    )

    assert client.complaints()[0]["detail"] == MicState.SILENT


async def test_the_session_distinguishes_a_quiet_speaker() -> None:
    client = await drive(
        *frames_of(ROOM_NOISE, 6000), until=lambda c: c.complaints()
    )

    assert client.complaints()[0]["detail"] == MicState.TOO_QUIET


async def test_the_complaint_is_made_once_not_fifty_times_a_second() -> None:
    """Repeating it every frame would bury the message under itself."""
    client = await drive(
        *frames_of(SILENCE, 8000), until=lambda c: c.complaints()
    )
    await asyncio.sleep(0.05)

    assert len(client.complaints()) == 1


async def test_the_session_keeps_listening_after_complaining() -> None:
    """The usual fix is the user unmuting and speaking again, so the session must
    still be able to hear them."""
    client = await drive(
        *frames_of(SILENCE, 6000),
        *frames_of(SPEECH, 600),
        *frames_of(SILENCE, 800),
        until=lambda c: c.audio,
    )

    assert client.complaints()
    assert client.audio


async def test_a_working_session_is_never_complained_about() -> None:
    client = await drive(
        *frames_of(SPEECH, 600), *frames_of(SILENCE, 800), until=lambda c: c.audio
    )

    assert client.complaints() == []
    assert ServerMessage.ERROR not in [m.get("type") for m in client.json]


def checkins(client: Client) -> list[dict]:
    return [m for m in client.json if m.get("type") == ServerMessage.CHECKING_IN]


async def test_a_quiet_speaker_also_gets_a_check_in() -> None:
    """M2.15. A working mic that has heard nothing looks identical to a broken one
    from the far end unless something says otherwise. TOO_QUIET is the state that
    means the input is fine and the person may simply have gone quiet."""
    client = await drive(*frames_of(ROOM_NOISE, 6000), until=lambda c: checkins(c))

    assert checkins(client)


async def test_a_muted_microphone_does_not_also_get_a_check_in() -> None:
    """Asking "are you there" of someone who cannot be heard at all is the wrong
    message; SILENT already has its own, more specific, diagnostic."""
    client = await drive(
        *frames_of(SILENCE, 6000), *frames_of(SPEECH, 600), *frames_of(SILENCE, 800),
        until=lambda c: c.audio,
    )

    assert client.complaints()
    assert checkins(client) == []


async def test_the_check_in_is_said_once_not_every_frame() -> None:
    client = await drive(*frames_of(ROOM_NOISE, 8000), until=lambda c: checkins(c))
    await asyncio.sleep(0.05)

    assert len(checkins(client)) == 1
