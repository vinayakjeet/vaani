"""The call site no test could reach, which is why it was the broken one.

`GroqWhisper.transcribe` needs an API key and a network, so every other test
replaced it with a fake. It recorded a transcript length on the request span where
that attribute was not declared, the contract raised, and a live turn died with
`UndeclaredAttribute` while the user heard nothing.

Nothing about that needed a key. The provider takes an `httpx.AsyncClient`, so a
mock transport exercises the whole method including the span, which is what these
tests do.
"""

from __future__ import annotations

import httpx
import pytest

from vaani.spans import CONTRACT, SHARED_ATTRIBUTES, STT_REQUEST
from vaani.stt import GroqWhisper, SttError


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def transcriber(handler) -> GroqWhisper:
    return GroqWhisper(
        api_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def replying(payload: dict, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


AUDIO = b"\x01\x00" * 16000


async def test_a_transcript_comes_back(exported) -> None:
    impl = transcriber(replying({"text": " kya main eligible hoon ", "language": "hi"}))

    result = await impl.transcribe(AUDIO)

    assert result.text == "kya main eligible hoon"
    assert result.language == "hi"
    assert result.streaming is False


async def test_the_request_span_carries_only_declared_attributes(exported) -> None:
    """The regression. Every attribute this method sets is checked against the table,
    on the real method rather than on a fake that never had one."""
    impl = transcriber(replying({"text": "haan", "model": "whisper-large-v3-turbo"}))

    await impl.transcribe(AUDIO)

    span = next(s for s in exported.get_finished_spans() if s.name == STT_REQUEST)
    allowed = set(CONTRACT[STT_REQUEST]) | SHARED_ATTRIBUTES

    assert span.attributes
    for key in span.attributes:
        assert key in allowed, f"{key} is not declared on {STT_REQUEST}"


async def test_the_audio_length_is_recorded_not_the_audio(exported) -> None:
    impl = transcriber(replying({"text": "haan"}))

    await impl.transcribe(AUDIO)

    span = next(s for s in exported.get_finished_spans() if s.name == STT_REQUEST)

    assert span.attributes["vaani.stt.audio_ms"] == round(len(AUDIO) / 32)


async def test_an_empty_transcript_is_refused() -> None:
    """A model handed nothing answers something. Silence transcribed to an empty
    string must not become a question."""
    impl = transcriber(replying({"text": "   "}))

    with pytest.raises(SttError):
        await impl.transcribe(AUDIO)


async def test_a_provider_error_reports_the_status_and_not_the_body() -> None:
    """A provider error body can quote the audio's transcript back, and that is user
    speech going into a log line this project promises never to write."""
    impl = transcriber(replying({"error": "kya main eligible hoon"}, status=500))

    with pytest.raises(SttError) as raised:
        await impl.transcribe(AUDIO)

    assert "500" in str(raised.value)
    assert "eligible" not in str(raised.value)


async def test_a_missing_key_fails_before_the_network() -> None:
    impl = GroqWhisper(api_key=None)

    with pytest.raises(SttError):
        await impl.transcribe(AUDIO)


async def test_the_audio_is_sent_as_a_wav_file() -> None:
    """Groq's endpoint takes a file, so raw PCM would be rejected as an unsupported
    format, which surfaces as a 400 with no obvious cause."""
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(200, json={"text": "haan"})

    await transcriber(handler).transcribe(AUDIO)

    assert b"RIFF" in seen[0]
    assert b"WAVE" in seen[0]


async def test_an_injected_client_survives_the_call_for_reuse() -> None:
    """`ChunkedStt` calls `transcribe` five to eleven times for one utterance, and
    the whole point of handing it a shared client is that the connection survives
    between calls. The default path closes what it opened; an injected one must
    not, or the second partial reopens exactly the connection the first one just
    proved it did not need to."""
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"text": "haan"}))
    )
    impl = GroqWhisper(api_key="test-key", client=client)

    await impl.transcribe(AUDIO)
    assert not client.is_closed

    # And actually usable a second time, not merely unclosed by accident.
    second = await impl.transcribe(AUDIO)
    assert second.text == "haan"
    assert not client.is_closed
