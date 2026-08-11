from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from llm import ChatClient, ChatMessage
from vaani.protocol import FRAME_BYTES
from vaani.stt import SttError, Transcript
from vaani.turn import SYSTEM_PROMPT, Turn

AUDIO = b"\x01\x02" * (FRAME_BYTES // 2) * 50


class StubStt:
    name = "stub"
    streaming = False

    def __init__(self, text: str = "PM Kisan ke liye eligibility kya hai") -> None:
        self.text = text
        self.received: bytes | None = None

    async def transcribe(self, pcm: bytes) -> Transcript:
        self.received = pcm
        return Transcript(text=self.text, language="hi", provider=self.name, streaming=False)


class StubTts:
    name = "stub"
    mime = "audio/mpeg"

    def __init__(self, chunks: int = 3) -> None:
        self.chunks = chunks
        self.spoken: str | None = None

    async def synthesize(self, text: str, voice: str = "hi") -> AsyncIterator[bytes]:
        self.spoken = text
        for n in range(self.chunks):
            yield f"chunk{n}".encode()


@pytest.fixture
def turn() -> Turn:
    return Turn(stt=StubStt(), tts=StubTts(), llm=ChatClient(), provider="mock")


async def test_a_turn_produces_a_transcript_a_reply_and_audio(turn: Turn) -> None:
    """M0.4's acceptance, with the providers stubbed. The deployed version of
    this is a person speaking Hindi and hearing Hindi back; this is the part CI
    can run without a microphone or a paid key."""
    result = await turn.run(AUDIO)

    assert result.transcript.text
    assert result.reply
    assert result.audio
    assert result.audio_mime == "audio/mpeg"


async def test_the_audio_is_every_chunk_joined() -> None:
    """Unstreamed means the caller gets one blob. Losing a chunk here truncates
    the answer mid-word, and a listener cannot scroll back to check."""
    tts = StubTts(chunks=4)
    result = await Turn(stt=StubStt(), tts=tts, llm=ChatClient(), provider="mock").run(AUDIO)

    assert result.audio == b"chunk0chunk1chunk2chunk3"


async def test_the_model_speaks_the_transcript_not_the_audio() -> None:
    """The reply is synthesised from what the model said, not from what the user
    said. Obvious until a refactor passes the wrong variable and the agent reads
    the question back."""
    tts = StubTts()
    stt = StubStt(text="kya main eligible hoon")
    result = await Turn(stt=stt, tts=tts, llm=ChatClient(), provider="mock").run(AUDIO)

    assert tts.spoken == result.reply
    assert tts.spoken != stt.text


async def test_the_audio_reaches_the_transcriber_unchanged(turn: Turn) -> None:
    stt = StubStt()
    await Turn(stt=stt, tts=StubTts(), llm=ChatClient(), provider="mock").run(AUDIO)

    assert stt.received == AUDIO


async def test_a_transcription_failure_stops_the_turn() -> None:
    """It must not reach the model with an empty question and answer confidently.
    A voice agent that answers a question nobody asked is worse than one that
    says it did not hear."""

    class Failing(StubStt):
        async def transcribe(self, pcm: bytes) -> Transcript:
            raise SttError("empty transcript")

    with pytest.raises(SttError):
        await Turn(stt=Failing(), tts=StubTts(), llm=ChatClient(), provider="mock").run(AUDIO)


def test_the_system_prompt_asks_for_short_spoken_answers() -> None:
    """The single highest-leverage latency decision in the project, and it is a
    prompt rather than code. A model that writes six sentences costs six
    sentences of synthesis before the listener hears the end."""
    assert "three short sentences" in SYSTEM_PROMPT
    assert "spoken aloud" in SYSTEM_PROMPT


def test_the_system_prompt_forbids_inventing_scheme_details() -> None:
    """This answers eligibility questions about welfare payments. A hallucinated
    income limit is not a wrong answer, it is a person not applying."""
    assert "Never invent" in SYSTEM_PROMPT


async def test_the_conversation_carries_the_system_prompt(turn: Turn) -> None:
    """Guards the two tests above, which assert on a constant that the turn is
    free to ignore."""
    sent: list[ChatMessage] = []
    original = turn._llm.complete

    async def capture(provider: str, messages: list[ChatMessage], **kwargs: object):
        sent.extend(messages)
        return await original(provider, messages, **kwargs)

    turn._llm.complete = capture
    await turn.run(AUDIO)

    assert sent[0].role == "system"
    assert sent[0].content == SYSTEM_PROMPT
