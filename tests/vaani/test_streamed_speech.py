from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from vaani.tts import AUDIO_MIME, speak_as_they_arrive


class RecordingTts:
    """A synthesiser that records what it was asked to say, and when."""

    name = "recording"
    mime = AUDIO_MIME

    def __init__(self, tokens_seen: list[str] | None = None) -> None:
        self.said: list[str] = []
        self.said_after_tokens: list[int] = []
        self._tokens_seen = tokens_seen if tokens_seen is not None else []

    async def synthesize(self, text: str, voice: str = "hi-IN-SwaraNeural") -> AsyncIterator[bytes]:
        self.said.append(text)
        self.said_after_tokens.append(len(self._tokens_seen))
        yield b"audio:" + text.encode()


async def stream(tokens: list[str], seen: list[str] | None = None) -> AsyncIterator[str]:
    for token in tokens:
        if seen is not None:
            seen.append(token)
        # Yield to the loop so a consumer that is genuinely overlapping gets a
        # chance to run between tokens. Without this the generator could deliver
        # everything before the synthesiser is ever scheduled, and the ordering
        # assertion below would be about nothing.
        await asyncio.sleep(0)
        yield token


async def test_each_sentence_is_synthesised_separately() -> None:
    tts = RecordingTts()

    audio = [chunk async for chunk in speak_as_they_arrive(
        stream(["Aap ", "eligible ", "hain. ", "Aapko ", "6000 ", "milega."]), tts
    )]

    assert tts.said == ["Aap eligible hain.", "Aapko 6000 milega."]
    assert audio == [b"audio:Aap eligible hain.", b"audio:Aapko 6000 milega."]


async def test_the_first_sentence_is_spoken_before_the_reply_is_finished() -> None:
    """The acceptance criterion for M1.4, and it is about ordering rather than content.

    Asserting only which sentences were synthesised would pass against an
    implementation that buffers the whole reply and then splits it, which is the
    M0 baseline this milestone exists to beat. What matters is that synthesis of
    sentence one started while later tokens had not arrived yet.
    """
    seen: list[str] = []
    tts = RecordingTts(tokens_seen=seen)
    tokens = ["Aap ", "eligible ", "hain. ", "Aapko ", "6000 ", "milega."]

    async for _chunk in speak_as_they_arrive(stream(tokens, seen), tts):
        break

    assert tts.said_after_tokens[0] < len(tokens)


async def test_a_reply_with_no_terminator_is_still_spoken() -> None:
    """Otherwise the listener gets silence, which is the one outcome worse than an
    awkward break."""
    tts = RecordingTts()

    tokens = stream(["Aap eligible hain lekin"])
    audio = [chunk async for chunk in speak_as_they_arrive(tokens, tts)]

    assert tts.said == ["Aap eligible hain lekin"]
    assert audio


async def test_an_empty_reply_synthesises_nothing() -> None:
    """A synthesis request for an empty string spends a round trip to produce no
    audio, and some providers reject it outright."""
    tts = RecordingTts()

    assert [chunk async for chunk in speak_as_they_arrive(stream([]), tts)] == []
    assert tts.said == []


async def test_whitespace_only_output_synthesises_nothing() -> None:
    tts = RecordingTts()

    assert [chunk async for chunk in speak_as_they_arrive(stream(["  ", "\n"]), tts)] == []
    assert tts.said == []
