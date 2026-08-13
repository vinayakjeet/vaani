from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from vaani.tts import (
    AUDIO_MIME,
    VOICE_EN_IN,
    FailingOverTts,
    FailoverReason,
    TtsError,
    failovers,
    speak_as_they_arrive,
)


class Voice:
    """A synthesiser that can fail before speaking, or partway through a sentence."""

    def __init__(
        self,
        name: str,
        chunks: int = 2,
        fail_after: int | None = None,
        mime: str = AUDIO_MIME,
    ) -> None:
        self.name = name
        self.mime = mime
        self.said: list[str] = []
        self.voices: list[str] = []
        self._chunks = chunks
        self._fail_after = fail_after

    async def synthesize(
        self, text: str, voice: str = "hi", index: int = 0
    ) -> AsyncIterator[bytes]:
        self.said.append(text)
        self.voices.append(voice)
        for chunk in range(self._chunks):
            if self._fail_after is not None and chunk >= self._fail_after:
                raise TtsError("edge-tts hung up")
            yield f"{self.name}:{text}:{chunk}".encode()


async def sentences(*texts: str) -> AsyncIterator[str]:
    for text in texts:
        yield text + " "


@pytest.fixture(autouse=True)
def clear_counters():
    failovers.clear()
    yield
    failovers.clear()


async def collect(tts, *texts: str) -> list[bytes]:
    return [chunk async for chunk in speak_as_they_arrive(sentences(*texts), tts)]


async def test_a_healthy_primary_never_reaches_the_fallback() -> None:
    """The control. A wrapper that always failed over would pass most of this file
    and would also throw away the voice the demo is built around."""
    primary, fallback = Voice("primary"), Voice("fallback")

    heard = await collect(FailingOverTts(primary, fallback), "Aap eligible hain.")

    assert fallback.said == []
    assert all(chunk.startswith(b"primary:") for chunk in heard)
    assert not failovers


async def test_a_primary_that_fails_before_speaking_leaves_no_artefact() -> None:
    """Nothing was heard yet, so the fallback simply speaks the sentence and a
    listener cannot tell anything happened."""
    primary = Voice("primary", fail_after=0)
    fallback = Voice("fallback")

    heard = await collect(FailingOverTts(primary, fallback), "Aap eligible hain.")

    assert fallback.said == ["Aap eligible hain."]
    assert all(chunk.startswith(b"fallback:") for chunk in heard)
    assert failovers[FailoverReason.BEFORE_AUDIO] == 1


async def test_a_sentence_cut_in_half_is_spoken_again_by_the_fallback() -> None:
    """It cannot be un-said, so the choice is a repeated clause or a missing one. For
    an eligibility answer a missing clause changes the answer and a repeat does not,
    so the whole sentence is spoken again."""
    primary = Voice("primary", chunks=4, fail_after=2)
    fallback = Voice("fallback")

    heard = await collect(FailingOverTts(primary, fallback), "Aapki aay seema se kam hai.")

    assert heard[:2] == [
        b"primary:Aapki aay seema se kam hai.:0",
        b"primary:Aapki aay seema se kam hai.:1",
    ]
    assert fallback.said == ["Aapki aay seema se kam hai."]
    assert failovers[FailoverReason.MID_SENTENCE] == 1


async def test_the_answer_continues_rather_than_restarting() -> None:
    """SPEC S7. Sentences already spoken are never synthesised again, which is the
    whole difference between continuing and starting the answer over."""
    primary = Voice("primary", chunks=2, fail_after=None)
    fallback = Voice("fallback")
    tts = FailingOverTts(primary, fallback)

    # First sentence succeeds, then the primary breaks for the second.
    heard_first = await collect(tts, "Pehla vaakya hai.")
    primary._fail_after = 0
    heard_second = await collect(tts, "Doosra vaakya hai.")

    assert "Pehla vaakya hai." not in fallback.said
    assert heard_first
    assert heard_second


async def test_failover_is_sticky_across_the_remaining_sentences() -> None:
    """Otherwise every later sentence pays a failed attempt first, and a provider that
    is down costs the whole answer its latency budget one sentence at a time."""
    primary = Voice("primary", fail_after=0)
    fallback = Voice("fallback")
    tts = FailingOverTts(primary, fallback)

    await collect(tts, "Pehla vaakya hai.", "Doosra vaakya hai.", "Teesra vaakya hai.")

    assert tts.failed_over
    assert len(primary.said) == 1
    assert len(fallback.said) == 3
    assert sum(failovers.values()) == 1


async def test_the_fallback_speaks_in_its_own_voice() -> None:
    """Which is what makes the change audible and visible. The trace shows a second
    synthesis span carrying a different `vaani.tts.voice`, and the failed attempt
    keeps its own span with `error.type`."""
    primary = Voice("primary", fail_after=0)
    fallback = Voice("fallback")

    await collect(FailingOverTts(primary, fallback), "Aap eligible hain.")

    assert fallback.voices == [VOICE_EN_IN]


async def test_both_voices_failing_still_raises() -> None:
    """Failover is not a way to hide an outage. Silence with no reason given is the
    outcome SPEC's degradation rules exist to prevent."""
    primary = Voice("primary", fail_after=0)
    fallback = Voice("fallback", fail_after=0)

    with pytest.raises(TtsError):
        await collect(FailingOverTts(primary, fallback), "Aap eligible hain.")


def test_a_fallback_with_a_different_codec_is_refused_at_construction() -> None:
    """The browser is handed one stream built from both providers, so a codec change
    partway through is a stream it cannot decode. Better a refusal here than a demo
    that plays half an answer."""
    with pytest.raises(TtsError):
        FailingOverTts(Voice("primary"), Voice("fallback", mime="audio/wav"))


async def test_the_sentence_index_survives_the_failover() -> None:
    """It is what separates sentence one's latency from sentence three's in the
    waterfall, and a failover must not collapse them."""
    primary = Voice("primary", fail_after=0)
    fallback = Voice("fallback")
    tts = FailingOverTts(primary, fallback)

    indices: list[int] = []

    class Recording(Voice):
        async def synthesize(self, text, voice="hi", index=0):
            indices.append(index)
            async for chunk in super().synthesize(text, voice, index):
                yield chunk

    tts._fallback = Recording("fallback")
    await collect(tts, "Pehla vaakya hai.", "Doosra vaakya hai.")

    assert indices == [1, 2]
