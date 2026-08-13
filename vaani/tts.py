"""Text to speech, streamed as chunks so playback can start early.

Chunked from the start even in M0, where nothing overlaps yet, because the
interface is what M1.4's first-sentence flush and M3.3's mid-utterance failover
are built on. An interface that returns one finished audio blob would have to be
replaced rather than extended, and the fallback path would never have been
exercised until the day it was needed.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import StrEnum
from typing import Protocol

import structlog

from vaani.sentences import from_stream
from vaani.spans import TTS_SYNTHESIZE, stage_span

logger = structlog.get_logger(__name__)

# Hindi voices. Neural voices from this provider handle Devanagari and read
# Latin-script Hinglish acceptably, which matters because half the eval set is
# code-mixed and a voice that spells out English words ruins the demo.
VOICE_HI = "hi-IN-SwaraNeural"
VOICE_EN_IN = "en-IN-NeerjaNeural"

# Output is MP3 from this provider. The browser decodes it; the alternative is
# raw PCM at three times the bytes over a socket that is also carrying the user's
# microphone in the other direction.
AUDIO_MIME = "audio/mpeg"

# How many bytes of this provider's output is one second of speech. 48 kbit/s at a
# constant bitrate, which is the only format edge-tts emits, so a byte count maps to
# playback time exactly rather than approximately.
#
# It belongs to the provider rather than to the caller, and that is the correction.
# The session estimated playout as `len(chunk) / 30000`, a number with no derivation
# anywhere, which ran the playout clock five times too fast: the grace period that is
# supposed to hold audio for 900ms of real speech expired after 180ms of it. Bolna
# refuses to derive duration from byte length for compressed audio at all, because a
# variable bitrate makes the sum wrong without saying so. Ours is constant and known,
# so the honest version is for the provider to declare its rate and for a mismatched
# fallback to be refused at construction.
EDGE_BYTES_PER_SECOND = 48_000 // 8


class TtsError(Exception):
    """Synthesis failed or produced nothing playable."""


async def speak_as_they_arrive(
    tokens: AsyncIterator[str],
    tts: TtsProvider,
    voice: str = VOICE_HI,
    on_sentence: Callable[[str], Awaitable[None]] | None = None,
    validate: Callable[[str], str] | None = None,
) -> AsyncIterator[bytes]:
    """Audio for each sentence as soon as the token stream completes that sentence.

    The overlap the milestone is named for: sentence one is being synthesised, and
    then played, while the model is still writing sentence three. The unstreamed
    baseline in `vaani/turn.py` waits for the whole reply before it starts, and the
    gap between the two is a row in the ablation.

    Each sentence is its own synthesis request, so the audio arrives as a sequence
    of separate encoded streams rather than one. Browsers play them back to back,
    and whether the joins are audible is an open question the M1.8 note already
    raises. It is a quality cost paid for milliseconds, so the ablation reports it
    rather than only the milliseconds.
    """
    sentences = 0
    async for sentence in from_stream(tokens):
        sentences += 1

        # The guardrail sits here rather than after synthesis, which is the only place it
        # can sit. Audio cannot be un-said, so a check that runs on a finished reply is a
        # check that runs after the wrong number has been spoken.
        spoken = sentence if validate is None else validate(sentence)
        if not spoken.strip():
            continue

        if on_sentence is not None:
            # Reported before synthesis, so the transcript pane shows the words at
            # the moment they exist rather than after the audio for them arrives. The
            # validated text, not the model's, so the pane cannot show a figure the
            # listener was deliberately not told.
            await on_sentence(spoken)
        async for chunk in tts.synthesize(spoken, voice, index=sentences):
            yield chunk

    logger.info("tts.stream_done", provider=tts.name, sentences=sentences)


class TtsProvider(Protocol):
    name: str
    mime: str
    # Bytes of output per second of speech, so a caller can turn a chunk into a
    # duration without knowing the codec. Zero means the provider cannot say, which a
    # variable-bitrate encoder cannot, and a caller must then not guess.
    bytes_per_second: int

    def synthesize(self, text: str, voice: str, index: int = 0) -> AsyncIterator[bytes]: ...


def _rate(provider: object) -> int:
    """A provider's encoded byte rate, or zero when it does not declare one."""
    return int(getattr(provider, "bytes_per_second", 0) or 0)


def playout_seconds(data: bytes, bytes_per_second: int) -> float:
    """How long this chunk takes to play, or zero when that cannot be derived.

    Zero rather than an estimate. A confidently wrong duration is worse than an
    absent one here, because the audio gate's grace period and the playout estimate
    both trust it, and both fail quietly in the direction of talking over somebody.
    """
    if bytes_per_second <= 0:
        return 0.0
    return len(data) / bytes_per_second


class EdgeTts:
    """The free stack's voice. No key, no quota, and no service guarantee.

    Being unofficial is the point of M3.3: this is the provider most likely to
    fail mid-utterance in the real world, so the failover path it forces is not
    a hypothetical.
    """

    name = "edge-tts"
    mime = AUDIO_MIME
    bytes_per_second = EDGE_BYTES_PER_SECOND

    async def synthesize(
        self, text: str, voice: str = VOICE_HI, index: int = 0
    ) -> AsyncIterator[bytes]:
        import edge_tts

        started = time.monotonic()
        with stage_span(
            TTS_SYNTHESIZE,
            **{
                "gen_ai.system": self.name,
                "vaani.tts.voice": voice,
                "vaani.tts.chars": len(text),
                "vaani.tts.sentence_index": index,
            },
        ) as stage:
            chunks = 0
            try:
                async for chunk in edge_tts.Communicate(text, voice).stream():
                    if chunk["type"] != "audio":
                        continue
                    if chunks == 0:
                        # The number the whole project is about. Time to first
                        # audio is what a listener experiences as responsiveness,
                        # and it is not the span's duration, which is how long
                        # synthesis took.
                        stage.record(
                            **{
                                "vaani.tts.first_chunk_ms": (time.monotonic() - started) * 1000
                            }
                        )
                        stage.mark("vaani.tts.first_chunk")
                    chunks += 1
                    yield chunk["data"]
            except Exception as exc:
                raise TtsError(f"{type(exc).__name__}") from exc

            stage.record(**{"vaani.tts.chunks": chunks})

        if chunks == 0:
            raise TtsError("no audio produced")
        logger.info("tts.done", provider=self.name, voice=voice, chunks=chunks)


# Why a sentence had to be spoken by the fallback. A closed set, for the same reason
# the recogniser's reasons are: a metric label built from an exception message grows
# without bound.
class FailoverReason(StrEnum):
    BEFORE_AUDIO = "before_audio"
    MID_SENTENCE = "mid_sentence"


failovers: Counter[str] = Counter()


class FailingOverTts:
    """A synthesiser with a second voice behind it. SPEC S7.

    The primary here is an unofficial service with no uptime promise, which is why
    M3.3 exists at all: it is the provider most likely to fail partway through a
    reply, so the fallback path is not a hypothetical.

    Failover is sticky. Once the primary has failed, the remaining sentences go
    straight to the fallback rather than each paying a failed attempt first, which is
    what SPEC S7 means by the remaining sentences being synthesised by the fallback.

    Two shapes of failure, and they get different treatment because the listener has
    already heard different amounts:

    - Nothing was emitted for this sentence yet, so the fallback speaks it and there
      is no artefact at all.
    - Some of the sentence was already spoken. It cannot be un-said, so the fallback
      speaks the whole sentence again. A listener hears part of a clause repeated,
      which is jarring and complete, where continuing from the next sentence would
      drop the rest of this one. For an eligibility answer a missing clause changes
      the answer and a repeated clause does not, so the repeat is the lesser harm.
      The ablation reports the rate rather than the choice.

    Previous sentences are never resynthesised. That is the difference between
    continuing and restarting the answer.
    """

    def __init__(
        self,
        primary: TtsProvider,
        fallback: TtsProvider,
        fallback_voice: str = VOICE_EN_IN,
    ) -> None:
        if primary.mime != fallback.mime:
            # The browser is handed one stream of audio built from both providers, so
            # a codec change partway through is a stream it cannot decode. Better a
            # refusal at construction than a demo that plays half an answer.
            raise TtsError(
                f"fallback speaks {fallback.mime} and the primary speaks {primary.mime}; "
                "playback cannot switch codecs mid-answer"
            )
        # Read defensively. A provider written before this field existed should degrade to
        # "cannot say", which stops the playout estimate, rather than taking a live turn down
        # with an AttributeError from inside a constructor. Instrumentation that throws turns
        # a missing field into an outage, which this project has already paid for once.
        if _rate(primary) != _rate(fallback):
            # Same argument one level down. The playout estimate is a byte count
            # divided by a rate, and after a mid-answer failover the remaining chunks
            # would be measured at the wrong rate, so the gate would hold or release
            # audio on a clock that silently changed halfway through the reply.
            raise TtsError(
                f"fallback encodes at {_rate(fallback)} bytes/s and the primary at "
                f"{_rate(primary)}; playout could not be timed across the switch"
            )

        self._primary = primary
        self._fallback = fallback
        self._fallback_voice = fallback_voice
        self._failed_over = False
        self.name = f"{primary.name}+{fallback.name}"
        self.mime = primary.mime
        self.bytes_per_second = _rate(primary)

    @property
    def failed_over(self) -> bool:
        return self._failed_over

    async def synthesize(
        self, text: str, voice: str = VOICE_HI, index: int = 0
    ) -> AsyncIterator[bytes]:
        if self._failed_over:
            async for chunk in self._fallback.synthesize(text, self._fallback_voice, index):
                yield chunk
            return

        spoken = 0
        try:
            async for chunk in self._primary.synthesize(text, voice, index):
                spoken += 1
                yield chunk
            return
        except TtsError:
            reason = (
                FailoverReason.MID_SENTENCE if spoken else FailoverReason.BEFORE_AUDIO
            )
            self._failed_over = True
            failovers[reason] += 1
            # The reason and the sentence number, never the sentence. The voice change
            # is visible in the trace as a second `tts.synthesize` span with a
            # different `vaani.tts.voice`, and the failed attempt keeps its own span
            # carrying `error.type`.
            logger.warning(
                "tts.failed_over",
                primary=self._primary.name,
                fallback=self._fallback.name,
                reason=str(reason),
                sentence_index=index,
            )

        async for chunk in self._fallback.synthesize(text, self._fallback_voice, index):
            yield chunk
