"""Every stage span name and `vaani.*` attribute, in one place.

SPEC's integration contract lists the span tree and the attributes each span
carries. This module is that table as code, and `stage_span` refuses an attribute
the table does not declare, so the contract is enforced where attributes are
written rather than only asserted afterwards. A name that drifts fails the first
test that touches it instead of quietly producing a panel that renders nothing.

`bench/stages.md` says what each of these spans starts and ends at. Read it before
adding one. A span whose boundaries are not written down measures whatever the
code happened to wrap, which in the previous project meant publishing a queue wait
as latency.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import spanlight
from opentelemetry.trace import Span, Status, StatusCode

TURN = "turn"
VAD_ENDPOINT = "vad.endpoint"
STT_STREAM = "stt.stream"
STT_REQUEST = "stt.request"
LLM_GENERATE = "llm.generate"
TTS_SYNTHESIZE = "tts.synthesize"
PLAYBACK_FIRST_AUDIO = "playback.first_audio"

# Span name to the `vaani.*` attributes it may carry. Attributes from a shared
# convention (`gen_ai.*`, `spanlight.*`) are set by Spanlight itself and are not
# listed: this table owns the names this project invented, which are the ones with
# nobody else to keep them honest.
CONTRACT: dict[str, frozenset[str]] = {
    TURN: frozenset({"vaani.turn.index", "vaani.turn.interrupted"}),
    VAD_ENDPOINT: frozenset(
        {
            "vaani.vad.speech_ms",
            "vaani.vad.trailing_silence_ms",
            "vaani.vad.aggressiveness",
        }
    ),
    STT_STREAM: frozenset(
        {
            "vaani.stt.partials",
            "vaani.stt.final_chars",
            "vaani.stt.streaming",
        }
    ),
    STT_REQUEST: frozenset({"vaani.stt.index", "vaani.stt.audio_ms"}),
    LLM_GENERATE: frozenset({"vaani.llm.rounds", "vaani.llm.first_token_ms"}),
    TTS_SYNTHESIZE: frozenset(
        {
            "vaani.tts.voice",
            "vaani.tts.sentence_index",
            "vaani.tts.chars",
            "vaani.tts.chunks",
            "vaani.tts.first_chunk_ms",
        }
    ),
    PLAYBACK_FIRST_AUDIO: frozenset({"vaani.playback.queued_ms", "vaani.playback.reported"}),
}

# Counts and durations only. No attribute in the table above holds a transcript, a
# reply, or a tool argument, and this test-enforced rule is why: audio and the text
# derived from it are what SPEC's threat model keeps out of exported spans, and an
# attribute added in a hurry is how that promise gets broken. Character counts say
# what an operator needs and cannot be read back.
FORBIDDEN_SUBSTRINGS = ("text", "transcript", "reply", "content", "prompt", "args")

# Names from shared conventions, allowed on any stage. They are not in the table
# above because that table owns what this project invented; these belong to the
# GenAI semantic convention and to Spanlight, which is where their meaning is
# defined and kept.
SHARED_ATTRIBUTES = frozenset({"gen_ai.system", "gen_ai.response.model", "error.type"})


class UndeclaredAttribute(KeyError):
    """An attribute this span is not allowed to carry, per SPEC's table."""


class StageSpan:
    """A stage span that checks names on the way in.

    Most stage attributes are only known at the end: how many partials arrived, how
    many chunks were synthesised. Handing out the raw span for those would leave the
    contract enforced on the attributes set at entry and unenforced on exactly the
    ones a hurried change adds.
    """

    def __init__(self, span: Span, stage: str) -> None:
        self.span = span
        self._stage = stage

    def record(self, **attributes: object) -> None:
        unknown = set(attributes) - CONTRACT[self._stage] - SHARED_ATTRIBUTES
        if unknown:
            raise UndeclaredAttribute(f"{self._stage} may not carry {sorted(unknown)}")
        for key, value in attributes.items():
            self.span.set_attribute(key, value)

    def mark(self, event: str) -> None:
        """A point in time on this span, for a moment that is not a duration."""
        self.span.add_event(event)


@contextmanager
def stage_span(
    name: str, *, start_time: int | None = None, **attributes: object
) -> Iterator[StageSpan]:
    """One pipeline stage, with its attributes checked against the contract.

    Raises rather than dropping an unknown attribute. A silently ignored one gives
    a green test, a working service, and a dashboard panel that is empty for a
    reason nobody can find, which is the failure this whole project is about.

    `start_time` backdates the span, in nanoseconds from `time.time_ns`. Some
    stages begin many iterations before anything knows they did: `vad.endpoint`
    starts on the first frame of speech and only ends once the endpointer fires,
    and holding a context manager open across a socket loop to express that would
    put the span's lifetime in the hands of every `continue` in the loop. Recording
    when it began and backdating is the same measurement with fewer ways to leak a
    span.
    """
    if name not in CONTRACT:
        raise UndeclaredAttribute(f"{name} is not a declared stage span")

    allowed = CONTRACT[name]
    unknown = set(attributes) - allowed - SHARED_ATTRIBUTES
    if unknown:
        raise UndeclaredAttribute(f"{name} may not carry {sorted(unknown)}")

    # A plain span, not `spanlight.tool_span`. That helper names the span
    # `tool <name>` and stamps the tool attributes on it, which would make five
    # pipeline stages per turn look like repeated tool calls: the loop detector
    # fires on identical calls and the silent-failure detector reasons about tool
    # spans, so instrumenting stages as tools would manufacture detections out of
    # a healthy pipeline.
    #
    # `record_exception` and `set_status_on_exception` are both off. Their defaults
    # attach `exception.message` and a full stack trace to the span, and the
    # exceptions in this pipeline carry provider error bodies, which can quote a
    # transcript back. That is the leak the previous project found with a redaction
    # canary after its error contract already looked correct: the message was not in
    # the attributes, it was in the events, and nothing was looking there.
    tracer = spanlight.get_tracer()
    with tracer.start_as_current_span(
        name,
        record_exception=False,
        set_status_on_exception=False,
        start_time=start_time,
    ) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        try:
            yield StageSpan(span, name)
        except Exception as exc:
            # The class name, never the message, for the reason above.
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
