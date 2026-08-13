"""The canary that enforces `docs/threat-model.md`.

An improbable phrase is planted where a real transcript would be, a whole turn is
run, and every span and every log line is swept for it. Sweeping rather than
checking an allowlist of fields known to be safe, because an allowlist can only
catch leaks somebody predicted. Spanlight's error contract was correct and had a
passing test proving an email address never reached `span.attributes`; the leak was
in `span.events`, and nothing was looking there.

The transcript is expected to reach the client, and that is not a leak: it is the
speaker's own words, shown so they can see what was heard. So the test asserts both
directions. Absent from logs and spans, present in what went to the browser. Without
the second half a canary that never entered the system at all would pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator

import pytest

from app.logging_config import configure_logging
from vaani.protocol import ClientMessage, Frame
from vaani.session import Incoming, VoiceSession
from vaani.spans import CONTRACT, TTS_SYNTHESIZE, stage_span
from vaani.tools import ToolError, check_eligibility

from .vaani.test_endpoint import SILENCE, SPEECH

# Improbable enough that a match cannot be a coincidence, and shaped like the thing
# it stands for: a question about somebody's own circumstances.
CANARY_HEARD = "mera-naam-kaanary-hai-aur-meri-aay-777777"
CANARY_REPLIED = "aap-eligible-hain-kaanary-jawaab-888888"


class Recorder:
    def __init__(self) -> None:
        self.json: list[dict] = []
        self.audio: list[bytes] = []
        self._messages: list[Incoming] = [
            Incoming(control=ClientMessage.START),
            *[Incoming(frame=Frame(generation=1, pcm=SPEECH)) for _ in range(30)],
            *[Incoming(frame=Frame(generation=1, pcm=SILENCE)) for _ in range(60)],
        ]

    async def receive(self) -> Incoming:
        if self._messages:
            return self._messages.pop(0)
        # Blocks rather than disconnecting, because the turn is answered on a task
        # and a disconnect tears it down in the session's own teardown. Returning
        # one here cancelled the turn before it produced anything, so the canary
        # never entered the system and the sweep had nothing to find.
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")  # pragma: no cover

    async def send_json(self, payload: dict) -> None:
        self.json.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)


async def answer(
    frames: AsyncIterator[bytes], still_current, on_transcript=None, on_sentence=None
) -> AsyncIterator[bytes]:
    async for _pcm in frames:
        pass
    if on_transcript is not None:
        await on_transcript(CANARY_HEARD)
    if on_sentence is not None:
        await on_sentence(CANARY_REPLIED)
    yield b"audio-for-the-canary"


async def filler() -> AsyncIterator[bytes]:
    yield b"achha"


@pytest.fixture
def spans(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import vaani.spans as spans_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        spans_module.spanlight, "get_tracer", lambda: provider.get_tracer("canary")
    )
    return exporter


def everything_about(span) -> list[str]:
    """Every field of a span a string could hide in.

    Attributes, events and their attributes, the status description, the name, and
    the resource. The events are the ones that matter most: that is where the leak
    was found last time.
    """
    fields = [str(span.name), str(span.status.description)]
    fields += [f"{key}={value}" for key, value in (span.attributes or {}).items()]
    for event in span.events:
        fields.append(str(event.name))
        fields += [f"{key}={value}" for key, value in (event.attributes or {}).items()]
    fields += [f"{key}={value}" for key, value in (span.resource.attributes or {}).items()]
    return fields


async def run_one_turn(voice: VoiceSession, transport: Recorder) -> None:
    """Drive the session until it has spoken, then stop it."""
    task = asyncio.create_task(voice.run())
    try:
        await asyncio.wait_for(_until(lambda: bool(transport.audio)), timeout=3.0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _until(predicate, interval: float = 0.005) -> None:
    while not predicate():
        await asyncio.sleep(interval)


async def test_no_transcript_or_reply_reaches_a_span_or_a_log(spans, capsys) -> None:
    configure_logging("INFO")
    transport = Recorder()
    voice = VoiceSession(transport=transport, answer=answer, filler=filler)

    await run_one_turn(voice, transport)

    logs = capsys.readouterr().out
    span_fields = [field for span in spans.get_finished_spans() for field in everything_about(span)]

    # The half that proves the test is exercising anything at all. A canary that
    # never entered the system would satisfy every assertion below.
    client_text = json.dumps(transport.json)
    assert CANARY_HEARD in client_text
    assert CANARY_REPLIED in client_text

    for canary in (CANARY_HEARD, CANARY_REPLIED):
        assert canary not in logs, "leaked into a log line"
        for field in span_fields:
            assert canary not in field, f"leaked into a span field: {field}"


async def test_a_failing_turn_leaks_nothing_either(spans, capsys) -> None:
    """The path that leaked last time. An exception carries a message, and the SDK
    attaches it to the span as an event unless told not to, so a failure is where a
    transcript escapes even when every success path is clean."""
    configure_logging("INFO")

    async def breaks(frames, still_current, on_transcript=None, on_sentence=None):
        async for _pcm in frames:
            pass
        if on_transcript is not None:
            await on_transcript(CANARY_HEARD)
        yield b"first"
        # Raised inside a stage span, because that is where a real failure happens:
        # synthesis and generation both run inside one. An earlier version raised
        # outside any span, so there was nothing for the message to be attached to
        # and turning `record_exception` back on did not trip this test at all.
        with stage_span(TTS_SYNTHESIZE):
            raise RuntimeError(f"synthesis died on {CANARY_REPLIED}")

    transport = Recorder()
    voice = VoiceSession(transport=transport, answer=breaks, filler=filler)

    await run_one_turn(voice, transport)

    logs = capsys.readouterr().out
    span_fields = [field for span in spans.get_finished_spans() for field in everything_about(span)]

    assert CANARY_HEARD in json.dumps(transport.json)
    assert CANARY_REPLIED not in logs
    for field in span_fields:
        assert CANARY_REPLIED not in field, f"leaked into a span field: {field}"


def test_a_rejected_tool_argument_never_repeats_the_value(capsys) -> None:
    """An applicant's income is the most sensitive number the pipeline handles, and a
    validation message is the easiest place to spill it."""
    configure_logging("INFO")

    with pytest.raises(ToolError) as raised:
        check_eligibility(
            {
                "scheme_id": "pm-kisan",
                "applicant": {"state": "Bihar", "annual_income_inr": -777777},
            }
        )

    assert "777777" not in str(raised.value)
    assert "777777" not in capsys.readouterr().out


def test_no_declared_span_attribute_could_hold_speech() -> None:
    """The contract, checked as a shape rather than a behaviour. A leak needs a field
    to live in, and this is the list of fields that exist."""
    for name, attributes in CONTRACT.items():
        for attribute in attributes:
            leaf = attribute.rsplit(".", 1)[-1]
            for banned in ("text", "transcript", "reply", "content", "prompt"):
                assert banned not in leaf, f"{name}.{attribute}"


def test_the_client_bundle_carries_no_secret() -> None:
    """Two static files on GitHub Pages. A key in there is public the moment it is
    pushed, and it would be pushed by the same workflow that deploys the page."""
    import pathlib
    import re

    suspicious = re.compile(
        r"(api[_-]?key|secret|token|Bearer\s+\S+|gsk_|sk-[A-Za-z0-9]{16,})", re.IGNORECASE
    )

    for path in pathlib.Path("web").rglob("*"):
        if not path.is_file():
            continue
        found = suspicious.findall(path.read_text(encoding="utf-8"))
        assert not found, f"{path} looks like it carries a credential: {found}"
