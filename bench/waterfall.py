"""M4.3. Per-stage p50 and p95 from real Spanlight spans, for the free stack.

    uv run python -m bench.waterfall --runs 3

Drives the fixed corpus (M4.2, `bench/corpus/`) through a real `VoiceSession` against
live Groq and EdgeTts, exactly the path a browser drives, and reads the numbers off the
spans `vaani/session.py` and the pipeline actually emit rather than timing anything
separately. `bench/stages.md` is what those spans are defined against; this script does
not repeat that definition, it reports what the definitions produce on real traffic.

**A modelled client, for the same reason `bench/bargein.py`'s is.** `playback.first_audio`
needs a browser's "I started playing" acknowledgement this project's protocol does not
send yet (`bench/stages.md`, and M2.15's and M1.6's own notes on the same gap), so that
span is never emitted and never appears in this report. Every other span comes from the
real, running pipeline: nothing here recomputes a duration Spanlight already measured.

**Frames are paced at 20ms, not fed in a tight loop.** A harness that delivers a
five-second utterance in five milliseconds is not measuring STT, LLM, or TTS latency, it
is measuring how fast this script can push bytes, and every number under it would be
this script's own speed rather than the pipeline's.

**n is small on purpose and the report says so.** Twenty corpus utterances, each several
seconds of paced audio plus however many live STT partials, an LLM round trip, and TTS
synthesis actually take, is minutes of real wall time per full pass. `--runs` controls
how many passes over the corpus this call makes; the published Proof Artifact number
needs more than the default here, and this script prints exactly how many it saw so a
reader is never left guessing at n from a summary table alone.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm import ChatClient
from vaani.endpoint import Endpointer
from vaani.fillers import FillerBank, Purpose, read_chunks
from vaani.history import Conversation
from vaani.llm_turn import StreamedTurn
from vaani.pipeline import StreamingPipeline
from vaani.protocol import FRAME_BYTES, FRAME_MS, ClientMessage, Frame, ServerMessage
from vaani.session import Incoming, VoiceSession
from vaani.stt import ChunkedStt, GroqWhisper, RecoveringStt, StreamingStt
from vaani.tts import EdgeTts, FailingOverTts, TtsProvider

# Two of the twenty corpus utterances are in Devanagari, and structlog's console
# renderer writes it straight to stdout. Windows' default stream encoding when
# stdout is not a real console (piped or redirected, which every run of this
# script that keeps a log is) is the ANSI code page, cp1252 here, which cannot
# represent Devanagari at all. The write then raises `UnicodeEncodeError`, from
# inside a session's own error-recovery log line, which fails the recovery it
# was reporting on: the turn's real exception is a rate limit or a dropped
# connection, and what actually reaches this script is an encoding error with no
# message, because `SAFE_TO_LOG` already stripped it before the print that
# cannot render it. Reconfiguring here is what makes a run of this script mean
# the same thing on Windows as everywhere else it runs.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CORPUS = Path(__file__).resolve().parent / "corpus"
MANIFEST = CORPUS / "manifest.json"

# Every span this report can actually read. `playback.first_audio` and `turn` on the
# verified-interruption path are the two gaps `bench/stages.md`'s own notes already name;
# neither is reachable from a clean, uninterrupted corpus run in the first place.
STAGES = ("vad.endpoint", "stt.stream", "llm.generate", "tts.synthesize", "turn")


class QueueTransport:
    """One turn's worth of a browser, enough for a clean run with no barge-in.

    `bench/bargein.py`'s `Browser` models preemption too, which this does not need:
    the corpus is read start to finish with nothing talking over it, so this only has
    to feed paced frames and know when the turn's audio has finished arriving.
    """

    def __init__(self) -> None:
        self._incoming: asyncio.Queue[Incoming] = asyncio.Queue()
        self.sent_json: list[dict] = []
        self.sent_audio_bytes = 0
        self.done = asyncio.Event()

    def feed(self, items: list[Incoming]) -> None:
        for item in items:
            self._incoming.put_nowait(item)

    async def receive(self) -> Incoming:
        return await self._incoming.get()

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)
        if payload.get("type") == ServerMessage.AUDIO_END:
            self.done.set()

    async def send_bytes(self, data: bytes) -> None:
        self.sent_audio_bytes += len(data)

    def kinds(self) -> list[str]:
        return [str(m.get("type")) for m in self.sent_json]


def build_answer(
    stt: StreamingStt,
    tts: TtsProvider,
    history: Conversation,
    endpointer: Endpointer | None = None,
):
    """The same construction `app/routers/voice.py` uses for a real session,
    generalised over which stack's `stt` this call was handed: the free stack's
    own recovery wrapping (`RecoveringStt`, falling back to `GroqWhisper`'s batch
    endpoint) happens at the call site in `build_stack`, not here, so this
    function does not have to know which stack it is building for.

    `endpointer` exists for M5: `StreamingPipeline` builds its own
    `Endpointer(semantic=True)` by default, a second, separate instance from the
    one `one_turn` gives `VoiceSession`, and ablating semantic endpointing or VAD
    aggressiveness has to change both or it is not really ablating anything, since
    the session's own endpointer decides when a turn ends at all and the
    pipeline's decides how much of the final transcription wait semantic
    completeness can skip.
    """
    pipeline = StreamingPipeline(
        stt=stt,
        turn=StreamedTurn(llm=ChatClient()),
        tts=tts,
        history=history,
        endpointer=endpointer,
    )
    return pipeline.run


def build_stack(name: str) -> tuple[StreamingStt, TtsProvider]:
    """The two providers one turn needs, chosen by name.

    `sarvam` is credit-limited (QUOTAS.md), unlike `free`: nothing here calls it
    except a run that named it explicitly, and there is no batch fallback to wrap
    it in the way `GroqWhisper` wraps the free stack's own stream, because that
    fallback would silently spend Groq credits on a run meant to measure Sarvam
    alone.
    """
    if name == "free":
        transcriber = GroqWhisper()
        return (
            RecoveringStt(ChunkedStt(transcriber), transcriber),
            FailingOverTts(EdgeTts(), EdgeTts()),
        )
    if name == "sarvam":
        from vaani.sarvam import SarvamBulbul, SarvamSaaras

        return SarvamSaaras(), SarvamBulbul()
    raise ValueError(f"unknown stack {name!r}, expected 'free' or 'sarvam'")


# One bank for the whole run, matching `app/routers/voice.py`'s own scope: the clips
# do not change between turns, and reading them fresh per turn would put a disk seek
# on a path this measurement is not trying to characterise.
_fillers = FillerBank()

# Roughly a frame of MP3 at this bitrate, matching the real router's own chunk size.
_FILLER_CHUNK_BYTES = 720


async def speak_filler(purpose: Purpose = Purpose.THINKING):
    """The real bank, not a stub. A fake filler that returns instantly would make
    `turn`'s span end on a fabricated first chunk rather than whatever the real
    pipeline actually produced first, silently reporting the deadline mechanism as
    faster than it is instead of the pipeline underneath it."""
    clip = _fillers.pick(purpose)
    if clip is None:
        return
    for chunk in read_chunks(clip, _FILLER_CHUNK_BYTES):
        yield chunk


def load_corpus() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest:
        path = CORPUS / entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path}: run scripts/build_corpus.py first")
    return manifest


def pcm_frames(entry: dict) -> list[bytes]:
    """One corpus utterance's audio, as the 20ms frames the protocol wants.

    The WAV header is 44 bytes; the rest is already 16kHz mono PCM16, exactly what
    `scripts/build_corpus.py` wrote. A short remainder frame at the end is padded
    with silence rather than dropped, so the last word is not clipped out of the
    measurement the same way a live endpoint would clip it if it fired mid-word.
    """
    raw = (CORPUS / entry["file"]).read_bytes()[44:]
    frames = []
    for start in range(0, len(raw), FRAME_BYTES):
        chunk = raw[start : start + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
        frames.append(chunk)
    # Trailing silence, so the endpointer's trailing-silence timeout actually has
    # something to wait out and fire on, the same as a real caller going quiet.
    frames += [b"\x00" * FRAME_BYTES] * (Endpointer().trailing_silence_ms // FRAME_MS + 5)
    return frames


@dataclass
class Run:
    utterance_id: str
    category: str
    audio_duration_s: float
    stage_ms: dict[str, list[float]]
    heard_chars: int
    reply_chars: int
    # The three numbers `bench/stages.md` requires published together, read off the
    # session's own `TurnClock` rather than recomputed: first audio of any kind
    # (filler included), the answer's own first chunk, and what the listener would
    # actually have heard given whatever was queued in front of it.
    first_audio_ms: float | None
    first_answer_audio_ms: float | None
    first_answer_heard_ms: float | None
    filler_spoken: bool
    # Set when the turn never reached AUDIO_END within the wait below. Seen live:
    # a provider's own rate limit asked for a multi-minute cooldown and the
    # retry loop honoured it faithfully, which is correct for a turn a real
    # caller is waiting on but means one slow turn must not cost the other
    # nineteen. Whatever spans and text exist by the time this fires are kept
    # rather than discarded, since a slow answer is still data about the tail.
    timed_out: bool = False


async def one_turn(
    entry: dict,
    exporter: InMemorySpanExporter,
    stack: str = "free",
    endpointer_factory=lambda: Endpointer(semantic=True),
) -> Run:
    before = len(exporter.get_finished_spans())

    stt, tts = build_stack(stack)
    history = Conversation()
    transport = QueueTransport()
    # Two instances, not one shared: `StreamingPipeline` and `VoiceSession` each
    # own their own endpointer's mutable state (started, trailing silence so
    # far), and sharing one between them would make each turn's second reader
    # see state the first one already consumed.
    session = VoiceSession(
        transport=transport,
        answer=build_answer(stt, tts, history, endpointer=endpointer_factory()),
        filler=speak_filler,
        endpointer=endpointer_factory(),
        bytes_per_second=tts.bytes_per_second,
        history=history,
    )

    task = asyncio.create_task(session.run())
    transport.feed([Incoming(control=ClientMessage.START)])

    frames = pcm_frames(entry)
    due = time.monotonic()
    interval = FRAME_MS / 1000
    for pcm in frames:
        remaining = due - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        transport.feed([Incoming(frame=Frame(generation=1, pcm=pcm))])
        due += interval

    # The task is cancelled in `finally` rather than only after a clean wait,
    # because `wait_for` raising `TimeoutError` on the line below skips every
    # line after it: the session's own task would otherwise be left running,
    # uncancelled, for however long its own retry loop takes to give up on its
    # own, which is exactly the orphaned-task shape `SpeakingTurn.cancel`
    # exists to prevent one layer in. Live once: a provider rate limit asked
    # for a cooldown long enough that the orphaned task ran for 45 minutes
    # after this function had already raised past it.
    timed_out = False
    try:
        await asyncio.wait_for(transport.done.wait(), timeout=60.0)
    except TimeoutError:
        timed_out = True
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    spans = exporter.get_finished_spans()[before:]
    by_stage: dict[str, list[float]] = defaultdict(list)
    for span in spans:
        if span.name in STAGES:
            duration_ms = (span.end_time - span.start_time) / 1_000_000
            by_stage[span.name].append(duration_ms)

    reply = "".join(m.get("text", "") for m in transport.sent_json if m.get("type") == "reply")
    heard = "".join(m.get("text", "") for m in transport.sent_json if m.get("type") == "transcript")
    clock = session.clock
    return Run(
        utterance_id=entry["id"],
        category=entry["category"],
        audio_duration_s=entry["duration_s"],
        stage_ms=dict(by_stage),
        heard_chars=len(heard),
        reply_chars=len(reply),
        first_audio_ms=None if clock is None else clock.first_audio_ms,
        first_answer_audio_ms=None if clock is None else clock.first_answer_audio_ms,
        first_answer_heard_ms=None if clock is None else clock.first_answer_heard_ms,
        filler_spoken=False if clock is None else clock.filler_spoken,
        timed_out=timed_out,
    )


def percentile(values: list[float], pct: int) -> float:
    """Nearest rank. Honest at small n rather than inventing a value never measured,
    the same reasoning `bench/bargein.py` already applies to its own p95."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    import math

    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[min(rank, len(ordered)) - 1]


def summarise(runs: list[Run]) -> dict:
    per_stage: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for stage, durations in run.stage_ms.items():
            # One number per turn per stage: the stage's own total this turn, since a
            # turn can carry several `tts.synthesize` spans (one per sentence) and a
            # per-turn waterfall reads their sum, not one sentence picked arbitrarily.
            per_stage[stage].append(sum(durations))

    summary = {
        "n_turns": len(runs),
        "n_utterances": len({r.utterance_id for r in runs}),
        "heard_nothing": sum(1 for r in runs if r.heard_chars == 0),
        "no_reply": sum(1 for r in runs if r.reply_chars == 0),
        "timed_out": sum(1 for r in runs if r.timed_out),
        "filler_rate": (
            round(sum(1 for r in runs if r.filler_spoken) / len(runs), 2) if runs else None
        ),
        "stages": {},
        "headline": {},
    }
    for stage in STAGES:
        values = per_stage.get(stage, [])
        summary["stages"][stage] = {
            "n": len(values),
            "p50_ms": round(statistics.median(values), 1) if values else None,
            "p95_ms": round(percentile(values, 95), 1) if values else None,
            "min_ms": round(min(values), 1) if values else None,
            "max_ms": round(max(values), 1) if values else None,
            "stdev_ms": round(statistics.pstdev(values), 1) if len(values) > 1 else None,
        }

    # The three numbers `bench/stages.md` requires published side by side, never one
    # standing in for another: first audio of any kind flatters every turn filler
    # covered, the answer's own send time is the number that looks right and is not,
    # and heard is what the target is actually judged against.
    for key, label in (
        ("first_audio_ms", "first_audio_ms"),
        ("first_answer_audio_ms", "first_answer_audio_ms"),
        ("first_answer_heard_ms", "first_answer_heard_ms"),
    ):
        values = [getattr(r, key) for r in runs if getattr(r, key) is not None]
        summary["headline"][label] = {
            "n": len(values),
            "p50_ms": round(statistics.median(values), 1) if values else None,
            "p95_ms": round(percentile(values, 95), 1) if values else None,
        }
    return summary


def render(summary: dict, stack: str = "free") -> str:
    stack_label = (
        "free stack (Groq + EdgeTts)" if stack == "free" else "sarvam stack (Saaras + Bulbul)"
    )
    lines = [
        f"Waterfall over {summary['n_turns']} turns "
        f"({summary['n_utterances']} corpus utterances), {stack_label}.",
        "",
    ]
    if summary["heard_nothing"] or summary["no_reply"]:
        lines.append(
            f"{summary['heard_nothing']} turn(s) transcribed nothing, "
            f"{summary['no_reply']} produced no reply text. Included below, not "
            f"dropped: a stage that failed to happen is not a stage that measured 0ms."
        )
        lines.append("")
    if summary["timed_out"]:
        lines.append(
            f"{summary['timed_out']} turn(s) never reached AUDIO_END within 60s and "
            f"were cut off rather than left to run: a provider rate limit can ask for "
            f"a cooldown of several minutes, and the retry loop honours it, which the "
            f"waterfall cannot afford to wait out one turn at a time. Their partial "
            f"spans are still counted below."
        )
        lines.append("")

    lines.append(
        f"Filler rate: {summary['filler_rate']}. A turn covered by filler is a turn "
        f"the answer was late in, not a fast one."
    )
    lines.append("")
    lines.append(f"{'headline':<24}{'n':>4}{'p50':>9}{'p95':>9}")
    for key in ("first_audio_ms", "first_answer_audio_ms", "first_answer_heard_ms"):
        h = summary["headline"][key]
        if h["n"] == 0:
            lines.append(f"{key:<24}{'(none)':>17}")
            continue
        lines.append(f"{key:<24}{h['n']:>4}{h['p50_ms']:>9.1f}{h['p95_ms']:>9.1f}")
    lines.append(
        "The target is judged on first_answer_heard_ms alone (bench/stages.md): "
        "first_audio_ms flatters every turn filler covered, and "
        "first_answer_audio_ms is the send time, not when it could be heard."
    )
    lines.append("")

    lines.append(f"{'stage':<18}{'n':>4}{'p50':>9}{'p95':>9}{'min':>9}{'max':>9}{'stdev':>9}")
    for stage in STAGES:
        s = summary["stages"][stage]
        if s["n"] == 0:
            lines.append(f"{stage:<18}{'(no spans)':>40}")
            continue
        lines.append(
            f"{stage:<18}{s['n']:>4}{s['p50_ms']:>9.1f}{s['p95_ms']:>9.1f}"
            f"{s['min_ms']:>9.1f}{s['max_ms']:>9.1f}"
            f"{'' if s['stdev_ms'] is None else s['stdev_ms']:>9}"
        )
    lines.append("")
    lines.append(
        "Stage durations do not sum to a turn total: the stages overlap by design "
        "(bench/stages.md), and turn's own span is the number a listener would call "
        "the wait, not a sum of the rows above it."
    )
    lines.append(
        "playback.first_audio is absent: it closes on a browser acknowledgement this "
        "protocol does not send yet, the gap bench/stages.md and M1.6's own note "
        "already name."
    )
    return "\n".join(lines)


async def measure(runs_per_utterance: int, stack: str = "free") -> tuple[list[Run], dict]:
    corpus = load_corpus()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    import vaani.spans as spans_module

    spans_module.spanlight.get_tracer = lambda: provider.get_tracer("waterfall")

    results: list[Run] = []
    total = runs_per_utterance * len(corpus)
    done = 0
    for _ in range(runs_per_utterance):
        for entry in corpus:
            results.append(await one_turn(entry, exporter, stack=stack))
            done += 1
            print(f"  {done}/{total}: {entry['id']}", file=sys.stderr)

    raw = [
        {
            "utterance_id": r.utterance_id,
            "category": r.category,
            "audio_duration_s": r.audio_duration_s,
            "stage_ms": r.stage_ms,
            "heard_chars": r.heard_chars,
            "reply_chars": r.reply_chars,
            "timed_out": r.timed_out,
        }
        for r in results
    ]
    return results, {"raw": raw}


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=1, help="passes over the full corpus")
    parser.add_argument(
        "--stack",
        choices=("free", "sarvam"),
        default="free",
        help="free is Groq+EdgeTts, no cost; sarvam is Saaras+Bulbul, credit-limited "
        "per QUOTAS.md and only ever run when named explicitly",
    )
    parser.add_argument("--json", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()

    runs, extra = await measure(args.runs, stack=args.stack)
    summary = summarise(runs)

    args.json.write_text(
        json.dumps({"summary": summary, "stack": args.stack, **extra}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(render(summary, stack=args.stack))
    print(f"\nRaw per-run data: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
