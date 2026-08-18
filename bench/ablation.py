"""M5.2. Each technique measured independently against the M0 baseline.

    uv run python -m bench.ablation --n 5

Three arms are real, independently switchable code paths; the other four names in
BACKLOG's original seven are not, and `ablation/hypothesis.md` states exactly why
before this script runs anything, per M5.1.

**Interleaved, not batched.** For each corpus utterance, every arm runs before the
next utterance starts, rather than one arm's full pass finishing before the next
arm begins. A provider having a slow ten minutes lands on every arm equally this
way; batched, it would land on whichever arm happened to run during it and read as
a technique effect that was actually a load effect.

**n is small here on purpose, and it says so.** `bench/waterfall.py`'s own docstring
makes the same distinction: a default run for iterating on the harness, and a larger
one for the number that actually gets published. Eight configurations (unstreamed,
streamed, semantic on, semantic off, four aggressiveness levels) times a real n=20
is 160 real turns, each a live Groq and EdgeTts call, well past what a single
unattended session should spend against a shared per-minute token budget. This
script's own summary prints the real n so nobody mistakes a default-sized run for
the Proof Artifact number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.waterfall import load_corpus, one_turn, pcm_frames  # noqa: E402
from vaani.endpoint import Endpointer  # noqa: E402
from vaani.stt import GroqWhisper  # noqa: E402
from vaani.tts import EdgeTts, FailingOverTts  # noqa: E402
from vaani.turn import Turn  # noqa: E402

# "unstreamed" is not here: it takes no endpointer at all, per `run_unstreamed`,
# since there is no socket and no turn-taking decision to make on a single
# direct call.
ARMS = {
    "streamed": lambda: Endpointer(semantic=True),
    "semantic_off": lambda: Endpointer(semantic=False),
    "aggressiveness_0": lambda: Endpointer.at(0),
    "aggressiveness_3": lambda: Endpointer.at(3),
}


@dataclass
class ArmRun:
    arm: str
    utterance_id: str
    first_answer_heard_ms: float | None
    reply_chars: int
    error: str | None = None


async def run_unstreamed(entry: dict) -> ArmRun:
    """The M0 baseline, direct: no socket, no session, one call in and one
    result out. Nothing is delivered until everything is, so wall-clock time
    for the `Turn.run` call is time to first audio here, and also time to
    last audio: that collapse is the baseline's own definition, not a
    simplification of this script's.

    What it is not exempt from is `bench/stages.md`'s own clock: measured
    from the last frame of user speech, not from whenever this function
    happens to be called. The first version of this called `Turn.run`
    immediately on the corpus's whole PCM buffer, which skipped the
    endpoint-detection wait entirely, since there was no real-time frame
    delivery for anything to wait on. That made the unstreamed arm look
    faster than the streamed ones by however long the trailing-silence
    timeout is, which is exactly the wrong direction for a baseline that is
    supposed to be the slow one: a span measures its own extent, and
    starting the unstreamed clock after a wait the streamed clock pays is
    the same mistake Spanlight's own field study made, the other way round.
    Frames are paced and fed through a real `Endpointer` here so both arms
    start their clock at the same event.
    """
    from vaani.protocol import FRAME_MS

    stt = GroqWhisper()
    tts = FailingOverTts(EdgeTts(), EdgeTts())
    turn = Turn(stt=stt, tts=tts)

    frames = pcm_frames(entry)
    endpointer = Endpointer(semantic=False)  # matches the free stack's own baseline arm
    due = time.monotonic()
    interval = FRAME_MS / 1000
    pcm = bytearray()
    fired_ns: int | None = None
    for frame in frames:
        remaining = due - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        pcm += frame
        if endpointer.accept(frame):
            fired_ns = time.monotonic_ns()
            break
        due += interval

    # The endpoint fires only once the trailing silence has been waited out, so
    # the last real frame of speech is that many milliseconds before this. Same
    # backdating `vaani/session.py` uses for the streamed arms' own clock, so
    # both sides of this comparison start counting from the same event.
    last_speech_ns = (fired_ns or time.monotonic_ns()) - endpointer.silence_ms * 1_000_000
    result = await turn.run(bytes(pcm))
    elapsed_ms = (time.monotonic_ns() - last_speech_ns) / 1_000_000

    return ArmRun(
        arm="unstreamed",
        utterance_id=entry["id"],
        first_answer_heard_ms=elapsed_ms,
        reply_chars=len(result.reply),
    )


async def run_streamed_arm(arm: str, entry: dict, exporter: InMemorySpanExporter) -> ArmRun:
    run = await one_turn(entry, exporter, stack="free", endpointer_factory=ARMS[arm])
    return ArmRun(
        arm=arm,
        utterance_id=entry["id"],
        first_answer_heard_ms=run.first_answer_heard_ms,
        reply_chars=run.reply_chars,
    )


async def measure(n: int, arms: list[str]) -> list[ArmRun]:
    corpus = load_corpus()[:n]
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    import vaani.spans as spans_module

    spans_module.spanlight.get_tracer = lambda: provider.get_tracer("ablation")

    results: list[ArmRun] = []
    total = len(corpus) * len(arms)
    done = 0
    for entry in corpus:
        for arm in arms:
            # A single transient failure (a live Groq rate limit, a DNS blip on
            # this machine) must not cost the whole run. n=20 across three arms
            # is 60 real network calls; losing all of them to one bad call is a
            # worse failure than the one it was reacting to. The turn is recorded
            # with its error and excluded from the summary statistics rather than
            # silently dropped, so a run's raw JSON always shows what happened.
            try:
                if arm == "unstreamed":
                    results.append(await run_unstreamed(entry))
                else:
                    results.append(await run_streamed_arm(arm, entry, exporter))
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {arm} / {entry['id']} FAILED: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                results.append(
                    ArmRun(
                        arm=arm,
                        utterance_id=entry["id"],
                        first_answer_heard_ms=None,
                        reply_chars=0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            done += 1
            print(f"  {done}/{total}: {arm} / {entry['id']}", file=sys.stderr)
    return results


def summarise(results: list[ArmRun], baseline_arm: str) -> dict:
    by_arm: dict[str, list[float]] = {}
    failures: dict[str, int] = {}
    for r in results:
        if r.first_answer_heard_ms is not None:
            by_arm.setdefault(r.arm, []).append(r.first_answer_heard_ms)
        else:
            failures[r.arm] = failures.get(r.arm, 0) + 1

    baseline_median = statistics.median(by_arm[baseline_arm]) if by_arm.get(baseline_arm) else None

    summary = {
        "n_per_arm": {arm: len(vals) for arm, vals in by_arm.items()},
        "failures_per_arm": failures,
        "arms": {},
    }
    for arm, values in by_arm.items():
        median = statistics.median(values)
        summary["arms"][arm] = {
            "n": len(values),
            "median_ms": round(median, 1),
            "min_ms": round(min(values), 1),
            "max_ms": round(max(values), 1),
            "delta_vs_baseline_ms": (
                round(baseline_median - median, 1) if baseline_median is not None else None
            ),
        }
    return summary


def render(summary: dict, baseline_arm: str) -> str:
    lines = [
        f"Ablation, n={summary['n_per_arm'].get(baseline_arm, 0)} per arm "
        f"(default size; see this script's own docstring for what n=20 would cost).",
        "",
        f"{'arm':<20}{'n':>4}{'median_ms':>12}{'min_ms':>10}{'max_ms':>10}"
        f"{'bought_vs_' + baseline_arm:>22}",
    ]
    for arm, row in summary["arms"].items():
        delta = row["delta_vs_baseline_ms"]
        delta_str = "" if delta is None or arm == baseline_arm else f"{delta:+.1f}ms"
        lines.append(
            f"{arm:<20}{row['n']:>4}{row['median_ms']:>12}{row['min_ms']:>10}"
            f"{row['max_ms']:>10}{delta_str:>22}"
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=5, help="corpus utterances per arm")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["unstreamed", "streamed", "semantic_off"],
        choices=["unstreamed", "streamed", "semantic_off", "aggressiveness_0", "aggressiveness_3"],
    )
    parser.add_argument("--json", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()

    baseline_arm = "unstreamed" if "unstreamed" in args.arms else args.arms[0]
    results = await measure(args.n, args.arms)
    summary = summarise(results, baseline_arm)

    args.json.write_text(
        json.dumps(
            {"summary": summary, "raw": [r.__dict__ for r in results]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(render(summary, baseline_arm))
    print(f"\nRaw per-run data: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
