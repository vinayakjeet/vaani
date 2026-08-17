"""M4.9. Whether tail multiplication is arithmetic or a story, checked against real traces.

    uv run python -m bench.tail_multiplication bench/waterfall.json

Five serial stages, each slow (at or past its own p95) five percent of the time, and
independent, put a turn past at least one stage's own p95 on `1 - 0.95**5`, about 22.6%
of turns, not 5%. That arithmetic is what a p95 floor is defending against and a p50
target cannot: even a pipeline whose every stage is individually well-behaved has a
meaningfully-sized tail because there are several places for a turn to be unlucky in,
not one. The number is textbook; whether this pipeline's own stages behave like the
independent draws the arithmetic assumes is not, and this script reads that off
`bench/waterfall.py`'s own raw per-run output rather than assuming it.

The check is honest about what n=20 can and cannot say. A stage's own p95 at n=20 sits at
rank 19 of 20 (nearest-rank), one below the worst observed run, so both the run at that
rank and the true worst one satisfy "at or past p95": every stage contributes its own top
two runs to the count, not an unusual few percent of them. The empirical multiplication
below is measured against thresholds this same data defined, not against an external,
previously-published p95, and a larger n is what would let this check mean "unusually
slow" rather than "close to the observed worst." That gap is stated rather than smoothed
over.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STAGES = ("vad.endpoint", "stt.stream", "llm.generate", "tts.synthesize", "turn")


def percentile(values: list[float], pct: int) -> float:
    import math

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[min(rank, len(ordered)) - 1]


def analyse(raw: list[dict]) -> dict:
    per_stage_p95: dict[str, float] = {}
    per_turn_totals: dict[str, list[float]] = {stage: [] for stage in STAGES}

    for run in raw:
        for stage in STAGES:
            durations = run.get("stage_ms", {}).get(stage, [])
            per_turn_totals[stage].append(sum(durations) if durations else 0.0)

    for stage in STAGES:
        values = [v for v in per_turn_totals[stage] if v > 0]
        per_stage_p95[stage] = percentile(values, 95) if values else float("nan")

    hit_any = 0
    hit_counts: dict[str, int] = {stage: 0 for stage in STAGES}
    n = len(raw)
    for i in range(n):
        hit_this_turn = False
        for stage in STAGES:
            value = per_turn_totals[stage][i]
            if value > 0 and value >= per_stage_p95[stage]:
                hit_counts[stage] += 1
                hit_this_turn = True
        if hit_this_turn:
            hit_any += 1

    measured_stages = [s for s in STAGES if per_turn_totals[s]]
    predicted = 1 - (0.95 ** len(measured_stages))

    return {
        "n_turns": n,
        "n_stages": len(measured_stages),
        "per_stage_p95_ms": per_stage_p95,
        "hit_counts": hit_counts,
        "turns_past_at_least_one_p95": hit_any,
        "empirical_fraction": round(hit_any / n, 3) if n else None,
        "predicted_fraction_if_independent": round(predicted, 3),
    }


def render(result: dict) -> str:
    lines = [
        f"Tail multiplication over {result['n_turns']} turns, {result['n_stages']} stages.",
        "",
        f"{'stage':<20}{'p95 (ms)':>12}{'turns >= p95':>16}",
    ]
    for stage in STAGES:
        p95 = result["per_stage_p95_ms"].get(stage)
        if p95 is None or p95 != p95:  # nan check
            continue
        lines.append(f"{stage:<20}{p95:>12.1f}{result['hit_counts'][stage]:>16}")
    lines.append("")
    lines.append(
        f"{result['turns_past_at_least_one_p95']} of {result['n_turns']} turns "
        f"({result['empirical_fraction']:.1%}) were at or past at least one stage's own "
        f"p95, against {result['predicted_fraction_if_independent']:.1%} predicted by "
        f"1 - 0.95^{result['n_stages']} if the stages were independent."
    )
    lines.append("")
    lines.append(
        "At n=20 each stage's p95 sits at rank 19 of 20, one below its own worst "
        "observed run, so every stage contributes its own top two runs to the count "
        "above rather than an unusual few percent of them. The turns-past-at-least-"
        "one-p95 total being lower than the sum of the per-stage hit counts is exactly "
        "the overlap where two stages were slow on the same turn, and that overlap, "
        "not the raw fraction, is the honest read of whether these stages behave "
        "independently."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("waterfall_json", type=Path)
    args = parser.parse_args()

    data = json.loads(args.waterfall_json.read_text(encoding="utf-8"))
    raw = data.get("raw", data if isinstance(data, list) else [])
    if not raw:
        print("No raw per-run data found in the given file.", file=sys.stderr)
        return 1

    result = analyse(raw)
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
