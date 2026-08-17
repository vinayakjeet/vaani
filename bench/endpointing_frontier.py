"""M4.13. The endpointing knob as a frontier, not a single chosen setting.

    uv run python -m bench.endpointing_frontier

X2-Turn's own Table 2 reports one knob against two axes at once, latency and accuracy,
rather than a single chosen tau with an argument for why it is correct. `AGGRESSIVENESS`
in `vaani/endpoint.py` is the same object: four named settings, each trading trailing-
silence latency against how readily it cuts someone off mid-sentence, and BACKLOG's own
plan before this script was to report one setting rather than the shape.

**What this cannot honestly claim.** A true false-endpoint rate needs real speech with
real mid-sentence pauses and a human judgement of which ones were genuinely finished.
`tests/vaani/test_endpoint.py`'s own fixtures are a synthesised tone standing in for
speech, not a recording, the same limitation `bench/corpus`'s own manifest states for a
different reason. What this script measures instead, honestly labelled as such: for a
battery of pause durations a person might plausibly produce mid-sentence, hesitating,
drawing breath, searching for a word, which of the four settings would end the turn
before the pause is over. That is accuracy's proxy here, not accuracy itself, and the
gap is the same one M4.2's own corpus note already names for this pipeline's numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaani.endpoint import AGGRESSIVENESS, Endpointer  # noqa: E402
from vaani.protocol import FRAME_MS  # noqa: E402

# Milliseconds a real pause plausibly lasts mid-utterance: a breath, a stumble, a beat
# while reaching for a word. Below the shortest named setting's own trailing silence and
# up through the longest, so every setting's own crossover point is inside the range
# rather than assumed from its endpoints.
PAUSE_DURATIONS_MS = tuple(range(100, 1300, 100))

SPEECH = b"\x00\x40" * 160  # one frame, loud enough to clear the energy threshold
SILENCE = b"\x00" * 320  # one frame, digital zero


def feed(endpointer: Endpointer, frame: bytes, ms: int) -> bool:
    ended = False
    for _ in range(ms // FRAME_MS):
        ended = endpointer.accept(frame) or ended
    return ended


def cuts_off_pause(aggressiveness: int, pause_ms: int) -> bool:
    """Whether this setting would end the turn during a pause of this length,
    with speech resuming right after it if the turn survived."""
    endpointer = Endpointer.at(aggressiveness)
    feed(endpointer, SPEECH, 400)
    ended = feed(endpointer, SILENCE, pause_ms)
    if not ended:
        feed(endpointer, SPEECH, 200)
    return ended


def measure() -> dict:
    rows = []
    for level in sorted(AGGRESSIVENESS):
        _threshold, trailing_ms = AGGRESSIVENESS[level]
        cut_at = [pause for pause in PAUSE_DURATIONS_MS if cuts_off_pause(level, pause)]
        shortest_pause_cut_off = min(cut_at) if cut_at else None
        rows.append(
            {
                "aggressiveness": level,
                "trailing_silence_ms": trailing_ms,
                "shortest_pause_cut_off_ms": shortest_pause_cut_off,
                "pauses_cut_off_of_n": (len(cut_at), len(PAUSE_DURATIONS_MS)),
            }
        )
    return {"pause_durations_ms": list(PAUSE_DURATIONS_MS), "rows": rows}


def render(result: dict) -> str:
    lines = [
        "Endpointing frontier: latency bought against pauses cut off, four named settings.",
        "",
        f"{'aggressiveness':<16}{'trailing_ms':>13}{'shortest cut (ms)':>19}{'cut / tested':>14}",
    ]
    for row in result["rows"]:
        shortest = row["shortest_pause_cut_off_ms"]
        shortest_str = "none" if shortest is None else str(shortest)
        cut, total = row["pauses_cut_off_of_n"]
        lines.append(
            f"{row['aggressiveness']:<16}{row['trailing_silence_ms']:>13}"
            f"{shortest_str:>19}{f'{cut}/{total}':>14}"
        )
    lines.append("")
    lines.append(
        "Pauses tested: " + ", ".join(f"{p}ms" for p in result["pause_durations_ms"]) + "."
    )
    lines.append(
        "Read as a frontier, not a ranking: the fastest setting cuts off the most of "
        "these plausible pauses by construction, since trailing_silence_ms is what a "
        "pause has to outlast either way. Which point on this line is correct is a "
        "product decision the number does not make on its own."
    )
    return "\n".join(lines)


def main() -> int:
    result = measure()
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
