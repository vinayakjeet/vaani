"""Pre-synthesised acknowledgements, chosen so the same one is never heard twice running.

The filler exists because a fixed phrase never needs a network: it was measured at
1796ms when it was synthesised on demand, against the 600ms deadline it was supposed to
protect, so it spent the budget it existed to cover. Reading it off disk is 0.18ms.

That fix left a different problem, and the deadline being shorter than the endpoint wait
makes it worse: the filler fires on most turns, and it has been one file. The same four
words before every answer is how an IVR sounds. It is also worse than silence, because a
listener stops hearing an acknowledgement and starts hearing a recording.

So a bank, chosen without immediate repetition rather than at random: pure random
repeats about one time in eight with eight clips, and a repeat is exactly the thing this
is meant to avoid. The clips are committed, because a bank that synthesises itself on
startup has reintroduced the network this mechanism exists to remove, on the cold start
where it hurts most.

Purposes are separate lists rather than one pool because they are different speech acts.
"Ek minute" says the answer is coming. "Ji, boliye" says the floor is yours. Playing the
second where the first belongs tells somebody to talk while you are trying to answer
them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"


class Purpose(StrEnum):
    """What the clip is for. Not interchangeable, see the module docstring."""

    THINKING = "thinking"
    RESUMING = "resuming"


# Short on purpose. A filler is spoken while somebody waits, so a long one is a second
# thing to wait for, and the clip that shipped first was 3.5 seconds of it.
PHRASES: dict[Purpose, tuple[str, ...]] = {
    Purpose.THINKING: (
        "Ek minute.",
        "Ji, dekh raha hoon.",
        "Thoda rukiye.",
        "Abhi batata hoon.",
        "Ek second.",
        "Dekhta hoon.",
        "Haan, check kar raha hoon.",
        "Bas, aata hoon.",
    ),
    Purpose.RESUMING: (
        "Ji, boliye.",
        "Haan, kahiye.",
        "Boliye.",
        "Ji?",
    ),
}


def filename(purpose: Purpose, index: int) -> str:
    return f"filler-{purpose}-{index:02d}.mp3"


class FillerBank:
    """The clips on disk, with the last one played remembered per purpose."""

    def __init__(self, assets: Path = ASSETS, rng: random.Random | None = None) -> None:
        self._assets = assets
        self._rng = rng or random.Random()
        self._last: dict[Purpose, Path] = {}

    def available(self, purpose: Purpose) -> list[Path]:
        found = [
            path
            for index in range(len(PHRASES[purpose]))
            if (path := self._assets / filename(purpose, index)).exists()
        ]
        if not found:
            # The clip that predates the bank. Kept as a fallback rather than deleted, so
            # a checkout that has not run the build script still speaks.
            legacy = self._assets / "filler-hi.mp3"
            if purpose is Purpose.THINKING and legacy.exists():
                return [legacy]
        return found

    def pick(self, purpose: Purpose = Purpose.THINKING) -> Path | None:
        """A clip for this purpose, never the one played last, or None if there are none.

        None rather than raising. A missing bank must not take a turn down: the caller
        falls back to synthesising, which is slow and audible in the logs, where an
        exception here would be silence with no reply at all.
        """
        clips = self.available(purpose)
        if not clips:
            logger.warning("filler.bank_empty", purpose=str(purpose))
            return None

        choices = [clip for clip in clips if clip != self._last.get(purpose)] or clips
        chosen = self._rng.choice(choices)
        self._last[purpose] = chosen
        return chosen


def read_chunks(path: Path, chunk_bytes: int) -> Sequence[bytes]:
    """One clip as the sequence of chunks the transport sends.

    Chunked rather than sent whole so playback starts on the first slice, which is the
    same shape the synthesiser's own output has. A client that handles one and not the
    other would work in tests and not in the demo.
    """
    audio = path.read_bytes()
    return [audio[start : start + chunk_bytes] for start in range(0, len(audio), chunk_bytes)]
