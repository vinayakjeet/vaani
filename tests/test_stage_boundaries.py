"""The stage definitions are pinned, so they cannot drift toward a flattering result.

M4.1. `bench/stages.md` says what every stage span starts and ends at, and what time
to first audio and barge-in latency are measured from. It is written before anything is
measured, and this test pins its digest so that changing a definition is a deliberate
act with its own commit rather than an edit nobody notices.

The failure this prevents is specific and this portfolio has already paid for it. A
previous project published a session latency of 1184ms against the provider's own 559ms,
because the span wrapped a concurrency semaphore and half of it was queueing reported as
work. Nobody noticed until one trace was opened by hand, well after the aggregates had
looked reasonable. A definition that can be edited during analysis is not a definition.

To change a boundary: edit `bench/stages.md`, run this test, and paste the digest it
reports into `bench/stages.sha256` in the same commit as the change and the reason.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFINITIONS = ROOT / "bench" / "stages.md"
PINNED = ROOT / "bench" / "stages.sha256"


def test_the_stage_definitions_match_their_pinned_digest() -> None:
    # Newlines normalised, because a checkout on Windows and one on the CI runner
    # differ by line ending alone and a digest that fails on the platform is a digest
    # people learn to ignore.
    body = DEFINITIONS.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(body).hexdigest()

    assert digest == PINNED.read_text().split()[0], (
        f"bench/stages.md changed. If that was deliberate, put {digest} in "
        f"bench/stages.sha256 in the same commit, with the reason the boundary moved."
    )


def test_the_definitions_cover_every_span_in_the_contract() -> None:
    """A stage instrumented but never defined measures whatever the code wrapped."""
    from vaani.spans import CONTRACT

    text = DEFINITIONS.read_text(encoding="utf-8")
    undefined = [name for name in CONTRACT if f"`{name}`" not in text]

    assert not undefined, f"instrumented but not defined in bench/stages.md: {undefined}"


def test_the_clock_is_stated_as_the_harder_one() -> None:
    """The single sentence the whole Proof Artifact rests on. Most published voice
    latency starts after endpointing, which silently removes 250 to 700ms from the
    figure a listener experiences, so two systems quoting 800ms can differ by a factor
    of two in the only thing anybody perceives."""
    text = DEFINITIONS.read_text(encoding="utf-8")

    assert "last frame of user speech" in text
    assert "Not at endpoint-detected" in text
