"""The filler has to be faster than the deadline it protects.

Measured in a browser against the live backend, an on-demand filler arrived 1796ms
after speech ended, against a 600ms deadline and an 800ms floor. Saying it meant a
network round trip to a synthesiser, on the one path that is already late, so the
mechanism spent the budget it existed to cover.

The phrase never changes, so it is the one piece of audio in this system that can be
a file.
"""

from __future__ import annotations

import time

import pytest

from app.routers.voice import FILLER_CHUNK_BYTES, speak_filler


async def collect() -> list[bytes]:
    return [chunk async for chunk in speak_filler()]


def test_the_bank_is_committed() -> None:
    """A fork without it falls back to synthesising, which works and is slow. The clips
    existing is what makes the fast path the normal one.

    A bank rather than a clip, because the deadline is shorter than the endpoint wait so
    this fires on most turns, and the same four words before every answer is how an IVR
    sounds. The legacy single clip still counts, so a checkout that has not run
    `scripts/build_fillers.py` is not reported as broken."""
    from vaani.fillers import FillerBank, Purpose

    clips = FillerBank().available(Purpose.THINKING)

    assert clips
    assert all(clip.stat().st_size > 1000 for clip in clips)


async def test_it_speaks_from_disk_with_no_provider() -> None:
    """One of the committed clips, byte for byte, rather than anything synthesised."""
    from vaani.fillers import FillerBank, Purpose

    committed = {clip.read_bytes() for clip in FillerBank().available(Purpose.THINKING)}
    chunks = await collect()

    assert chunks
    assert b"".join(chunks) in committed


async def test_it_is_chunked_rather_than_sent_whole() -> None:
    """Playback starts on the first slice, which is the same shape the synthesiser's
    own output has, so the client needs no special case for filler."""
    chunks = await collect()

    assert len(chunks) > 1
    assert all(len(chunk) <= FILLER_CHUNK_BYTES for chunk in chunks)


async def test_the_first_chunk_is_effectively_free() -> None:
    """The whole point. A budget of 50ms is generous against a 600ms deadline and
    still two orders of magnitude below the round trip it replaces."""
    started = time.monotonic()

    async for _chunk in speak_filler():
        break

    assert (time.monotonic() - started) * 1000 < 50


async def test_a_missing_asset_still_speaks(monkeypatch, tmp_path) -> None:
    """A fork that has not generated one must not go silent. Silence is the outcome
    every degradation rule in SPEC exists to prevent, and it is worse than a slow
    acknowledgement."""
    import app.routers.voice as router
    from vaani.fillers import FillerBank, Purpose

    synthesised: list[str] = []

    class Stub:
        async def synthesize(self, text, voice, index=0):
            synthesised.append(text)
            yield b"synthesised-filler"

    # An empty directory rather than a stubbed reader, so this exercises the same
    # emptiness check a checkout that never ran the build script would hit.
    monkeypatch.setattr(router, "_fillers", FillerBank(assets=tmp_path))
    monkeypatch.setattr(router, "EdgeTts", lambda: Stub())

    assert await collect() == [b"synthesised-filler"]
    assert synthesised == [router.FILLER_TEXT[Purpose.THINKING]]


async def test_the_resuming_purpose_speaks_a_different_phrase(monkeypatch, tmp_path) -> None:
    """M2.8. "Ek minute" and "Ji, boliye" are different speech acts, and the on-demand
    fallback has to say the right one for whichever purpose was actually asked for, not
    always the thinking one."""
    import app.routers.voice as router
    from vaani.fillers import FillerBank, Purpose

    synthesised: list[str] = []

    class Stub:
        async def synthesize(self, text, voice, index=0):
            synthesised.append(text)
            yield b"synthesised-filler"

    monkeypatch.setattr(router, "_fillers", FillerBank(assets=tmp_path))
    monkeypatch.setattr(router, "EdgeTts", lambda: Stub())

    chunks = [chunk async for chunk in speak_filler(Purpose.RESUMING)]

    assert chunks == [b"synthesised-filler"]
    assert synthesised == [router.FILLER_TEXT[Purpose.RESUMING]]
    assert synthesised != [router.FILLER_TEXT[Purpose.THINKING]]


async def test_the_resuming_purpose_is_read_from_the_bank_when_present() -> None:
    """The ordinary path, not only the fallback: a committed `resuming` clip exists
    and is what actually gets played, the same way the thinking purpose already is."""
    from vaani.fillers import FillerBank, Purpose

    committed = {clip.read_bytes() for clip in FillerBank().available(Purpose.RESUMING)}
    assert committed, "no committed resuming clips: run scripts/build_fillers.py"

    chunks = [chunk async for chunk in speak_filler(Purpose.RESUMING)]

    assert b"".join(chunks) in committed


@pytest.mark.parametrize("attempt", [1, 2, 3])
async def test_repeated_turns_all_get_it_quickly(attempt: int) -> None:
    """Read per call rather than cached in a module global, because a cache that is
    populated on first use makes the first visitor of every deploy pay for it, and
    that visitor is the one watching."""
    started = time.monotonic()
    chunks = await collect()

    assert chunks
    assert (time.monotonic() - started) * 1000 < 100
