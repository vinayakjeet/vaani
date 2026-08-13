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

from app.routers.voice import FILLER_AUDIO, FILLER_CHUNK_BYTES, speak_filler


async def collect() -> list[bytes]:
    return [chunk async for chunk in speak_filler()]


def test_the_asset_is_committed() -> None:
    """A fork without it falls back to synthesising, which works and is slow. The
    asset existing is what makes the fast path the normal one."""
    assert FILLER_AUDIO.exists()
    assert FILLER_AUDIO.stat().st_size > 1000


async def test_it_speaks_from_disk_with_no_provider() -> None:
    chunks = await collect()

    assert chunks
    assert b"".join(chunks) == FILLER_AUDIO.read_bytes()


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


async def test_a_missing_asset_still_speaks(monkeypatch) -> None:
    """A fork that has not generated one must not go silent. Silence is the outcome
    every degradation rule in SPEC exists to prevent, and it is worse than a slow
    acknowledgement."""
    import app.routers.voice as router

    class Missing:
        def read_bytes(self) -> bytes:
            raise OSError("no such file")

        def __str__(self) -> str:
            return "missing.mp3"

    synthesised: list[str] = []

    class Stub:
        async def synthesize(self, text, voice, index=0):
            synthesised.append(text)
            yield b"synthesised-filler"

    monkeypatch.setattr(router, "FILLER_AUDIO", Missing())
    monkeypatch.setattr(router, "EdgeTts", lambda: Stub())

    assert await collect() == [b"synthesised-filler"]
    assert synthesised == [router.FILLER_TEXT]


@pytest.mark.parametrize("attempt", [1, 2, 3])
async def test_repeated_turns_all_get_it_quickly(attempt: int) -> None:
    """Read per call rather than cached in a module global, because a cache that is
    populated on first use makes the first visitor of every deploy pay for it, and
    that visitor is the one watching."""
    started = time.monotonic()
    chunks = await collect()

    assert chunks
    assert (time.monotonic() - started) * 1000 < 100
