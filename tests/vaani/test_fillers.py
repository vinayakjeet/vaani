"""The bank, and the property that makes it a bank rather than a folder.

The deadline is shorter than the endpoint wait, so the filler fires on most turns. One
clip repeating is how an IVR sounds, and pure random choice repeats about one time in
eight with eight clips, which is exactly the thing being avoided.
"""

from __future__ import annotations

import random

import pytest

from vaani.fillers import PHRASES, FillerBank, Purpose, filename, read_chunks


def test_the_same_clip_is_never_played_twice_running(tmp_path) -> None:
    for index in range(len(PHRASES[Purpose.THINKING])):
        (tmp_path / filename(Purpose.THINKING, index)).write_bytes(b"x" * 100)

    bank = FillerBank(assets=tmp_path, rng=random.Random(0))
    heard = [bank.pick(Purpose.THINKING) for _ in range(40)]

    assert all(a != b for a, b in zip(heard, heard[1:], strict=False))


def test_a_bank_of_one_still_speaks_rather_than_refusing(tmp_path) -> None:
    """Otherwise a checkout with a partial bank goes silent on its second turn, which is
    worse than the repetition this exists to avoid."""
    (tmp_path / filename(Purpose.THINKING, 0)).write_bytes(b"x" * 100)

    bank = FillerBank(assets=tmp_path)

    assert bank.pick(Purpose.THINKING) is not None
    assert bank.pick(Purpose.THINKING) is not None


def test_an_empty_bank_returns_none_rather_than_raising(tmp_path) -> None:
    """A missing bank must not take a turn down. The caller falls back to synthesising,
    which is slow and audible in the logs, where an exception here would be silence with
    no reply at all."""
    assert FillerBank(assets=tmp_path).pick(Purpose.THINKING) is None


def test_the_two_purposes_do_not_share_clips(tmp_path) -> None:
    """"Ek minute" says the answer is coming and "Ji, boliye" says the floor is yours.
    Playing the second where the first belongs tells somebody to talk while you are
    trying to answer them."""
    (tmp_path / filename(Purpose.THINKING, 0)).write_bytes(b"x" * 100)
    (tmp_path / filename(Purpose.RESUMING, 0)).write_bytes(b"y" * 100)

    bank = FillerBank(assets=tmp_path)

    assert bank.pick(Purpose.THINKING).read_bytes() == b"x" * 100
    assert bank.pick(Purpose.RESUMING).read_bytes() == b"y" * 100


def test_the_legacy_clip_is_used_when_the_bank_was_never_built(tmp_path) -> None:
    """A fork that has not run scripts/build_fillers.py still speaks."""
    (tmp_path / "filler-hi.mp3").write_bytes(b"z" * 100)

    assert FillerBank(assets=tmp_path).pick(Purpose.THINKING) is not None
    assert FillerBank(assets=tmp_path).pick(Purpose.RESUMING) is None


def test_a_clip_is_delivered_in_chunks_like_the_synthesiser(tmp_path) -> None:
    """A client that handles one and not the other works in tests and not in the demo."""
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"a" * 1000)

    chunks = read_chunks(clip, 300)

    assert [len(chunk) for chunk in chunks] == [300, 300, 300, 100]
    assert b"".join(chunks) == b"a" * 1000


def test_the_committed_bank_exists_and_every_clip_is_playable() -> None:
    """The bank is committed rather than built at startup, because a bank that
    synthesises itself has put the network back on the path this mechanism exists to
    remove, on the cold start where it hurts most."""
    bank = FillerBank()
    thinking = bank.available(Purpose.THINKING)

    if not thinking or thinking[0].name == "filler-hi.mp3":
        pytest.skip("bank not built in this checkout; run scripts/build_fillers.py")

    assert len(thinking) == len(PHRASES[Purpose.THINKING])
    assert all(clip.stat().st_size > 1000 for clip in thinking)
    assert len(bank.available(Purpose.RESUMING)) == len(PHRASES[Purpose.RESUMING])
