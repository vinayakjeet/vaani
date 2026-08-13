"""Synthesise the filler bank once, so the demo never synthesises one on a live turn.

    python scripts/build_fillers.py

The clips are committed. That is the whole mechanism: a filler exists to cover a 600ms
deadline, and synthesising it on demand was measured at 1796ms, so a bank that builds
itself at startup would put the network back on the one path that is already late, and
would do it on the cold start where it hurts most.

Rerun this when a phrase changes or the voice does. It refuses to overwrite a clip whose
text has not changed, so a rerun costs nothing and cannot silently reshuffle the bank.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaani.fillers import ASSETS, PHRASES, Purpose, filename  # noqa: E402
from vaani.tts import VOICE_HI, EdgeTts  # noqa: E402

# What each committed clip was built from, so a rerun can tell a changed phrase from an
# unchanged one without listening to the audio.
MANIFEST = ASSETS / "fillers.json"


async def build() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest: dict[str, str] = {}
    tts = EdgeTts()
    written = 0

    for purpose in Purpose:
        for index, phrase in enumerate(PHRASES[purpose]):
            name = filename(purpose, index)
            stamp = hashlib.sha256(f"{VOICE_HI}:{phrase}".encode()).hexdigest()[:16]
            manifest[name] = stamp

            if previous.get(name) == stamp and (ASSETS / name).exists():
                print(f"{name}: unchanged")
                continue

            audio = b"".join([chunk async for chunk in tts.synthesize(phrase, VOICE_HI)])
            if not audio:
                print(f"{name}: produced no audio, refusing to write an empty clip")
                return 1
            (ASSETS / name).write_bytes(audio)
            written += 1
            print(f"{name}: {len(audio)} bytes, {len(audio) / 6000:.2f}s, {phrase!r}")

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\n{written} written, {len(manifest)} clips in the bank")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(build()))
