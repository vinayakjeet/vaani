"""Download the optional ONNX arms, checking each against a pinned digest.

Not committed to the repo and not downloaded at import time. A model file is
executable input to a decision about somebody's welfare eligibility, so it is fetched
deliberately, verified, and pinned: "it downloaded something" is not "it downloaded
this", and both of these are served from URLs whose contents can change.

    python scripts/fetch_models.py

Without them the free stack runs exactly as before, on the energy detector and the
word-order rule, which are the ablation's baseline arms rather than a degraded mode.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

# Run as a script from anywhere, without asking the caller to set PYTHONPATH first. A
# setup step that fails on invocation is a setup step people work around.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaani.models import CHECKSUMS, MODEL_DIR, SOURCES  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str) -> bool:
    target = MODEL_DIR / name
    expected = CHECKSUMS[name]

    if target.exists() and digest(target.read_bytes()) == expected:
        print(f"{name}: already present and matches")
        return True

    print(f"{name}: downloading from {SOURCES[name]}")
    with urllib.request.urlopen(SOURCES[name]) as response:  # noqa: S310
        payload = response.read()

    found = digest(payload)
    if found != expected:
        # Not written. A file that fails its digest is either the wrong version or
        # tampered with, and keeping it on disk means the next run finds it and uses it.
        print(f"{name}: digest mismatch\n  expected {expected}\n  got      {found}")
        return False

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    print(f"{name}: {len(payload) // 1024}KB, digest verified")
    return True


def main() -> int:
    return 0 if all(fetch(name) for name in sorted(CHECKSUMS)) else 1


if __name__ == "__main__":
    sys.exit(main())
