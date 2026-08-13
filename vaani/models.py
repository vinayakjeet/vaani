"""Where the ONNX models live, and refusing to run without the one you asked for.

Two small models are optional arms in the ablation: Silero for "is this speech at all"
and smart-turn for "has this turn ended". Neither is committed. They are fetched by
`scripts/fetch_models.py`, which pins a SHA-256 for each, because a model file is
executable input to a decision about somebody's welfare eligibility and "it downloaded
something" is not the same as "it downloaded this".

Absence is a supported state and not an error. The energy detector and the word-order
rule are the baseline arms the ablation measures against, so a checkout with no models
runs the free stack exactly as before. What is not supported is silently falling back:
a caller that asks for Silero and gets energy would publish an ablation row for a
technique that never ran, which is the failure this project exists to avoid.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo-local by default so a bench run and the server read the same file, and
# overridable because Render's filesystem is not this one.
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR = Path(os.environ.get("VAANI_MODEL_DIR", _DEFAULT_DIR))

SILERO_VAD = "silero_vad.onnx"
SMART_TURN = "smart-turn-v3.2-cpu.onnx"

# Pinned at the versions measured. A model that changes underneath a published number
# makes the number unreproducible, and these are both served from mutable URLs.
CHECKSUMS = {
    SILERO_VAD: "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    SMART_TURN: "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
}

SOURCES = {
    SILERO_VAD: "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
    SMART_TURN: "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
}


class ModelMissing(RuntimeError):
    """The model this arm needs is not on disk, or onnxruntime is not installed."""


def path(name: str) -> Path:
    found = MODEL_DIR / name
    if not found.exists():
        raise ModelMissing(
            f"{name} is not in {MODEL_DIR}. Run scripts/fetch_models.py, or use the "
            "baseline arm instead of asking for this one."
        )
    return found


def session(name: str):
    """One ONNX inference session, single-threaded.

    Single-threaded on purpose. This runs inside the event loop on a 512MB instance
    handling one conversation, and letting onnxruntime spawn a thread per core on a
    2MB model spends more time coordinating than computing.
    """
    try:
        import onnxruntime
    except ImportError as exc:
        raise ModelMissing("onnxruntime is not installed") from exc

    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(path(name)), options, providers=["CPUExecutionProvider"]
    )
