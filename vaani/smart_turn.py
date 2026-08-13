"""smart-turn as the measured arm against our word-order rule.

`vaani/completeness.py` decides whether an utterance sounds finished by looking at its
last word, because Hindi is verb-final. It is a hand-written approximation of a model
that now exists in 8MB: `pipecat-ai/smart-turn-v3`, BSD-2, a Whisper-tiny backbone at 8M
parameters, 10ms to 100ms on a CPU, taking 16kHz mono PCM, which is exactly what the
capture worklet already produces.

The rule stays as the baseline arm. SPEC wants both reported with their false-endpoint
rates, and a model adopted without measuring it against what it replaced is an upgrade
nobody can defend. There is also a real reason to expect the rule to win sometimes: the
model is trained on English and Chinese conversational speech, and this is Hinglish.

**It runs once per turn, not per frame.** At up to 100ms an inference, fifty times a
second is not an option. The one moment the answer changes anything is when trailing
silence first reaches the short timeout: cut now, or wait out the full one. That is the
same decision the rule makes and the same place it is asked.

**The features are Whisper's, computed here rather than pulled from torch.** The model
takes an 80 by 800 log-mel spectrogram, so the extractor is part of the arm: get the
filterbank wrong and the model returns confident numbers from noise, the suite stays
green, and the ablation concludes that neural endpointing does not help. The filterbank
is therefore checked against OpenAI's published `mel_filters.npz` by a test rather than
trusted, and it agrees to within float32 round-off.

What is *not* verified here is whether the model is any good on Hinglish. That is a
measurement against the recorded corpus, and until it exists this is an arm that has
been wired up rather than one shown to be better.
"""

from __future__ import annotations

import math

import structlog

from vaani.models import SMART_TURN, session
from vaani.protocol import SAMPLE_RATE

logger = structlog.get_logger(__name__)

# Whisper's feature geometry, which the exported graph's input shape fixes rather than
# suggests: 80 mel bins by 800 frames is 8 seconds at a 10ms hop.
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
FRAMES = 800
MAX_SAMPLES = SAMPLE_RATE * 8

# Slaney mel scale constants, which are what "norm=slaney, mel_scale=slaney" means in
# the feature extractor smart-turn was trained with. Written out rather than imported so
# this file does not depend on librosa for four numbers.
_F_SP = 200.0 / 3.0
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = 15.0

# Above this the turn is called complete. The project default rather than a measurement
# of ours, and swept in the ablation for the same reason the endpointer's other
# thresholds are.
DEFAULT_COMPLETION_PROBABILITY = 0.5


class SmartTurn:
    """Whether the audio so far is a finished utterance, from the model rather than a rule."""

    name = "smart-turn-v3.2"

    def __init__(self, threshold: float = DEFAULT_COMPLETION_PROBABILITY) -> None:
        import numpy

        self._numpy = numpy
        self._session = session(SMART_TURN)
        self._threshold = threshold
        self._filters = mel_filters(numpy)
        # Periodic Hann, which is what torch.stft uses and what the training-time
        # extractor produced. numpy's `hanning` is the symmetric window and is a
        # different function; using it shifts every frame's energy slightly, which is
        # invisible in the output and wrong in the input.
        self._window = numpy.hanning(N_FFT + 1)[:-1].astype(numpy.float64)
        self.probability = 0.0

    def __call__(self, pcm: bytes) -> bool:
        return self.accept(pcm) >= self._threshold

    def accept(self, pcm: bytes) -> float:
        numpy = self._numpy
        samples = numpy.frombuffer(pcm, dtype="<i2").astype(numpy.float64) / 32768.0

        # The last eight seconds, padded at the end. That is the shape the model was
        # trained on: speech, then the silence that may or may not be the end of a turn.
        # Padding at the front instead would put the candidate endpoint in the middle of
        # the window and ask a different question.
        samples = samples[-MAX_SAMPLES:]
        if len(samples) < MAX_SAMPLES:
            samples = numpy.pad(samples, (0, MAX_SAMPLES - len(samples)))

        features = log_mel(numpy, samples, self._filters, self._window)
        logits = self._session.run(
            None, {"input_features": features.reshape(1, N_MELS, FRAMES)}
        )[0]
        self.probability = 1.0 / (1.0 + math.exp(-float(logits[0][0])))
        return self.probability


def hz_to_mel(numpy, hz):
    hz = numpy.asarray(hz, dtype=numpy.float64)
    linear = hz / _F_SP
    # Guarded so hz=0 does not evaluate log(0) inside the unused branch of the where.
    logarithmic = _MIN_LOG_MEL + numpy.log(
        numpy.maximum(hz, 1e-10) / _MIN_LOG_HZ
    ) / (numpy.log(6.4) / 27.0)
    return numpy.where(hz < _MIN_LOG_HZ, linear, logarithmic)


def mel_to_hz(numpy, mel):
    mel = numpy.asarray(mel, dtype=numpy.float64)
    return numpy.where(
        mel < _MIN_LOG_MEL,
        _F_SP * mel,
        _MIN_LOG_HZ * numpy.exp((numpy.log(6.4) / 27.0) * (mel - _MIN_LOG_MEL)),
    )


def mel_filters(numpy):
    """Whisper's 80 by 201 mel filterbank, triangular with Slaney area normalisation."""
    mels = numpy.linspace(
        hz_to_mel(numpy, 0.0), hz_to_mel(numpy, SAMPLE_RATE / 2), N_MELS + 2
    )
    hz = mel_to_hz(numpy, mels)
    fft_hz = numpy.linspace(0, SAMPLE_RATE / 2, 1 + N_FFT // 2)

    ramps = hz[:, None] - fft_hz[None, :]
    widths = numpy.diff(hz)
    weights = numpy.zeros((N_MELS, len(fft_hz)))
    for band in range(N_MELS):
        rising = -ramps[band] / widths[band]
        falling = ramps[band + 2] / widths[band + 1]
        weights[band] = numpy.maximum(0, numpy.minimum(rising, falling))

    # Slaney normalisation makes each filter's area rather than its peak constant, so a
    # wide high-frequency band does not dominate a narrow low one.
    return weights * (2.0 / (hz[2 : N_MELS + 2] - hz[:N_MELS]))[:, None]


def log_mel(numpy, samples, filters, window):
    """Whisper's log-mel spectrogram, including its two normalisation steps.

    The clamp to eight decades below the maximum and the affine rescale are not
    cosmetic: they are what the model's input distribution is, and omitting either
    produces features in the wrong range that the graph will still happily consume.
    """
    padded = numpy.pad(samples, (N_FFT // 2, N_FFT // 2), mode="reflect")
    starts = numpy.arange(0, len(padded) - N_FFT + 1, HOP_LENGTH)
    frames = numpy.stack([padded[start : start + N_FFT] * window for start in starts])
    spectrum = numpy.fft.rfft(frames, n=N_FFT, axis=-1)

    # The last frame is dropped, as it is in Whisper: the reflected padding makes it a
    # duplicate of the boundary rather than audio.
    magnitudes = (numpy.abs(spectrum) ** 2).T[:, :-1]

    mel = filters @ magnitudes
    log_spec = numpy.log10(numpy.maximum(mel, 1e-10))
    log_spec = numpy.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(numpy.float32)
