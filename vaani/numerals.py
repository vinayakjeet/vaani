"""Spoken Hindi and Hinglish numbers into digits, in code rather than in a model.

The rule this exists for: never delegate numeric normalisation to the thing that is
also doing the reasoning. For most domains that is a tidiness preference. Here the
number *is* the answer. "pachaas hazaar" read as 50 instead of 50000 moves an
applicant across an income limit, and the reply that follows is confident, spoken
aloud, and wrong in the direction that sends somebody to an office for nothing.

Two properties matter more than coverage.

It refuses rather than guesses. A run of number words containing anything this
module does not know is left exactly as it was and reported in `unresolved`, so the
caller can confirm it out loud instead of letting the model invent a reading. That
is the same call `vaani/tools.py` makes about a scheme with no thresholds recorded:
"we cannot check this" is an answer and a plausible-looking number is not.

And two adjacent unit words are ambiguous rather than additive. Hindi composes 55 as
one word, "pachpan", so "paanch pachaas" is not 55: it is either a misrecognition or
a speaker correcting themselves, and both of those want a human to hear about it.
English habits leak into Hinglish here, which is exactly why the guess is refused.

The Indian grouping is not decoration. lakh and crore are section boundaries, so
"ek lakh pachaas hazaar" is 1x100000 + 50x1000 and not 1x100000x50x1000, and an
algorithm that folds multipliers left to right gets that wrong silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# 0 to 20 plus the round tens, in both scripts. Deliberately not all 99: the words
# in between are single tokens ("pachpan", "chauvvan") and adding them one at a time
# as the eval set turns them up is how the endpointer's marker list is maintained
# too. A missing word is reported, not guessed, so partial coverage is safe.
_UNITS: dict[str, float] = {
    "shunya": 0, "zero": 0, "शून्य": 0,
    "ek": 1, "एक": 1,
    "do": 2, "दो": 2,
    "teen": 3, "तीन": 3,
    "char": 4, "chaar": 4, "चार": 4,
    "panch": 5, "paanch": 5, "पांच": 5, "पाँच": 5,
    "chhe": 6, "che": 6, "chah": 6, "छह": 6, "छे": 6,
    "sat": 7, "saat": 7, "सात": 7,
    "ath": 8, "aath": 8, "आठ": 8,
    "nau": 9, "नौ": 9,
    "das": 10, "dus": 10, "दस": 10,
    "gyarah": 11, "ग्यारह": 11,
    "barah": 12, "बारह": 12,
    "terah": 13, "तेरह": 13,
    "chaudah": 14, "चौदह": 14,
    "pandrah": 15, "पंद्रह": 15,
    "solah": 16, "सोलह": 16,
    "satrah": 17, "सत्रह": 17,
    "atharah": 18, "अठारह": 18,
    "unnees": 19, "उन्नीस": 19,
    "bees": 20, "बीस": 20,
    "tees": 30, "तीस": 30,
    "chalees": 40, "चालीस": 40,
    "pachaas": 50, "pachas": 50, "पचास": 50,
    "saath": 60, "साठ": 60,
    "sattar": 70, "सत्तर": 70,
    "assi": 80, "अस्सी": 80,
    "nabbe": 90, "नब्बे": 90,
}

# Values in their own right rather than modifiers: "dedh lakh" is 1.5 lakh.
_FRACTIONS: dict[str, float] = {
    "dedh": 1.5, "derh": 1.5, "डेढ़": 1.5, "डेढ": 1.5,
    "dhai": 2.5, "adhai": 2.5, "ढाई": 2.5, "अढ़ाई": 2.5,
    "aadha": 0.5, "adha": 0.5, "आधा": 0.5,
}

# A quarter added to or taken off whatever number follows. "sava lakh" is 1.25 lakh
# with the one implied; "paune do lakh" is 1.75 lakh.
_OFFSETS: dict[str, float] = {
    "sava": 0.25, "sawa": 0.25, "सवा": 0.25,
    "paune": -0.25, "पौने": -0.25,
}

_SCALES: dict[str, int] = {
    "sau": 100, "सौ": 100,
    "hazaar": 1_000, "hazar": 1_000, "hajar": 1_000, "हजार": 1_000, "हज़ार": 1_000,
    "lakh": 100_000, "lac": 100_000, "lakhs": 100_000, "लाख": 100_000,
    "crore": 10_000_000, "karod": 10_000_000, "करोड़": 10_000_000, "करोड": 10_000_000,
}

# Scales that open a new section of the Indian grouping. Below this a scale
# multiplies the group in place, so "do sau pachaas" can add fifty afterwards.
_SECTION_SCALE = 1_000

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Digits, optionally with a decimal point or Indian-style grouping commas. Matched
# as a number token so "50 hazaar" and "2.5 lakh" resolve: a recogniser routinely
# writes the count as digits and the scale as a word, and treating those as two
# unrelated tokens leaves the multiplier for the model to apply.
_NUMERIC = re.compile(r"^\d+(?:,\d+)*(?:\.\d+)?$")

_TOKEN = re.compile(r"\S+")

# Trailing punctuation a recogniser attaches to the last word of a phrase. Stripped
# for lookup and put back on the way out, so "pachaas hazaar." keeps its full stop
# and `vaani.sentences` still sees a sentence boundary there.
_EDGE = ".,!?;:।॥\"'()"


@dataclass(frozen=True)
class Normalised:
    """The transcript with numbers as digits, and what could not be resolved.

    `unresolved` is the whole point of the return type. A number nobody could read
    is a fact the turn needs, because the honest response to it is a confirmation
    question rather than an answer, and a plain string gives the caller no way to
    tell a resolved number from one that was left alone.
    """

    text: str
    numbers: tuple[int | float, ...]
    unresolved: tuple[str, ...]

    @property
    def confident(self) -> bool:
        return not self.unresolved


def normalise(transcript: str) -> Normalised:
    """Rewrite spoken numbers as digits, leaving anything ambiguous untouched."""
    tokens = [(match.group(), match.start()) for match in _TOKEN.finditer(transcript)]
    if not tokens:
        return Normalised(text=transcript, numbers=(), unresolved=())

    pieces: list[str] = []
    numbers: list[int | float] = []
    unresolved: list[str] = []

    index = 0
    while index < len(tokens):
        word = _bare(tokens[index][0])
        if not _is_number_word(word):
            pieces.append(tokens[index][0])
            index += 1
            continue

        run_end = index
        while run_end < len(tokens) and _is_number_word(_bare(tokens[run_end][0])):
            run_end += 1

        run = [token for token, _ in tokens[index:run_end]]
        value = _evaluate([_bare(token) for token in run])
        raw = " ".join(run)

        if value is None:
            # Left exactly as spoken. A partially normalised phrase is worse than an
            # untouched one, because it reads as though something checked it.
            unresolved.append(raw)
            logger.info("numerals.unresolved", words=len(run))
            pieces.append(raw)
        else:
            numbers.append(value)
            pieces.append(_render(value) + _trailing(run[-1]))

        index = run_end

    return Normalised(
        text=" ".join(pieces),
        numbers=tuple(numbers),
        unresolved=tuple(unresolved),
    )


def _bare(token: str) -> str:
    return token.strip(_EDGE).translate(_DEVANAGARI_DIGITS).casefold()


def _trailing(token: str) -> str:
    """Punctuation the last word of a run carried, so it survives the rewrite."""
    stripped = token.rstrip(_EDGE)
    return token[len(stripped):]


def _is_number_word(word: str) -> bool:
    return bool(word) and (
        word in _UNITS
        or word in _FRACTIONS
        or word in _OFFSETS
        or word in _SCALES
        or bool(_NUMERIC.match(word))
    )


def _evaluate(words: list[str]) -> int | float | None:
    """One run of number words as a single value, or None if it is ambiguous.

    None is a real answer here rather than a failure. Every branch that cannot be
    read with certainty returns it, because the cost of being wrong is an
    eligibility verdict and the cost of refusing is one confirmation turn.
    """
    total = 0.0
    group = 0.0
    offset: float | None = None
    seen = False

    for word in words:
        if word in _OFFSETS:
            if offset is not None:
                return None
            offset = _OFFSETS[word]
            continue

        if word in _SCALES:
            scale = _SCALES[word]
            if offset is not None:
                # "sava lakh": the one is implied, so the quarter applies to it.
                group = (group or 1) + offset
                offset = None
            elif not group:
                # A scale with nothing counting it. Reading "hazaar" as 1000 looks
                # harmless and is how this module invented a number: the word before
                # it was "pachpan", which the lexicon does not know, so the run began
                # at the scale and "pachpan hazaar" normalised to "pachpan 1000". A
                # count that was spoken and not understood is indistinguishable from
                # one that was never there, so both refuse.
                return None
            factor = group if group else 1.0
            if scale >= _SECTION_SCALE:
                total += factor * scale
                group = 0.0
            else:
                group = factor * scale
            seen = True
            continue

        value = _value_of(word)
        if value is None:
            return None
        if offset is not None:
            value += offset
            offset = None

        if group:
            # Only a hundreds group takes an addition, which is what makes "do sau
            # pachaas" work and "paanch pachaas" a refusal rather than 55.
            if group % 100 or value >= 100:
                return None
            group += value
        else:
            group = value
        seen = True

    if not seen or offset is not None:
        return None

    result = total + group
    return int(result) if float(result).is_integer() else round(result, 3)


def _value_of(word: str) -> float | None:
    if word in _UNITS:
        return _UNITS[word]
    if word in _FRACTIONS:
        return _FRACTIONS[word]
    if _NUMERIC.match(word):
        return float(word.replace(",", ""))
    return None


def _render(value: int | float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
