"""Cut a token stream into sentences so synthesis can start on the first one.

This is where the largest single latency win in the project lives. Waiting for a
whole reply before synthesising cost 2000ms in the M0 measurement; the first
sentence of the same reply is a few hundred milliseconds of audio and can start
almost immediately.

Segmentation is harder here than for English prose, for three reasons that are
all represented in the tests:

- Devanagari ends sentences with a danda, not a full stop.
- Hinglish mixes both in one reply, sometimes in one sentence.
- A decimal point, an abbreviation, or "Rs. 6000" is not a sentence boundary,
  and splitting there produces synthesis that pauses mid-number.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing

# Danda and double danda for Devanagari, then the Latin terminators. A newline
# is deliberately not a terminator: models emit them inside lists and mid-clause,
# and flushing on one produces a fragment that reads as a stutter when spoken.
TERMINATORS = "।॥.?!"

# A first sentence shorter than this is almost always a false split: an
# abbreviation, an initial, or a stray full stop. Synthesising it alone costs a
# round trip to say two syllables and then a gap.
MIN_SENTENCE_CHARS = 12

# Beyond this, flush whatever is buffered even without a terminator. A model that
# never emits one, or emits a very long first clause, must not hold the audio
# forever. The cost is a slightly awkward break; the alternative is silence.
MAX_BUFFER_CHARS = 220

# How many digits before a full stop make it an amount rather than a list marker.
# "2. Form bharein" and "300000. Aap eligible hain" are the same three characters
# and opposite answers, and nothing local to the full stop separates them except
# how long the number is. A marker is one or two digits; an amount, a year or a
# pincode is three or more. Set from the shape of the domain rather than measured,
# so it is stated here rather than buried in a condition.
MIN_AMOUNT_DIGITS = 3


def _digit_run(buffer: str, end: int) -> int:
    """How many consecutive digits end at `end`, inclusive.

    Digit grouping separators count, so "1,50,000" reads as one number rather
    than as a three-digit one, which is how amounts are written here.
    """
    length = 0
    index = end
    while index >= 0 and (buffer[index].isdigit() or buffer[index] == ","):
        if buffer[index].isdigit():
            length += 1
        index -= 1
    return length


def _is_boundary(buffer: str, index: int, *, final: bool) -> bool:
    """Whether the terminator at `index` actually ends a sentence."""
    char = buffer[index]
    if char != ".":
        return True

    before = buffer[index - 1] if index > 0 else ""
    if not before.isdigit():
        return True

    after = buffer[index + 1] if index + 1 < len(buffer) else ""
    if after.isdigit():
        # A full stop between two digits is a decimal. "2.5 hectare" and
        # "1.5 lakh" appear in almost every eligibility answer.
        return False
    if after in (" ", ","):
        # Either a list marker or a sentence that happens to end in a number.
        # Suppressing both, which is what a flat digit rule does, costs the
        # boundary in "Aapki limit hai 300000. Aap eligible hain", so sentence one
        # never flushes early and the milestone's whole point is lost on the
        # commonest shape of answer there is.
        return _digit_run(buffer, index - 1) >= MIN_AMOUNT_DIGITS

    # Nothing after it yet, which is two different situations. Mid-stream the
    # buffer ends at "2." for as long as the next token takes to arrive, and a
    # decimal is indistinguishable from a finished sentence at that instant, so
    # the decision waits. In a reply that has ended it is a full stop, and
    # reading it as one is what lets "limit hai 300000." flush at all.
    return final


def split(text: str, *, final: bool) -> tuple[list[str], str]:
    """Return complete sentences and whatever is left unterminated.

    `final` says whether more text can still arrive. It only changes the reading
    of a full stop that sits at the very end after a digit.
    """
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char not in TERMINATORS or not _is_boundary(text, index, final=final):
            continue
        candidate = text[start : index + 1].strip()
        if len(candidate) < MIN_SENTENCE_CHARS:
            continue
        sentences.append(candidate)
        start = index + 1
    return sentences, text[start:]


def sentences(text: str) -> Iterator[str]:
    """Every sentence in a finished string, including an unterminated tail."""
    complete, tail = split(text, final=True)
    yield from complete
    if tail.strip():
        yield tail.strip()


async def from_stream(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield each sentence as soon as the stream completes it.

    The point of the whole module: the caller can hand sentence one to synthesis
    while the model is still writing sentence three.
    """
    buffer = ""
    # `aclosing`: `tokens` is `StreamedTurn.run`, which holds its own `stage_span`
    # open across a `yield`. This generator is itself abandonable (by
    # `speak_as_they_arrive`, which wraps it the same way), and being idle at one
    # of the `yield sentence` calls below when that happens would otherwise leave
    # `tokens` open with nothing to close it.
    async with aclosing(tokens):
        async for token in tokens:
            buffer += token
            complete, buffer = split(buffer, final=False)
            for sentence in complete:
                yield sentence
            if len(buffer) >= MAX_BUFFER_CHARS:
                if buffer.strip():
                    yield buffer.strip()
                buffer = ""

    # The stream has ended, so re-read the remainder knowing nothing more is
    # coming. A trailing "300000." is a sentence now, where a moment ago it could
    # still have been a decimal waiting for its digits.
    complete, buffer = split(buffer, final=True)
    for sentence in complete:
        yield sentence
    if buffer.strip():
        yield buffer.strip()
