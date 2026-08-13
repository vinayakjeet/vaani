"""Every figure in a spoken reply has to come from somewhere checkable.

`vaani/tools.py` refuses to invent a threshold. Nothing stopped the model stating
one in prose the tool never returned, and prose is what gets synthesised and played
into somebody's ear. The exposure is not a wrong metric. It is telling a person they
qualify for a payment they do not, after which they travel to an office.

So probabilism stops here. The model may phrase, order and explain; it may not
introduce a number. The allowed set is the figures the tools returned plus the
figures the user themselves said, and anything else refuses the reply rather than
speaking it.

**It is deliberately conservative and that has a cost.** A reply saying the applicant
is "50000 above the limit" states a true subtraction that no tool returned, and this
refuses it. Allowing arithmetic would mean allowing any number that happens to be
derivable, which is most of them, so the guardrail would stop being one. The cost is
therefore reported as a rate rather than argued away: `refusals` counts it, and a
configuration whose refusal rate is high has a prompt problem to fix, not a
threshold to loosen.

Only figures of `MIN_CHECKED_DIGITS` or more are checked. Below that a number is a
list marker, a document count or a step number, and the harm in this domain lives in
amounts, limits and years. That is the same reasoning, and the same boundary, that
`vaani/sentences.py` uses to tell "2." in a numbered list from an amount ending a
sentence, and it is set from the shape of the domain rather than measured.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import structlog

from vaani.numerals import normalise

logger = structlog.get_logger(__name__)

# Below three digits a number is structure rather than substance: "2. Form bharein",
# "do documents". Amounts, income limits, years and pincodes are three or more.
MIN_CHECKED_DIGITS = 3

# Said instead of the reply, not appended to it. A refusal that follows the
# ungrounded sentence has already spoken the number it exists to suppress.
REFUSED = (
    "Mujhe is baare mein pakka number nahi mila, to main galat aankda nahi bataunga. "
    "Kripya scheme ke daftar se confirm kar lijiye."
)

refusals: Counter[str] = Counter()


def figures(text: str) -> set[float]:
    """Every number a listener would hear in this text, however it is written.

    Runs through `vaani.numerals`, so "3 lakh", "3,00,000" and "300000" are one
    figure rather than three. A comparison on the written form would let a model
    launder an invented number through a different notation.
    """
    return {
        float(value)
        for value in normalise(text).numbers
        if _significant(float(value))
    }


def sourced(*sources: Any) -> set[float]:
    """The figures a reply is allowed to contain, gathered from tool output.

    Walks whatever the tools returned, including the numbers inside their reason
    strings, because "annual income limit is 500000" is where the threshold a reply
    should quote actually lives.
    """
    found: set[float] = set()
    for source in sources:
        found |= _walk(source)
    return found


def ungrounded(reply: str, allowed: Iterable[float]) -> tuple[float, ...]:
    """Figures in the reply that no source accounts for, smallest first."""
    permitted = {float(value) for value in allowed}
    return tuple(sorted(figures(reply) - permitted))


def check(reply: str, allowed: Iterable[float]) -> str:
    """The reply if every figure in it is sourced, the refusal if not.

    Counts, never the figures. An ungrounded number is derived from an applicant's
    income and this project's threat model keeps those out of logs.
    """
    missing = ungrounded(reply, allowed)
    if not missing:
        return reply

    refusals["ungrounded_figure"] += 1
    logger.warning("grounding.refused", figures=len(missing), reply_chars=len(reply))
    return REFUSED


def _walk(value: Any) -> set[float]:
    if isinstance(value, bool):
        # Checked before int, which bool subclasses. Not a figure either way.
        return set()
    if isinstance(value, int | float):
        return {float(value)} if _significant(float(value)) else set()
    if isinstance(value, str):
        return figures(value)
    if isinstance(value, dict):
        return _walk(list(value.keys())) | _walk(list(value.values()))
    if isinstance(value, list | tuple | set):
        found: set[float] = set()
        for item in value:
            found |= _walk(item)
        return found
    return set()


def _significant(value: float) -> bool:
    """Whether this figure is an amount rather than a list marker or a count."""
    whole = abs(int(value))
    if whole >= 10 ** (MIN_CHECKED_DIGITS - 1):
        return True
    # A decimal is checked whatever its magnitude: 2.5 acres is a landholding
    # against a limit, and the limit is the thing this protects.
    return not float(value).is_integer()
