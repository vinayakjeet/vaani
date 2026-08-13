"""Treat a figure nobody could read as a question rather than an answer.

One confirmation turn is cheaper than one wrong eligibility answer, and the second
thing is not a metric: it is somebody travelling to an office because an agent told
them confidently that they qualify.

`vaani/numerals.py` already refuses to guess, so the signal exists: `unresolved` holds
the number phrases that were spoken and could not be read. Answering anyway means one
of two things happened, and both are bad. Either the model normalises the phrase itself,
which is the job that module exists to take away from it, or the amount is simply
missing from the prompt and the reply is built on whatever the model assumes.

**This is a technique with a cost, not an improvement.** It spends a whole turn of the
latency this project exists to reduce, on exactly the utterances a real caller produces:
Hindi says most numbers between twenty and a hundred as single irregular words, and the
lexicon deliberately covers the round ones rather than guessing at the rest. So the
confirmation rate is a measured output, `confirmations` is a counter, and the ablation
runs the arm with `confirm=False` to show what the turn buys and what it costs. A
configuration that never confirms is not obviously worse and is not obviously better.

**It does not repeat the number back.** The usual confirmation UX echoes what was heard,
and here what was heard is precisely the thing that could not be read, so echoing it
would either say the words back uselessly or say a figure this module has no right to
have. Asking again is weaker UX and it is the only version that cannot invent an amount.
"""

from __future__ import annotations

from collections import Counter

import structlog

from vaani.numerals import Normalised

logger = structlog.get_logger(__name__)

# Hinglish, because the reply will be. Short, because it is the whole turn.
ASK_AGAIN = (
    "Maaf kijiye, raqam main theek se sun nahi paaya. "
    "Kya aap wo number dobara bata sakte hain?"
)

confirmations: Counter[str] = Counter()


def needs_confirming(spoken: Normalised) -> bool:
    """Whether this transcript has a number in it that nobody could read.

    Only unresolved figures, not every figure. Confirming a number that was read
    cleanly would double the length of every conversation to protect against a failure
    that did not happen, and a caller asked to repeat themselves constantly stops
    answering carefully, which makes the next reading worse rather than better.
    """
    return bool(spoken.unresolved)


def question() -> str:
    confirmations["unreadable_figure"] += 1
    # The count, never the phrase. The phrase is an applicant's income.
    logger.info("confirm.asked")
    return ASK_AGAIN
