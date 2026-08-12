"""Whether a partial transcript already sounds like a finished question.

This exists to cut the endpoint wait when the answer is obvious, which is the
largest single term in the optimised latency budget. It buys those milliseconds by
risking cutting people off, and the risk is not evenly distributed: it lands on
whoever phrases things unusually, pauses to think mid-sentence, or speaks a dialect
the rule below was not written for. So the false-endpoint rate is measured and
published beside the milliseconds, never instead of them.

The rule leans on word order rather than on a model. Hindi is verb-final, so a
finished utterance almost always ends in a verb or an auxiliary, and Hinglish keeps
that frame even when the nouns are English: "mera income kitna hona chahiye" ends
Hindi however much of the middle is not. Anything else is treated as unfinished,
which is the safe direction to be wrong in: the cost is dead air the full timeout
would have spent anyway, where the other direction talks over somebody.
"""

from __future__ import annotations

import re

# Sentence-final verbs and auxiliaries, in both scripts. Not a grammar, and not
# meant to be one: a short list of the endings this domain actually produces,
# extended when the eval set turns up a miss rather than by guessing wider.
FINAL_MARKERS = frozenset(
    {
        # Copulas and existentials.
        "hai",
        "hain",
        "hoon",
        "hun",
        "ho",
        "tha",
        "thi",
        "the",
        "है",
        "हैं",
        "हूँ",
        "हूं",
        "हो",
        "था",
        "थी",
        "थे",
        # Modals and the handful of verbs an eligibility question ends on.
        "chahiye",
        "milega",
        "milegi",
        "milta",
        "milti",
        "sakta",
        "sakti",
        "sakte",
        "karna",
        "karoon",
        "karun",
        "hoga",
        "hogi",
        "batao",
        "bataiye",
        "kijiye",
        "चाहिए",
        "मिलेगा",
        "मिलेगी",
        "मिलता",
        "मिलती",
        "सकता",
        "सकती",
        "सकते",
        "करना",
        "करूँ",
        "करूं",
        "होगा",
        "होगी",
        "बताओ",
        "बताइए",
        "कीजिए",
    }
)

# An utterance ending on one of these is mid-clause whatever else it looks like.
# Postpositions and conjunctions cannot end a sentence, so they are a reliable
# negative even when the marker list above would otherwise have nothing to say.
CONTINUATIONS = frozenset(
    {
        "aur",
        "lekin",
        "ya",
        "ka",
        "ki",
        "ke",
        "se",
        "me",
        "mein",
        "par",
        "ko",
        "mera",
        "meri",
        "mere",
        "kya",
        "और",
        "लेकिन",
        "या",
        "का",
        "की",
        "के",
        "से",
        "में",
        "पर",
        "को",
        "मेरा",
        "मेरी",
        "मेरे",
    }
)

TERMINATORS = "।॥?!."

# Below this, a partial is too short to be a question. "haan" and "theek" are
# answers to something nobody asked yet, and endpointing on them starts a turn on
# a fragment.
MIN_WORDS = 3

_WORD = re.compile(r"[^\s]+")


def looks_complete(partial: str) -> bool:
    """Whether this partial can be treated as a finished utterance.

    Deliberately conservative. Every branch that is not clearly complete returns
    False and the caller waits the full trailing silence, so a miss costs dead air
    and never costs the end of somebody's sentence.
    """
    text = partial.strip()
    if not text:
        return False

    # Punctuation on its own is not a word. A recogniser that emits a question mark
    # before any speech would otherwise endpoint the turn on nothing.
    words = [stripped for word in _WORD.findall(text) if (stripped := word.strip(TERMINATORS))]
    if not words:
        return False
    last = words[-1].casefold()

    # Checked before the terminator shortcut below, which is the whole point of
    # having it. A recogniser routinely punctuates after a number, so "meri aay
    # 50000." arrives looking finished while the speaker is still saying "pachaas
    # hazaar rupaye". An amount cut in half is a wrong eligibility answer delivered
    # confidently, which is the most expensive mistake this pipeline can make.
    if last.isdigit():
        return False

    # An explicit terminator settles the rest. This is what catches a Hinglish
    # question ending on an English word, since a recogniser that heard question
    # intonation usually punctuates it, and the word-order rule below has nothing to
    # say about "documents".
    if text[-1] in TERMINATORS:
        return True

    if len(words) < MIN_WORDS:
        return False
    if last in CONTINUATIONS:
        return False

    return last in FINAL_MARKERS
