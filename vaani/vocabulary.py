"""Domain words fed to the recogniser so it stops mishearing scheme names.

Whisper's transcription endpoint takes a prompt that biases decoding toward the
vocabulary in it. Costed at zero milliseconds and, on entity accuracy, frequently
worth more than changing vendors: "PM-KISAN" comes back as "PM Kissan", "peeem
kisaan" or "P.M. Kissan" from a model that has never been told the string exists.

The list is derived from `vaani/tools.py` rather than written out again. Two copies
of the scheme names would drift, and the copy that drifted would be the one the
recogniser was biased toward, so the pipeline would be tuned to hear a name the
tool cannot look up.

Two properties this needs and a naive prompt does not have.

**It is bounded.** Whisper's prompt window is 224 tokens and it silently keeps the
tail when you exceed it, so an unbounded vocabulary quietly biases toward whichever
names happen to sort last.

**Its echo is caught.** A biasing prompt is prepended to the decoder's context, so a
model given one and near-silence sometimes transcribes the prompt back. That is not a
rare edge: it is the documented failure of Whisper's `initial_prompt`, and here it
would put "PM-KISAN Ayushman Bharat" into a turn where the user said nothing, which
the pipeline would then answer confidently. `is_echo` is the check, and the recogniser
treats a hit as an empty transcript rather than as speech.
"""

from __future__ import annotations

from vaani.tools import RULES

# Whisper's prompt window. The bias list is trimmed to fit rather than truncated by
# the provider, because a provider-side truncation drops the tail without saying so.
MAX_PROMPT_CHARS = 600

# Words the domain needs and no scheme name carries: the units an amount is spoken
# in, and the two documents every eligibility question mentions. Short on purpose,
# because every term added dilutes the bias on the ones that matter.
DOMAIN_TERMS: tuple[str, ...] = (
    "hazaar",
    "lakh",
    "crore",
    "acre",
    "aay",
    "yojana",
    "Aadhaar",
    "ration card",
)


def bias_terms() -> tuple[str, ...]:
    """Scheme names in both scripts, then the domain terms, in a stable order.

    Stable because a prompt that changes between requests changes the decoding of
    identical audio, and the ablation would then measure the ordering as if it were
    a technique.
    """
    names: list[str] = []
    for rule in RULES:
        names.append(rule.scheme.name_en)
        names.append(rule.scheme.name_hi)
    return tuple(names) + DOMAIN_TERMS


def bias_prompt() -> str:
    """The bias list as one prompt, trimmed to the window rather than past it."""
    prompt = ""
    for term in bias_terms():
        candidate = f"{prompt}, {term}" if prompt else term
        if len(candidate) > MAX_PROMPT_CHARS:
            break
        prompt = candidate
    return prompt


def is_echo(text: str, prompt: str) -> bool:
    """Whether the recogniser transcribed the bias prompt instead of speech.

    Compared on the terms rather than on the whole string, because the echo comes
    back reordered and repunctuated. A transcript whose words are all bias terms and
    nothing else is the prompt coming home, and no real question about a scheme
    consists solely of scheme names: it has a verb.
    """
    heard = {word for word in _words(text) if word}
    if not heard:
        return False
    offered = {word for term in prompt.split(",") for word in _words(term)}
    return heard.issubset(offered)


def _words(text: str) -> list[str]:
    return [word.strip(".,!?;:।॥-").casefold() for word in text.split()]
