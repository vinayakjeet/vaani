"""The conversation the model is shown, holding only what the listener actually heard.

This is the first version of this record, and it is built with the truncation discipline
rather than gaining it later, because the order matters. Build the record first and the
bug ships: a reply is written into history when the model produces it, the user barges in
after the first sentence, and from then on the model believes it said two sentences
nobody heard. Coherence does not fail immediately. It fails two turns later, when the
agent refers back to a figure it never spoke, and by then the cause is three exchanges
upstream of the symptom.

**The ordinary case needs no reconstruction at all.** An assistant turn is staged when it
is produced and committed only when its audio has actually reached the transport. Text
whose audio was blocked or abandoned never enters the record, so the common interruption
is handled by not writing rather than by repairing. That is the half of Bolna's design
that two earlier readings of `sync_history` missed, because it is not in `sync_history`:
it is `_stage_assistant_history` and its pair, forty lines away and much simpler than the
mechanism they exist to make rare.

Reconstruction is the fallback for the one turn that was half spoken, and it is
proportional: the fraction of the reply's audio that had played, applied to its
characters, trimmed back to a word boundary. Characters proportional to time is a crude
model of speech and it is obviously better than the two alternatives, which are keeping
the whole sentence and dropping it entirely.

**Five guards, each of which is a bug we would otherwise ship**, three read off Bolna and
two that fall out of our own shape:

- Refuse to trim without evidence. A cleanup that finds nothing queued must leave the
  record alone rather than truncate the last committed message, or a second interruption
  with nothing pending silently deletes a reply that was fully heard.
- Never trim an already-committed turn twice. Idempotence is the point: cleanups run more
  than once and the second one has less information than the first.
- Only answer audio counts. Filler is audio the listener heard and is not part of the
  reply, so counting it shifts every offset after it, and it plays *before* the answer,
  which would push the estimate past the end.
- A turn with no audio at all is dropped, not committed empty. An empty assistant message
  is not the same as no message, and models treat it as a refusal.
- A question interrupted before any reply was staged is merged with whatever the caller
  says next, not recorded as a second question. The record alternates roles by
  construction once a turn is answered, so two user messages in a row can only mean the
  first was not, and the model needs the two halves as one thought, not two questions in
  a row it never answered the first of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from llm import ChatMessage
from vaani.turn_taking import whole_words

logger = structlog.get_logger(__name__)


@dataclass
class Staged:
    """One assistant turn in progress: its text, and the audio time queued for it."""

    text: str = ""
    audio_s: float = 0.0

    def add(self, sentence: str) -> None:
        self.text = f"{self.text} {sentence}".strip() if self.text else sentence.strip()


@dataclass
class Conversation:
    """One session's exchanges, with assistant turns admitted only once they were heard.

    Within a session only. SPEC non-goal 8 excludes memory across sessions, so this is
    built and discarded with the socket and there is nowhere for it to be persisted to.
    """

    # How many exchanges the model is shown. Bounded because the prompt grows with the
    # conversation and this runs on a free tier, and because an eligibility question two
    # turns back is context while one from twenty turns back is mostly cost.
    max_turns: int = 8

    messages: list[ChatMessage] = field(default_factory=list)
    _staged: dict[int, Staged] = field(default_factory=dict, init=False)
    _committed: set[int] = field(default_factory=set, init=False)

    def user_said(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.messages and self.messages[-1].role == "user":
            previous = (self.messages[-1].content or "").strip()
            # A transcript that extends the previous one is the same utterance, not a
            # new one. Ours does this by construction: a reused partial and the final
            # that follows it share a prefix, and two user messages where the second
            # contains the first reads to the model as the question being asked twice.
            if text.startswith(previous):
                logger.info("history.collapsed_overlapping_user")
                self.messages[-1] = ChatMessage(role="user", content=text)
                return
            # Two user messages in a row with no overlap can only mean the first one
            # never got a reply: every path that answers a turn, whole or truncated,
            # appends an assistant message right after it, so the record alternates by
            # construction unless that never happened. That is a turn interrupted
            # before the model staged anything, not a second question, so the second
            # transcript is the rest of the first one rather than a fresh message.
            # Bolna's `pop_and_merge_user`; here the record has nothing to pop, only
            # to replace with the two halves joined.
            logger.info("history.merged_unanswered_user")
            self.messages[-1] = ChatMessage(role="user", content=f"{previous} {text}")
            return
        self.messages.append(ChatMessage(role="user", content=text))
        self._trim()

    def stage(self, generation: int, sentence: str) -> None:
        """A sentence the model produced, not yet known to have been heard."""
        if not sentence.strip():
            return
        self._staged.setdefault(generation, Staged()).add(sentence)

    def note_audio(self, generation: int, duration_s: float) -> None:
        """Answer audio queued for this turn. Filler is not passed here, deliberately."""
        if duration_s <= 0 or generation not in self._staged:
            return
        self._staged[generation].audio_s += duration_s

    def commit(self, generation: int) -> None:
        """The turn's audio ran to the end, so all of it was heard."""
        staged = self._staged.pop(generation, None)
        if staged is None or generation in self._committed:
            return
        if not staged.text:
            return
        self._committed.add(generation)
        self.messages.append(ChatMessage(role="assistant", content=staged.text))
        self._trim()
        logger.info("history.committed", generation=generation, chars=len(staged.text))

    def truncate(self, generation: int, remaining_ms: float) -> None:
        """The turn was cut off, so commit the part whose audio had played.

        Given what is left to play rather than what has played, because that is what the
        transport knows: the playout estimate says when queued audio ends, and audio is
        handed over far faster than real time so the send times say nothing.

        The answer is queued last, after any filler, so whatever is still to play is
        drawn from the end of it. If more is outstanding than the answer's own length,
        none of the answer was heard and the filler was still going.
        """
        staged = self._staged.pop(generation, None)
        if staged is None or generation in self._committed:
            # No evidence, or already committed. Both mean leave the record alone: a
            # second cleanup knows less than the first and must not act on the
            # difference. This is the guard whose absence deletes a fully heard reply.
            logger.info("history.trim_refused", generation=generation)
            return

        total_ms = staged.audio_s * 1000
        if total_ms <= 0 or not staged.text:
            # Nothing was ever queued for this turn, so nothing was heard. Dropped rather
            # than committed empty: an empty assistant message is not the same as no
            # message, and a model reads it as a refusal.
            logger.info("history.dropped_unheard", generation=generation)
            return

        played_ms = max(0.0, total_ms - max(0.0, remaining_ms))
        share = min(1.0, played_ms / total_ms)
        heard = _prefix(staged.text, share)
        if not heard:
            logger.info("history.dropped_unheard", generation=generation)
            return

        self._committed.add(generation)
        self.messages.append(ChatMessage(role="assistant", content=heard))
        self._trim()
        logger.info(
            "history.truncated",
            generation=generation,
            share=round(share, 2),
            kept_chars=len(heard),
            of_chars=len(staged.text),
        )

    def drop(self, generation: int, reason: str) -> None:
        """This turn's text never reached the listener at all."""
        if self._staged.pop(generation, None) is not None:
            logger.info("history.dropped", generation=generation, reason=reason)

    def for_model(self) -> list[ChatMessage]:
        """A copy, because the turn appends its own tool round trips to what it is given."""
        return [message.model_copy(deep=True) for message in self.messages]

    def _trim(self) -> None:
        keep = self.max_turns * 2
        if len(self.messages) > keep:
            del self.messages[: len(self.messages) - keep]


def _prefix(text: str, share: float) -> str:
    """The share of this text that was spoken, cut back to a whole word.

    A transcript cut mid-word puts a fragment into the record, and a fragment in Hinglish
    is frequently a different word.
    """
    if share >= 1.0:
        return text
    cut = int(len(text) * share)
    if cut <= 0:
        return ""
    return whole_words(text[:cut])
