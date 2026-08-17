"""One turn, streamed, with the tool round trip inside it.

The unstreamed baseline in `vaani/turn.py` stays exactly as it is. This is the
overlapped path built beside it, because `bench/ablation.py` runs both on demand
and "streaming bought 400ms" means nothing without a measured before.

What this yields is text, not sentences. Segmentation belongs to
`vaani.sentences`, and keeping them apart is what lets the sentence rules be
tested against a string rather than against a live model.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing

import structlog

from llm import ChatClient, ChatMessage
from llm.types import (
    ProviderClientError,
    StreamCompleted,
    TextChunk,
    ToolCall,
    ToolCallsRequested,
)
from vaani.grounding import sourced
from vaani.spans import LLM_GENERATE, stage_span
from vaani.tools import ToolError, dispatch, tool_schemas
from vaani.turn import SYSTEM_PROMPT

logger = structlog.get_logger(__name__)

# How many times the model may go round the tool loop before the turn gives up.
# Bounded because an unbounded loop is the failure Spanlight's loop detector was
# built to catch, and here it would be audible as an agent that goes quiet and
# then bills for it. Two rounds covers find_schemes followed by check_eligibility,
# which is the deepest sequence the contract allows.
MAX_TOOL_ROUNDS = 2

COULD_NOT_CHECK = (
    "Main aapki eligibility check nahi kar paaya. Kripya thodi der baad koshish kijiye."
)

# The tag Groq's Llama models write when they put a tool call in prose instead of a
# structured `tool_calls` delta. Seen live: "...eligibility check karne ke liye,
# <function=check_eligibility>{"applicant":{"annual_income_inr":"50000",...}}</function>
# ka use karna hoga." The surrounding Hindi is a real reply and stays; the tag and
# everything inside it is a defect in the model's own streaming format and must never
# reach a synthesiser, both because it is unpronounceable and because it can carry an
# applicant's own figures inside it.
LEAK_OPEN = "<function="
LEAK_CLOSE = "</function>"


class ToolCallLeakFilter:
    """Strips a tool call the model wrote as prose out of the text stream around it.

    Feed it in the order the model wrote it, get back only the words a listener should
    hear. Its whole reason to exist is that `LEAK_OPEN` arrives a few characters per
    chunk, not as one string: Groq streams text a handful of characters at a time, so
    a check against each chunk in isolation never sees the marker.

    A small suffix of confirmed-safe text is always held back rather than yielded
    immediately, exactly as many characters as `LEAK_OPEN` needs to rule itself out.
    Releasing text the moment it arrives would mean releasing the first half of the
    marker before the second half proves what it was.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._inside_leak = False

    def feed(self, piece: str) -> str:
        self._pending += piece
        safe = ""

        while True:
            if self._inside_leak:
                end = self._pending.find(LEAK_CLOSE)
                if end == -1:
                    return safe
                # The leak and its payload, gone. What follows the tag in the same
                # chunk is ordinary text again, and the loop re-examines it below
                # rather than assuming nothing else is in the buffer.
                self._pending = self._pending[end + len(LEAK_CLOSE) :]
                self._inside_leak = False
                continue

            start = self._pending.find(LEAK_OPEN)
            if start != -1:
                safe += self._pending[:start]
                self._pending = self._pending[start:]
                self._inside_leak = True
                continue

            # No marker in the buffer. Held back is only the tail that could still
            # become one: a chunk boundary landing mid-marker must not have already
            # released the prefix that would prove it.
            hold = len(LEAK_OPEN) - 1
            if len(self._pending) > hold:
                cut = len(self._pending) - hold
                safe += self._pending[:cut]
                self._pending = self._pending[cut:]
            return safe

    def flush(self) -> str:
        """Whatever the round ends holding. Called once the stream is known to be done.

        Text held back because it merely resembled the marker's prefix is released,
        since the round is over and it never became one. Text confirmed to be inside
        an unclosed leak is dropped rather than spoken: the tag never closed, so the
        payload is truncated, and a truncated JSON fragment read aloud is worse than
        the words it interrupted going missing.
        """
        if self._inside_leak:
            logger.warning("turn.tool_call_leak_unclosed", chars=len(self._pending))
            self._inside_leak = False
            self._pending = ""
            return ""
        text, self._pending = self._pending, ""
        return text


class StreamedTurn:
    """A question in, a stream of reply text out, tools resolved on the way."""

    def __init__(
        self,
        llm: ChatClient | None = None,
        provider: str = "groq",
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self._llm = llm or ChatClient()
        self._provider = provider
        self._max_tool_rounds = max_tool_rounds
        # Figures the tools returned during this turn, for the validator that runs
        # between here and the synthesiser. Populated as the rounds complete, which is
        # before any sentence containing a number can have been flushed: a reply cannot
        # quote a threshold it has not asked for yet.
        self.sourced: set[float] = set()

    async def run(
        self,
        question: str,
        still_current: Callable[[], bool] | None = None,
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the reply, abandoning the turn if it stops being the current one.

        `still_current` is asked before a tool result is applied. SPEC S4: a tool
        result from an interrupted turn must be discarded rather than folded into
        the next one, because an answer built on a question nobody asked is worse
        than no answer, and it arrives sounding just as confident.
        """
        # One span for the turn's generation, not one per round, because the
        # listener waited for all of them. Time to first token is an event on it
        # rather than the duration, which is how long the model talked for.
        with stage_span(LLM_GENERATE, **{"gen_ai.system": self._provider}) as stage:
            # `aclosing`: this generator is itself abandonable (`vaani.sentences`
            # closes it explicitly rather than trust exhaustion), and being idle
            # at `yield chunk` when that happens would otherwise leave `_rounds`
            # suspended mid-round with its provider stream still open.
            rounds = self._rounds(question, stage, still_current, history)
            async with aclosing(rounds):
                async for chunk in rounds:
                    yield chunk

    async def _rounds(
        self,
        question: str,
        stage,
        still_current: Callable[[], bool] | None,
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[str]:
        # The static prompt stays first and the volatile state follows it, which is the
        # ordering that makes the prefix cacheable. Groq caches automatically on supported
        # models, so putting history in front of the system prompt would silently give up
        # about half the input cost and a measurable slice of time to first token.
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            *(history or []),
            ChatMessage(role="user", content=question),
        ]
        started = time.monotonic()
        first_token = False
        self.sourced = set()

        for round_index in range(self._max_tool_rounds + 1):
            requested: list[ToolCall] = []
            # Fresh per round: a leak, if the model writes one, is confined to the
            # text this round produced, and each round is an independent sample from
            # the model regardless of what a previous round did.
            leak_filter = ToolCallLeakFilter()

            try:
                async for event in self._llm.stream(
                    self._provider, messages, tools=tool_schemas()
                ):
                    match event:
                        case TextChunk():
                            if not first_token:
                                first_token = True
                                stage.record(
                                    **{
                                        "vaani.llm.first_token_ms": (
                                            time.monotonic() - started
                                        )
                                        * 1000
                                    }
                                )
                            if safe := leak_filter.feed(event.text):
                                yield safe
                        case ToolCallsRequested():
                            requested = event.calls
                        case StreamCompleted():
                            # Length, never the text. The reply is about a real
                            # person's eligibility, so it stays out of logs entirely.
                            logger.info(
                                "turn.round",
                                round=round_index,
                                finish_reason=event.finish_reason,
                                tool_calls=len(requested),
                            )
            except ProviderClientError as exc:
                # Groq can fail to turn the model's own generation into a tool call at
                # all: `tool_use_failed` with "Failed to call a function", raised before
                # `ToolCallsRequested` ever fires. That is not our schema rejecting the
                # arguments, which the model can read and correct within its remaining
                # rounds; it is the provider never producing a structured call to
                # correct, so there is no `tool_call_id` to attach a result to and
                # nothing to feed back.
                #
                # Left uncaught this propagates out of the whole turn, past every
                # degradation rule SPEC has, to the session's generic playback handler,
                # which is the exact "filler, then silence" a listener hears from a
                # cause that has nothing to do with a dropped connection. Ending the
                # turn here, the same way running out of rounds already does, is a
                # known and tested failure path rather than a new one.
                logger.warning(
                    "turn.tool_call_rejected", round=round_index, detail=str(exc)
                )
                stage.record(**{"vaani.llm.rounds": round_index + 1})
                yield COULD_NOT_CHECK
                return

            # Whatever the round's stream ended holding: text kept back on the chance
            # it was the start of a leaked tag, released now that the round is over
            # and it never became one.
            if trailing := leak_filter.flush():
                yield trailing

            if not requested:
                stage.record(**{"vaani.llm.rounds": round_index + 1})
                return

            if round_index == self._max_tool_rounds:
                # Out of rounds with a tool still pending. Saying so is the only
                # honest option: silence reads as a hang, and answering anyway
                # would be a confident reply built on a check that never ran.
                logger.warning("turn.tool_rounds_exhausted", rounds=round_index + 1)
                stage.record(**{"vaani.llm.rounds": round_index + 1})
                yield COULD_NOT_CHECK
                return

            if still_current is not None and not still_current():
                # Interrupted while the tools were in flight. The results are
                # dropped rather than applied, and nothing further is spoken: the
                # new turn is already listening.
                logger.info("turn.abandoned", round=round_index)
                stage.record(**{"vaani.llm.rounds": round_index + 1})
                return

            messages.append(_assistant_asking_for(requested))
            results = [_result_of(call) for call in requested]
            messages.extend(results)
            # What the tools actually said, as figures. Everything the reply is then
            # allowed to state, along with what the user said themselves.
            self.sourced |= sourced(*[message.content for message in results])


def _assistant_asking_for(calls: list[ToolCall]) -> ChatMessage:
    """The model's own tool request, echoed back so the results have an antecedent.

    Providers reject a `tool` message whose `tool_call_id` does not match a call
    in the preceding assistant message, so this is not bookkeeping, it is what
    makes the second round accepted at all.
    """
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls
        ],
    )


def _result_of(call: ToolCall) -> ChatMessage:
    """Run one tool and phrase whatever happened as something the model can use.

    A failure comes back as a result rather than an exception, because the model
    is the only thing that can turn "that scheme id does not exist" into a
    sentence a listener understands. What it must never do is answer as though the
    check succeeded, which is why the failure is stated rather than omitted.
    """
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        # Models truncate JSON under a token limit often enough that this is a
        # normal branch, not a defensive one.
        return _failed(call, "the arguments were not valid JSON")

    if not isinstance(arguments, dict):
        return _failed(call, "the arguments were not an object")

    try:
        result = dispatch(call.name, arguments)
    except ToolError as exc:
        # `ToolError` messages name the field and the rule and never the value,
        # which is what makes them safe to hand back to a model that will say
        # them out loud.
        return _failed(call, str(exc))

    return ChatMessage(
        role="tool",
        tool_call_id=call.id,
        content=json.dumps(result, ensure_ascii=False),
    )


def _failed(call: ToolCall, reason: str) -> ChatMessage:
    logger.warning("turn.tool_failed", tool=call.name, reason=reason)
    return ChatMessage(
        role="tool",
        tool_call_id=call.id,
        content=json.dumps({"error": reason, "completed": False}),
    )
