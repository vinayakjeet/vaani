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

import structlog

from llm import ChatClient, ChatMessage
from llm.types import StreamCompleted, TextChunk, ToolCall, ToolCallsRequested
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

    async def run(
        self, question: str, still_current: Callable[[], bool] | None = None
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
            async for chunk in self._rounds(question, stage, still_current):
                yield chunk

    async def _rounds(
        self, question: str, stage, still_current: Callable[[], bool] | None
    ) -> AsyncIterator[str]:
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=question),
        ]
        started = time.monotonic()
        first_token = False

        for round_index in range(self._max_tool_rounds + 1):
            requested: list[ToolCall] = []

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
                        yield event.text
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
            messages.extend(_result_of(call) for call in requested)


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
