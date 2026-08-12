"""What a turn is doing, and what it is allowed to do next.

One small module with an explicit transition table, because the alternative is
four booleans spread across the socket handler and the pipeline, and every
barge-in race in this project will be a disagreement between two of them.

Illegal transitions raise rather than being ignored. A voice agent that silently
accepts "start speaking" while already speaking produces two overlapping audio
streams into the same ear, and the bug reports say the agent is babbling rather
than that a state check is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class IllegalTransition(RuntimeError):
    def __init__(self, current: State, requested: State) -> None:
        super().__init__(f"cannot go from {current} to {requested}")
        self.current = current
        self.requested = requested


# Every legal move, and nothing else is legal.
#
# THINKING and SPEAKING both go back to LISTENING because that is barge-in: the
# user speaking during either one abandons the turn and starts a new one. They
# do not go to IDLE, because the microphone is still open and the session is
# still alive; only an explicit stop or a disconnect ends the session.
TRANSITIONS: dict[State, frozenset[State]] = {
    State.IDLE: frozenset({State.LISTENING}),
    State.LISTENING: frozenset({State.THINKING, State.IDLE}),
    State.THINKING: frozenset({State.SPEAKING, State.LISTENING, State.IDLE}),
    State.SPEAKING: frozenset({State.LISTENING, State.IDLE}),
}


@dataclass
class TurnState:
    """What the current turn is doing, and the generation that identifies it.

    The generation increments on every new turn rather than on every state
    change. It is what lets audio already on the wire be discarded on arrival:
    stopping a producer does not unsend what it already sent, in either
    direction.
    """

    state: State = State.IDLE
    generation: int = 0
    interrupted_previous: bool = field(default=False, init=False)

    def can(self, requested: State) -> bool:
        return requested in TRANSITIONS[self.state]

    def to(self, requested: State) -> None:
        if not self.can(requested):
            raise IllegalTransition(self.state, requested)
        self.state = requested

    def begin(self) -> int:
        """Start a fresh turn from anywhere. Returns the new generation.

        Legal from LISTENING as a normal turn, and from THINKING or SPEAKING as
        a barge-in. The caller is responsible for cancelling the work the old
        generation left running; this only records that it is no longer current.

        The flag is named for the turn it describes. It reports whether the turn
        being displaced was cut off, which is what SPEC S4 asks to be recorded
        and what `vaani.turn.interrupted` carries on the closing span. An
        `interrupted` flag on this object would be read as a property of the turn
        now starting, and would then be true for the whole life of a turn that
        nobody interrupted.
        """
        self.interrupted_previous = self.state in (State.THINKING, State.SPEAKING)
        if self.state is not State.LISTENING:
            self.to(State.LISTENING)
        self.generation += 1
        return self.generation

    def owns(self, generation: int) -> bool:
        """Whether work tagged with this generation still belongs to the turn.

        Everything crossing a boundary is checked against this: audio chunks,
        tool results, and model tokens. A tool result from an abandoned turn
        applied to the current one is an answer built on a question nobody
        asked.
        """
        return generation == self.generation
