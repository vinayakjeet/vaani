from __future__ import annotations

import itertools

import pytest

from vaani.state import TRANSITIONS, IllegalTransition, State, TurnState


def at(state: State) -> TurnState:
    """A machine parked in one state, reached without going through `to`.

    Constructed directly rather than driven there, because a test that has to
    walk a legal path to reach a state cannot then assert which paths are legal.
    """
    return TurnState(state=state)


@pytest.mark.parametrize(("start", "requested"), list(itertools.product(State, State)))
def test_every_pair_of_states_matches_the_table(start: State, requested: State) -> None:
    """Exhaustive over all sixteen pairs, in both directions.

    Asserting only the legal moves would pass against a machine that permits
    everything, which is the failure mode that matters: an illegal transition
    that is quietly accepted is two audio streams into the same ear.
    """
    machine = at(start)
    legal = requested in TRANSITIONS[start]

    assert machine.can(requested) is legal

    if legal:
        machine.to(requested)
        assert machine.state is requested
    else:
        with pytest.raises(IllegalTransition):
            machine.to(requested)
        assert machine.state is start


@pytest.mark.parametrize("state", list(State))
def test_no_state_transitions_to_itself(state: State) -> None:
    """Re-entering the current state is always a bug, never a no-op. "Start
    speaking" while already speaking is the babbling agent."""
    assert not at(state).can(state)


LEGAL = [
    (State.IDLE, State.LISTENING),
    (State.LISTENING, State.THINKING),
    (State.THINKING, State.SPEAKING),
    (State.THINKING, State.LISTENING),
    (State.SPEAKING, State.LISTENING),
    (State.LISTENING, State.IDLE),
    (State.THINKING, State.IDLE),
    (State.SPEAKING, State.IDLE),
]

ILLEGAL = [
    (State.IDLE, State.THINKING),
    (State.IDLE, State.SPEAKING),
    (State.LISTENING, State.SPEAKING),
    (State.SPEAKING, State.THINKING),
]


@pytest.mark.parametrize(("start", "requested"), LEGAL)
def test_the_transitions_the_pipeline_needs_are_legal(start: State, requested: State) -> None:
    """Written out rather than read from `TRANSITIONS`.

    The exhaustive test above uses the table as its oracle, so it proves `can` and
    `to` agree with the table and says nothing about whether the table is right.
    Deleting a row would leave it green. These two lists are the table's second
    opinion, and the pair of barge-in rows is why: THINKING and SPEAKING both
    return to LISTENING, and losing either one silently disables interrupting.
    """
    assert at(start).can(requested)


@pytest.mark.parametrize(("start", "requested"), ILLEGAL)
def test_the_transitions_that_would_be_bugs_are_illegal(start: State, requested: State) -> None:
    """LISTENING to SPEAKING skips generating an answer, and SPEAKING to THINKING
    would let a second answer start while the first is still in the ear."""
    assert not at(start).can(requested)


def test_the_two_lists_cover_every_pair() -> None:
    """Otherwise a pair could be dropped from both lists and lose its coverage
    without any test going red."""
    assert set(LEGAL) | set(ILLEGAL) == set(itertools.product(State, State)) - {
        (state, state) for state in State
    }


def test_the_table_covers_every_state() -> None:
    """Guards every test above. `TRANSITIONS[self.state]` would raise KeyError on
    a state nobody added a row for, and adding a state is exactly the change that
    forgets one."""
    assert set(TRANSITIONS) == set(State)


def test_an_illegal_transition_names_both_states() -> None:
    with pytest.raises(IllegalTransition) as raised:
        at(State.IDLE).to(State.SPEAKING)

    assert raised.value.current is State.IDLE
    assert raised.value.requested is State.SPEAKING


def test_a_first_turn_starts_listening_and_is_generation_one() -> None:
    machine = TurnState()

    assert machine.begin() == 1
    assert machine.state is State.LISTENING
    assert not machine.interrupted_previous


def test_a_normal_next_turn_does_not_count_as_an_interruption() -> None:
    machine = TurnState()
    machine.begin()
    machine.to(State.THINKING)
    machine.to(State.SPEAKING)
    machine.to(State.LISTENING)

    assert machine.begin() == 2
    assert not machine.interrupted_previous


@pytest.mark.parametrize("interrupted_during", [State.THINKING, State.SPEAKING])
def test_barging_in_records_the_displaced_turn_as_interrupted(interrupted_during: State) -> None:
    """SPEC S4: the conversation state reflects the interrupted turn as
    interrupted. A user can cut in while the model is generating as well as while
    it is speaking, and only the second one is audible, so only the second one
    tends to get tested."""
    machine = TurnState()
    machine.begin()
    machine.to(State.THINKING)
    if interrupted_during is State.SPEAKING:
        machine.to(State.SPEAKING)

    assert machine.begin() == 2
    assert machine.interrupted_previous
    assert machine.state is State.LISTENING


def test_the_flag_clears_on_the_turn_after_an_interruption() -> None:
    """The reason the flag is named for the turn it describes.

    Turn two is born from a barge-in; turn three is not. A flag that stayed true
    would label turn three interrupted, and `vaani.turn.interrupted` would be
    wrong on every span after the first barge-in of a session.
    """
    machine = TurnState()
    machine.begin()
    machine.to(State.THINKING)
    machine.begin()
    assert machine.interrupted_previous

    machine.to(State.THINKING)
    machine.to(State.SPEAKING)
    machine.to(State.LISTENING)
    machine.begin()

    assert not machine.interrupted_previous


def test_beginning_a_turn_while_already_listening_still_advances_the_generation() -> None:
    """LISTENING to LISTENING is not a legal transition, so `begin` has to skip
    the move rather than make it. The generation still advances, because a second
    utterance is a second turn whether or not the state changed."""
    machine = TurnState()
    machine.begin()

    assert machine.begin() == 2
    assert machine.state is State.LISTENING


def test_work_from_a_displaced_generation_is_disowned() -> None:
    """The barge-in race, from the state machine's side. A tool result or an
    audio chunk from the abandoned turn arriving late is an answer to a question
    nobody asked."""
    machine = TurnState()
    stale = machine.begin()
    machine.to(State.THINKING)
    current = machine.begin()

    assert machine.owns(current)
    assert not machine.owns(stale)


def test_generations_are_never_reused() -> None:
    machine = TurnState()
    seen = {machine.begin() for _ in range(20)}

    assert len(seen) == 20
