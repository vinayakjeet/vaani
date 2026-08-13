"""The conversation-quality numbers, and the two that carry a bar.

Most of these are ratios this project reports and does not grade. Two are different and
the tests say which: barge-in recovery rate has an external benchmark, and
`agent_interrupted_user` counts a failure the latency numbers score as a success.
"""

from __future__ import annotations

import time

from vaani.quality import (
    RECOVERY_RATE_CRITICAL,
    RECOVERY_RATE_GOOD,
    Interactivity,
    Interruption,
)


def test_a_session_with_no_barge_ins_has_no_recovery_rate() -> None:
    """None, not zero, and the difference is the whole point. Charting a quiet
    conversation at 0% puts it below the critical bar for having gone well."""
    assert Interactivity().recovery_rate is None


def test_recovery_rate_counts_full_replies_after_a_barge_in() -> None:
    quality = Interactivity()

    quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)
    quality.response_delivered()
    quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)

    assert quality.recovery_rate == 50.0
    assert quality.recovery_rate < RECOVERY_RATE_CRITICAL


def test_a_recovered_session_clears_the_benchmark() -> None:
    quality = Interactivity()
    for _ in range(10):
        quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)
        quality.response_delivered()

    assert quality.recovery_rate >= RECOVERY_RATE_GOOD


def test_the_agent_interrupting_the_user_is_not_in_the_recovery_denominator() -> None:
    """The rate is about turns the user cut off. Folding in the other direction would
    make the denominator two different events and the benchmark meaningless."""
    quality = Interactivity()
    quality.interrupted(Interruption.AGENT_INTERRUPTED_USER)

    assert quality.recovery_rate is None
    assert quality.agent_interrupted_user == 1
    assert quality.total_interruptions == 1


def test_a_reply_delivered_with_no_interruption_is_not_a_recovery() -> None:
    quality = Interactivity()
    quality.response_delivered()

    assert quality.recoveries == 0


def test_the_longest_monologue_is_the_longest_and_not_the_last() -> None:
    quality = Interactivity()

    quality.agent_started_speaking()
    quality._agent_started = time.monotonic() - 3.0
    quality.agent_stopped_speaking()

    quality.agent_started_speaking()
    quality._agent_started = time.monotonic() - 0.5
    quality.agent_stopped_speaking()

    assert quality.longest_agent_monologue_ms >= 3000
    assert quality.agent_speaking_ms >= 3500


def test_talk_to_listen_is_the_agents_share_of_speaking_time() -> None:
    quality = Interactivity()

    quality.agent_started_speaking()
    quality._agent_started = time.monotonic() - 3.0
    quality.agent_stopped_speaking()

    quality.user_started_speaking()
    quality._user_started = time.monotonic() - 1.0
    quality.user_stopped_speaking()

    assert 70.0 <= quality.talk_to_listen <= 80.0


def test_synthesis_slower_than_playback_is_visible_as_stuttering() -> None:
    """The failure no latency number in this project can see: time to first audio is
    excellent in exactly the run where the rest of the reply arrives in pieces."""
    quality = Interactivity()

    quality.note_answer_audio(0.5)
    quality._answer_first_chunk = time.monotonic() - 2.0
    quality.note_answer_audio(0.5)

    assert quality.tts_speed_ratio < 1.0
    assert quality.stuttering


def test_synthesis_ahead_of_playback_is_not_stuttering() -> None:
    quality = Interactivity()
    quality.note_answer_audio(1.0)
    quality.note_answer_audio(1.0)

    # Two seconds of audio handed over in microseconds, which is what a healthy stream
    # looks like: it runs far ahead of real time.
    assert quality.tts_speed_ratio > 1.0
    assert not quality.stuttering


def test_the_speed_ratio_starts_at_the_first_answer_chunk() -> None:
    """Measuring from the turn's start would fold time to first audio into a ratio that
    is about a different failure, and a slow first token would read as a stutter."""
    quality = Interactivity()
    quality.note_answer_audio(1.0)

    assert quality.answer_wall_clock_s == 0.0
    assert quality.tts_speed_ratio is None


def test_a_backchannel_is_counted_rather_than_only_ignored() -> None:
    """The other reading of a low interruption count is that the detector never fires,
    and the two are identical from the latency numbers."""
    quality = Interactivity()
    quality.backchannel_ignored()

    assert quality.summary()["backchannels_ignored"] == 1


def test_the_summary_carries_no_text() -> None:
    """Same rule as the span contract: counts and durations only, nothing that can be
    read back into what somebody said about their income."""
    quality = Interactivity()
    quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)

    assert all(
        value is None or isinstance(value, int | float)
        for value in quality.summary().values()
    )


def test_where_an_interruption_landed_is_recorded_as_a_position() -> None:
    """A count of interruptions says the agent talks too much. A distribution of
    positions says which sentence it should stop saying, which is the only thing a
    barge-in produces that nothing else can."""
    quality = Interactivity()
    quality.turn_started()
    quality.sentence_spoken()
    quality.note_answer_audio(1.5)
    quality.sentence_spoken()
    quality.note_answer_audio(0.5)

    quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)

    assert quality.interrupted_at == [{"sentence": 2, "answer_ms": 2000}]


def test_the_position_is_per_turn_and_not_cumulative() -> None:
    """A cumulative figure would say every later interruption happened deeper into the
    reply than the first one, whatever actually happened."""
    quality = Interactivity()

    quality.turn_started()
    quality.note_answer_audio(4.0)
    quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)

    quality.turn_started()
    quality.note_answer_audio(0.5)
    quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)

    assert [point["answer_ms"] for point in quality.interrupted_at] == [4000, 500]


def test_being_cut_off_before_saying_anything_is_its_own_position() -> None:
    quality = Interactivity()
    quality.turn_started()

    quality.interrupted(Interruption.USER_INTERRUPTED_AGENT)

    assert quality.interrupted_at == [{"sentence": 0, "answer_ms": 0}]
