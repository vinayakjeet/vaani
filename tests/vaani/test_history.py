"""The conversation record, and the five guards that stop it lying about what was said.

Every one of these is a bug that ships if the record is built first and truncation is
added later, which is why it was built with them.
"""

from __future__ import annotations

from vaani.history import Conversation


def content(record: Conversation) -> list[tuple[str, str]]:
    return [(m.role, m.content or "") for m in record.messages]


def test_a_reply_whose_audio_never_went_out_is_not_in_the_record() -> None:
    """The ordinary interrupted turn, handled by not writing rather than by repairing.
    This is the half of the design that two readings of sync_history missed."""
    record = Conversation()
    record.user_said("kya main eligible hoon")
    record.stage(1, "Haan, aap eligible hain.")

    record.drop(1, "blocked")

    assert content(record) == [("user", "kya main eligible hoon")]


def test_a_reply_that_played_to_the_end_is_committed_whole() -> None:
    record = Conversation()
    record.user_said("kya main eligible hoon")
    record.stage(1, "Haan, aap eligible hain.")
    record.note_audio(1, 2.0)

    record.commit(1)

    assert content(record)[-1] == ("assistant", "Haan, aap eligible hain.")


def test_a_half_played_reply_is_truncated_to_what_was_heard() -> None:
    """Characters proportional to time is a crude model of speech, and it is obviously
    better than the two alternatives, which are keeping all of it and dropping all."""
    record = Conversation()
    record.stage(1, "Aapki aay seema se kam hai aur aap eligible hain.")
    record.note_audio(1, 4.0)

    # Two of the four seconds are still queued, so half was heard.
    record.truncate(1, remaining_ms=2000)

    heard = content(record)[-1][1]
    assert heard
    assert "Aapki aay seema" in heard
    assert "eligible" not in heard


def test_the_kept_text_ends_on_a_whole_word() -> None:
    """A fragment in the record is read back by the model as a word, and a fragment in
    Hinglish is frequently a different word."""
    record = Conversation()
    record.stage(1, "Aapko chhe hazaar rupaye milenge")
    record.note_audio(1, 10.0)

    record.truncate(1, remaining_ms=6900)

    assert not content(record)[-1][1].endswith(("cha", "haz", "ru"))
    assert " " not in content(record)[-1][1][-1:]


def test_a_turn_interrupted_before_any_audio_is_dropped_rather_than_committed_empty() -> None:
    """An empty assistant message is not the same as no message: a model reads it as a
    refusal and answers the next question as though it had already declined."""
    record = Conversation()
    record.user_said("kya main eligible hoon")
    record.stage(1, "Haan, aap eligible hain.")

    record.truncate(1, remaining_ms=0)

    assert content(record) == [("user", "kya main eligible hoon")]


def test_a_second_cleanup_cannot_delete_an_already_committed_reply() -> None:
    """Idempotence is the point. Cleanups run more than once and the second one knows
    less than the first, so it must not act on the difference. Without this guard, a
    later interruption with nothing pending silently deletes a fully heard reply."""
    record = Conversation()
    record.stage(1, "Haan, aap eligible hain.")
    record.note_audio(1, 2.0)
    record.commit(1)

    record.truncate(1, remaining_ms=2000)

    assert content(record) == [("assistant", "Haan, aap eligible hain.")]


def test_a_truncation_with_no_evidence_leaves_the_record_alone() -> None:
    """Nothing was staged for this generation, so there is nothing to say about it, and
    the last thing to do is trim whatever happens to be last."""
    record = Conversation()
    record.stage(1, "Pehla jawab.")
    record.note_audio(1, 1.0)
    record.commit(1)

    record.truncate(2, remaining_ms=500)

    assert content(record) == [("assistant", "Pehla jawab.")]


def test_only_answer_audio_moves_the_estimate() -> None:
    """Filler is audio the listener heard and is not part of the reply. It also plays
    before the answer, so counting it would push the estimate past the end and commit a
    reply that was never spoken."""
    record = Conversation()
    record.stage(1, "Ek do teen chaar")
    # Two seconds of answer. A filler that played first is never passed here.
    record.note_audio(1, 2.0)

    record.truncate(1, remaining_ms=2000)

    assert content(record) == []


def test_a_final_transcript_that_extends_the_partial_replaces_it() -> None:
    """Ours does this by construction: the reused partial and the final that follows it
    share a prefix, and two user messages where the second contains the first reads to
    the model as the question being asked twice."""
    record = Conversation()
    record.user_said("meri aay")
    record.user_said("meri aay 300000 hai")

    assert content(record) == [("user", "meri aay 300000 hai")]


def test_a_genuinely_new_question_after_a_real_answer_is_not_merged() -> None:
    """The record alternates roles once a turn is actually answered, so a fresh
    question that follows a real reply is a fresh question, not a continuation.
    Committing the first turn between the two calls is the part the old version of
    this test skipped, and skipping it made the fixture indistinguishable from the
    interrupted-turn case the merge exists to catch."""
    record = Conversation()
    record.user_said("mujhe ghar chahiye")
    record.stage(1, "PM Awas Yojana dekhiye.")
    record.commit(1)
    record.user_said("aur pension bhi")

    assert content(record) == [
        ("user", "mujhe ghar chahiye"),
        ("assistant", "PM Awas Yojana dekhiye."),
        ("user", "aur pension bhi"),
    ]


def test_a_question_interrupted_before_any_reply_is_merged_with_what_follows() -> None:
    """Bolna's `pop_and_merge_user`. Nothing was ever staged for this turn, so the
    record has only the bare question when the next one arrives, and two user
    messages in a row can only mean the first was never answered: not a second
    question, the rest of the first one."""
    record = Conversation()
    record.user_said("mera annual income pachaas hazar hai aur")
    record.user_said("do acre zameen hai")

    assert content(record) == [
        ("user", "mera annual income pachaas hazar hai aur do acre zameen hai")
    ]


def test_a_question_interrupted_after_a_dropped_reply_is_still_merged() -> None:
    """The same case reached the other way: a reply was staged but its audio never
    went out, so `truncate` dropped it rather than committing it (the guard two
    tests up from here). The record still ends on a bare user message, and the
    merge has to see through the drop rather than only through the empty case."""
    record = Conversation()
    record.user_said("mera annual income pachaas hazar hai aur")
    record.stage(1, "Aap eligible hain.")
    # No `note_audio` call: text was staged but no audio was ever queued for it, so
    # `truncate` drops it rather than committing anything, the same as if nothing had
    # been staged at all.
    record.truncate(1, remaining_ms=0)
    record.user_said("do acre zameen hai")

    assert content(record) == [
        ("user", "mera annual income pachaas hazar hai aur do acre zameen hai")
    ]


def test_the_record_is_bounded() -> None:
    """The prompt grows with the conversation and this runs on a free tier. A question
    two turns back is context; one from twenty turns back is mostly cost."""
    record = Conversation(max_turns=2)
    for index in range(6):
        record.user_said(f"sawaal {index}")
        record.stage(index, f"jawab {index}")
        record.note_audio(index, 1.0)
        record.commit(index)

    assert len(record.messages) == 4
    assert content(record)[-1] == ("assistant", "jawab 5")


def test_what_the_model_is_given_is_a_copy() -> None:
    """The turn appends its own tool round trips to what it is given, and those must not
    land in the record: a tool result is not something the agent said out loud."""
    record = Conversation()
    record.user_said("kya main eligible hoon")

    given = record.for_model()
    given.append(given[0])

    assert len(record.messages) == 1
