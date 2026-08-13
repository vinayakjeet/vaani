import pytest

from vaani.grounding import (
    REFUSED,
    check,
    figures,
    refusals,
    sourced,
    ungrounded,
)
from vaani.tools import check_eligibility


@pytest.fixture(autouse=True)
def _clear_counter():
    refusals.clear()


def test_the_same_amount_written_three_ways_is_one_figure():
    # A comparison on the written form would let a model launder an invented number
    # through a different notation.
    assert figures("3 lakh") == figures("300000") == figures("3,00,000") == {300000.0}


def test_list_markers_and_small_counts_are_not_checked():
    assert figures("2. Form bharein aur do documents laayein") == set()


def test_a_decimal_is_checked_whatever_its_size():
    assert figures("aapke paas 2.5 acre zameen hai") == {2.5}


def test_a_threshold_inside_a_tool_reason_string_is_a_source():
    result = check_eligibility(
        {
            "scheme_id": "pm-jay",
            "applicant": {"state": "up", "annual_income_inr": 400000},
        }
    ).model_dump()
    assert 500000.0 in sourced(result)


def test_a_reply_quoting_the_tool_threshold_is_allowed():
    allowed = sourced({"limit": 500000})
    reply = "Ayushman Bharat ki income limit 5 lakh hai, aap eligible hain."
    assert ungrounded(reply, allowed) == ()
    assert check(reply, allowed) == reply


def test_a_reply_inventing_a_threshold_is_refused_and_counted():
    allowed = sourced({"limit": 500000})
    reply = "Ayushman Bharat ki income limit 800000 hai."
    assert ungrounded(reply, allowed) == (800000.0,)
    assert check(reply, allowed) == REFUSED
    assert refusals["ungrounded_figure"] == 1


def test_the_applicants_own_figure_counts_as_a_source():
    # The user said it, so repeating it back is not an invention. This is what makes
    # a confirmation sentence possible at all.
    allowed = sourced({"limit": 500000}, "meri aay 4 lakh hai")
    reply = "Aapki aay 400000 hai aur limit 500000 hai."
    assert check(reply, allowed) == reply


def test_a_true_subtraction_is_still_refused():
    # Deliberate, and the cost is reported rather than argued away: allowing derived
    # arithmetic means allowing any number that happens to be derivable.
    allowed = sourced({"limit": 500000}, "meri aay 6 lakh hai")
    reply = "Aap limit se 100000 zyada kama rahe hain."
    assert ungrounded(reply, allowed) == (100000.0,)


def test_a_boolean_in_a_tool_result_is_not_a_figure():
    # bool subclasses int, so an unguarded walk turns eligible=True into the figure 1
    # and then a reply may quote any 1 it likes.
    assert sourced({"eligible": True, "indicative": True}) == set()


def test_a_reply_with_no_figures_at_all_passes():
    reply = "Aap eligible hain, form bhar dijiye."
    assert check(reply, set()) == reply


def test_the_refusal_replaces_the_reply_rather_than_following_it():
    # A refusal appended to the ungrounded sentence has already said the number it
    # exists to suppress.
    refused = check("Limit 999000 hai.", set())
    assert "999000" not in refused
    assert refused == REFUSED
