from __future__ import annotations

import json

import pytest

from vaani.tools import (
    RULES,
    Eligibility,
    ToolError,
    check_eligibility,
    dispatch,
    find_schemes,
    tool_schemas,
)


def applicant(**overrides: object) -> dict[str, object]:
    return {"state": "Bihar", "annual_income_inr": 100000, **overrides}


def test_within_every_threshold_is_eligible() -> None:
    result = check_eligibility(
        {"scheme_id": "pm-kisan", "applicant": applicant(land_holding_acres=2.0)}
    )

    assert result.eligible


def test_above_a_threshold_is_not_eligible() -> None:
    result = check_eligibility(
        {"scheme_id": "pm-kisan", "applicant": applicant(land_holding_acres=9.0)}
    )

    assert not result.eligible


def test_a_verdict_always_says_what_decided_it() -> None:
    """A spoken "not eligible" with no reason is what people come back about, and
    the reply cannot say what would change the answer if the tool did not."""
    result = check_eligibility({"scheme_id": "pmay-g", "applicant": applicant()})

    assert result.reasons
    assert all(reason.strip() for reason in result.reasons)


def test_every_result_is_marked_indicative() -> None:
    """The thresholds here are fixtures, not policy. A caller that forgets this
    tells a real person they qualify for a payment on the strength of a number
    written for a test."""
    result = check_eligibility({"scheme_id": "ujjwala", "applicant": applicant()})

    assert result.indicative


def test_exactly_on_the_threshold_is_within_it() -> None:
    """Off-by-one on an income limit is the difference between a family getting a
    cylinder and being turned away, so the boundary is pinned rather than left to
    whichever comparison operator got typed."""
    result = check_eligibility(
        {"scheme_id": "ujjwala", "applicant": applicant(annual_income_inr=200000)}
    )

    assert result.eligible

    just_over = check_eligibility(
        {"scheme_id": "ujjwala", "applicant": applicant(annual_income_inr=200001)}
    )

    assert not just_over.eligible


def test_an_unknown_scheme_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ToolError):
        check_eligibility({"scheme_id": "not-a-scheme", "applicant": applicant()})


@pytest.mark.parametrize(
    "arguments",
    [
        {"scheme_id": "pm-kisan"},
        {"scheme_id": "pm-kisan", "applicant": {"state": "Bihar"}},
        {"scheme_id": "pm-kisan", "applicant": applicant(annual_income_inr="bahut zyada")},
        {"scheme_id": "pm-kisan", "applicant": applicant(annual_income_inr=-5000)},
        {"scheme_id": "pm-kisan", "applicant": applicant(pet_name="Sheru")},
    ],
)
def test_unusable_arguments_are_rejected_at_the_boundary(arguments: dict[str, object]) -> None:
    """A model invents fields, sends strings where integers go, and occasionally
    sends a negative income. None of that may reach the rules."""
    with pytest.raises(ToolError):
        check_eligibility(arguments)


def test_a_rejection_never_repeats_the_value_it_rejected() -> None:
    """The field name and the rule are safe to say aloud. The value is an
    applicant's income, and SPEC's threat model keeps it out of logs and spans, so
    it must not travel inside an exception message either."""
    with pytest.raises(ToolError) as raised:
        check_eligibility(
            {"scheme_id": "pm-kisan", "applicant": applicant(annual_income_inr=-777777)}
        )

    assert "777777" not in str(raised.value)
    assert "annual_income_inr" in str(raised.value)


def test_schemes_are_found_in_either_script() -> None:
    """The transcript keeps whichever script the user spoke (SPEC S2), so the
    lookup has to match both rather than assuming a transliteration happened."""
    latin = find_schemes({"query": "mujhe ghar chahiye"})
    devanagari = find_schemes({"query": "मुझे घर चाहिए"})

    assert [scheme.scheme_id for scheme in latin] == ["pmay-g"]
    assert [scheme.scheme_id for scheme in devanagari] == ["pmay-g"]


def test_a_query_matching_nothing_returns_nothing() -> None:
    assert find_schemes({"query": "cricket ka score"}) == []


def test_the_limit_is_honoured() -> None:
    every_keyword = " ".join(rule.keywords[0] for rule in RULES)

    assert len(find_schemes({"query": every_keyword, "limit": 2})) == 2


def test_an_absurd_limit_is_rejected_rather_than_served() -> None:
    with pytest.raises(ToolError):
        find_schemes({"query": "ghar", "limit": 10_000})


def test_dispatch_returns_json_ready_output() -> None:
    """The result goes back to the model as a JSON string, so anything that
    survives here and not through `json.dumps` fails inside the turn instead."""
    eligibility = dispatch(
        "check_eligibility", {"scheme_id": "pm-kisan", "applicant": applicant()}
    )
    schemes = dispatch("find_schemes", {"query": "ghar"})

    assert json.loads(json.dumps(eligibility))["eligible"] is True
    assert json.loads(json.dumps(schemes))["schemes"][0]["scheme_id"] == "pmay-g"


def test_dispatching_an_unknown_tool_is_refused() -> None:
    """Answering it with an empty result teaches the model the call worked."""
    with pytest.raises(ToolError):
        dispatch("check_my_horoscope", {})


def test_the_advertised_schema_matches_what_dispatch_accepts() -> None:
    """Both directions, so a renamed tool cannot be advertised under one name and
    implemented under another. The model only ever sees this list, and a name that
    appears here and not in `TOOLS` is a call that always fails."""
    advertised = {schema["function"]["name"] for schema in tool_schemas()}

    assert advertised == {"check_eligibility", "find_schemes"}
    for name in advertised:
        with pytest.raises(ToolError):
            dispatch(name, {})


def test_every_advertised_tool_declares_its_parameters() -> None:
    for schema in tool_schemas():
        parameters = schema["function"]["parameters"]

        assert parameters["type"] == "object"
        assert parameters["properties"]
        assert schema["function"]["description"]


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.scheme.scheme_id)
# Every threshold in RULES appears exactly, plus one either side of it. A matrix
# of round numbers passed an off-by-one comparison, because none of its values
# landed on a limit and the edge is the only place `<=` and `<` differ.
@pytest.mark.parametrize(
    "income", [0, 119999, 120000, 120001, 199999, 200000, 200001, 300000, 500000, 900000]
)
@pytest.mark.parametrize("acres", [0.0, 4.9, 5.0, 5.1, 7.5])
def test_the_verdict_matches_the_thresholds_and_not_the_wording(
    rule: object, income: int, acres: float
) -> None:
    """Recomputed from the rule rather than read out of the reasons.

    The verdict was briefly derived by testing whether the word "within" appeared
    in prose formatted three lines earlier, so rewording a message would have
    flipped eligibility answers with every test still green. Checking the whole
    matrix against the thresholds themselves cannot be fooled that way.
    """
    expected = True
    if rule.max_annual_income_inr is not None:
        expected = expected and income <= rule.max_annual_income_inr
    if rule.max_land_holding_acres is not None:
        expected = expected and acres <= rule.max_land_holding_acres

    result = check_eligibility(
        {
            "scheme_id": rule.scheme.scheme_id,
            "applicant": applicant(annual_income_inr=income, land_holding_acres=acres),
        }
    )

    assert result.eligible is expected


def test_a_scheme_with_no_thresholds_cannot_be_checked() -> None:
    """`all([])` is True, so a scheme with nothing recorded against it would come
    back eligible for everybody. Not knowing is not the same as qualifying."""
    from vaani.tools import _BY_ID, Scheme, _Rule

    _BY_ID["blank"] = _Rule(scheme=Scheme(scheme_id="blank", name_hi="x", name_en="x"))
    try:
        with pytest.raises(ToolError):
            check_eligibility({"scheme_id": "blank", "applicant": applicant()})
    finally:
        del _BY_ID["blank"]


def test_eligibility_confidence_is_not_a_probability() -> None:
    """Pinned so nobody later reads it as one. A stub knows its two thresholds
    exactly and knows nothing else, so the number is constant on purpose."""
    assert Eligibility(eligible=True, reasons=[], confidence=0.5).confidence == 0.5
