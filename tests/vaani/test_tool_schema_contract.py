"""The tool schema is read by a provider's validator, not only by ours.

Every eligibility question on the deployed service failed, and every test here was
green while it did. The model sent `"50000"` for a field declared `integer`, Groq
validated the arguments against the declared schema on its own side, rejected the call
with `tool_use_failed`, and streamed back an error frame with HTTP 200. No text was ever
generated, so the listener heard the filler and then silence.

The tool tests could not see it because they hand `check_eligibility` a dictionary
directly. That path never builds the schema, never sends it anywhere, and never meets
the validator that actually rejected it, so the arguments under test were the ones we
would have written rather than the ones a model writes. A check whose inputs cannot move
passes forever, and this is the third time that shape has cost this portfolio something.

So these assert on the two things that were wrong: what the published schema says, and
whether the values a model really sends survive it.

**What this cannot do.** It does not call Groq, so it does not prove their validator
accepts the schema. Nothing that runs without a key can. That was verified by hand
against the live API when the fix was made, and `find_schemes` failed on `limit` after
`check_eligibility` was already passing, which is why every numeric field is enumerated
below rather than the two that were reported.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vaani.tools import EligibilityRequest, SchemeQuery, tool_schemas

# Exactly what llama-3.3-70b sent, taken from the `failed_generation` field of the
# rejection rather than imagined. The string "null" for an absent optional is its
# doing too.
AS_THE_MODEL_SENDS_IT = {
    "scheme_id": "pm-kisan",
    "applicant": {
        "state": "Bihar",
        "annual_income_inr": "50000",
        "land_holding_acres": "2.5",
    },
}


def numeric_properties() -> list[tuple[str, str, dict]]:
    """Every number in every published tool schema, with where it came from."""
    found = []
    for tool in tool_schemas():
        name = tool["function"]["name"]
        parameters = tool["function"]["parameters"]
        blocks = [parameters, *(parameters.get("$defs") or {}).values()]
        for block in blocks:
            for field, spec in (block.get("properties") or {}).items():
                declared = spec.get("type")
                types = declared if isinstance(declared, list) else [declared]
                if "integer" in types or "number" in types:
                    found.append((name, field, spec))
    return found


def test_every_number_in_the_schema_also_accepts_a_string() -> None:
    """The bug, as a property of the contract rather than of one field.

    Enumerated rather than spot-checked. `annual_income_inr` and `land_holding_acres`
    were the two in the first rejection, and fixing exactly those two moved the failure
    to `find_schemes` and `limit` on the very next call. A model that writes one number
    as text writes all of them that way.
    """
    numbers = numeric_properties()

    assert numbers, "no numeric fields found: the schema shape changed, not the types"
    for tool, field, spec in numbers:
        declared = spec.get("type")
        assert isinstance(declared, list) and "string" in declared, (
            f"{tool}.{field} is declared {declared!r}. A model sends numbers as text and "
            f"the provider rejects the whole call, so the reply is never generated."
        )


def test_the_schema_carries_no_reference_a_validator_has_to_resolve() -> None:
    """Written for a reader that is not us. The first fix used a PEP 695 alias, which
    Pydantic emits as a `$ref` into `$defs`, and a validator that does not follow it
    sees a field with no type at all."""
    for tool, field, spec in numeric_properties():
        assert "$ref" not in spec, f"{tool}.{field} is published as a reference"


def test_the_arguments_the_model_actually_sent_are_accepted() -> None:
    request = EligibilityRequest.model_validate(AS_THE_MODEL_SENDS_IT)

    # Narrowed on the way in, so nothing downstream compares a string to a threshold.
    assert request.applicant.annual_income_inr == 50000
    assert isinstance(request.applicant.annual_income_inr, int)
    assert request.applicant.land_holding_acres == 2.5


def test_a_number_written_the_way_somebody_says_it_is_still_a_number() -> None:
    """Commas and a rupee sign, because a model asked for an income in a Hindi
    conversation writes one."""
    request = EligibilityRequest.model_validate(
        {
            "scheme_id": "pm-jay",
            "applicant": {"state": "Bihar", "annual_income_inr": "₹1,20,000"},
        }
    )

    assert request.applicant.annual_income_inr == 120000


def test_an_integer_is_still_an_integer() -> None:
    """The control. A schema widened until it accepts anything has not been fixed."""
    request = EligibilityRequest.model_validate(
        {"scheme_id": "pm-kisan", "applicant": {"state": "Bihar", "annual_income_inr": 50000}}
    )

    assert request.applicant.annual_income_inr == 50000


@pytest.mark.parametrize("income", ["bahut kam", "", "  ", "kuch nahi"])
def test_a_quantity_that_is_not_a_number_is_still_refused(income: str) -> None:
    """Widened for the model's habits, not for its guesses. Coercing "bahut kam" to a
    figure would invent an income for somebody who then travels to an office."""
    with pytest.raises(ValidationError):
        EligibilityRequest.model_validate(
            {"scheme_id": "pm-kisan", "applicant": {"state": "Bihar", "annual_income_inr": income}}
        )


def test_a_negative_income_is_still_refused_when_it_arrives_as_text() -> None:
    """The bound has to survive the new door into the model. A string that coerces
    cleanly and then breaks `ge=0` must fail the same way the integer would."""
    with pytest.raises(ValidationError):
        EligibilityRequest.model_validate(
            {"scheme_id": "pm-kisan", "applicant": {"state": "Bihar", "annual_income_inr": "-5000"}}
        )


def test_the_word_for_empty_is_not_a_state_name() -> None:
    """It fills an optional it has nothing for with the string "null", which passes
    `str | None` and then filters a scheme search by a state of that name. A wrong
    answer nobody is told about, rather than a refusal."""
    assert SchemeQuery.model_validate({"query": "kisan", "state": "null"}).state is None
    assert SchemeQuery.model_validate({"query": "kisan", "state": "Bihar"}).state == "Bihar"


def test_the_limit_bounds_survive_arriving_as_text() -> None:
    assert SchemeQuery.model_validate({"query": "kisan", "limit": "5"}).limit == 5
    with pytest.raises(ValidationError):
        SchemeQuery.model_validate({"query": "kisan", "limit": "99"})
