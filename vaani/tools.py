"""The tool contract Vaani needs from Dastavez, and a local stub of it.

Dastavez does not exist yet (SPEC A2), so Vaani defines the contract it needs and
ships an implementation behind it. When Dastavez lands, the implementation is
replaced and the contract does not move.

The scheme data below is fixture data for testing a pipeline, not policy. Real
eligibility rules vary by state and by budget year, and an agent that tells
somebody they qualify for a welfare payment on the strength of a threshold
invented for a unit test has done harm to a real person who then travels to an
office. Every result carries `indicative=True` and the spoken reply says so.
Replacing this with sourced data is Dastavez's job, and SPEC non-goal 10 says
Vaani is not that.

Arguments arrive from a language model, which means they are untrusted input in
the ordinary sense: wrong types, invented fields, negative incomes, and scheme
ids that do not exist. They are validated at this boundary and rejected here
rather than deeper in, which is Quality Bar point 6.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ToolError(Exception):
    """A tool call that cannot be answered, with a reason safe to say aloud.

    Never carries the arguments back. A validation message that quotes what the
    model sent would put an applicant's income into a log line on its way to
    being spoken.
    """


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    annual_income_inr: int = Field(ge=0)
    land_holding_acres: float = Field(default=0.0, ge=0)
    category: str = "general"


class EligibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheme_id: str
    applicant: Applicant


class Eligibility(BaseModel):
    eligible: bool
    reasons: list[str]
    confidence: float
    indicative: bool = True


class SchemeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    state: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class Scheme(BaseModel):
    scheme_id: str
    name_hi: str
    name_en: str


class _Rule(BaseModel):
    scheme: Scheme
    max_annual_income_inr: int | None = None
    max_land_holding_acres: float | None = None
    keywords: tuple[str, ...] = ()


# Deliberately small. Five schemes exercise every branch the pipeline has, and a
# larger set would invite treating this as a source of truth.
RULES: tuple[_Rule, ...] = (
    _Rule(
        scheme=Scheme(scheme_id="pm-kisan", name_hi="पीएम किसान", name_en="PM-KISAN"),
        max_land_holding_acres=5.0,
        keywords=("kisan", "farmer", "किसान", "kheti", "zameen"),
    ),
    _Rule(
        scheme=Scheme(scheme_id="pm-jay", name_hi="आयुष्मान भारत", name_en="Ayushman Bharat"),
        max_annual_income_inr=500000,
        keywords=("health", "ilaj", "इलाज", "hospital", "bimari"),
    ),
    _Rule(
        scheme=Scheme(scheme_id="pmay-g", name_hi="पीएम आवास योजना", name_en="PM Awas Yojana"),
        max_annual_income_inr=300000,
        keywords=("ghar", "awas", "घर", "house", "makan"),
    ),
    _Rule(
        scheme=Scheme(scheme_id="ujjwala", name_hi="उज्ज्वला योजना", name_en="PM Ujjwala Yojana"),
        max_annual_income_inr=200000,
        keywords=("gas", "cylinder", "गैस", "chulha", "lpg"),
    ),
    _Rule(
        scheme=Scheme(scheme_id="nsap-oap", name_hi="वृद्धावस्था पेंशन", name_en="Old Age Pension"),
        max_annual_income_inr=120000,
        keywords=("pension", "पेंशन", "budhapa", "old age", "vridha"),
    ),
)

_BY_ID = {rule.scheme.scheme_id: rule for rule in RULES}


def check_eligibility(arguments: dict[str, Any]) -> Eligibility:
    """Whether one applicant clears one scheme's thresholds.

    Every reason is phrased as a fact about a threshold rather than a verdict, so
    a reply built from them can say what would change the answer. "Not eligible"
    with no reason is the failure mode people come back about.
    """
    request = _validated(EligibilityRequest, arguments)

    rule = _BY_ID.get(request.scheme_id)
    if rule is None:
        raise ToolError(f"no scheme with id {request.scheme_id!r}")

    # The verdict comes from the comparisons, never from reading the sentences
    # they produced. Deriving a boolean by matching a substring of prose written
    # three lines earlier means rewording a message silently flips an eligibility
    # answer, and the test that catches it is the one nobody writes.
    checks: list[bool] = []
    reasons: list[str] = []

    if rule.max_annual_income_inr is not None:
        within = request.applicant.annual_income_inr <= rule.max_annual_income_inr
        checks.append(within)
        reasons.append(
            f"annual income limit is {rule.max_annual_income_inr} and the applicant is "
            f"{'within' if within else 'above'} it"
        )
    if rule.max_land_holding_acres is not None:
        within = request.applicant.land_holding_acres <= rule.max_land_holding_acres
        checks.append(within)
        reasons.append(
            f"land holding limit is {rule.max_land_holding_acres} acres and the applicant is "
            f"{'within' if within else 'above'} it"
        )

    if not checks:
        # Nothing to compare against is not the same as clearing every bar. SPEC
        # S5's rule applies: say the check could not be completed rather than
        # answering it confidently.
        raise ToolError(f"no thresholds are recorded for scheme {request.scheme_id!r}")

    eligible = all(checks)

    # A stub knows its thresholds exactly and knows nothing else, so the number
    # says "these two rules were checked" rather than expressing a belief. It is
    # not a probability and the README says so.
    return Eligibility(eligible=eligible, reasons=reasons, confidence=0.5)


def find_schemes(arguments: dict[str, Any]) -> list[Scheme]:
    """Schemes whose keywords appear in the query, in declaration order.

    Keyword matching in both scripts, because the user says "ghar" and "घर" and
    the transcript preserves whichever they used (SPEC S2). Ranking is Dastavez's
    problem; this returns matches in a fixed order so a test can assert on them.
    """
    query = _validated(SchemeQuery, arguments)
    needle = query.query.casefold()

    matched = [
        rule.scheme for rule in RULES if any(word in needle for word in rule.keywords)
    ]
    return matched[: query.limit]


TOOLS = {"check_eligibility": check_eligibility, "find_schemes": find_schemes}


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool by the name the model used, and return JSON-ready output.

    Unknown names raise rather than returning empty, because a model calling a
    tool that does not exist is a prompt or contract problem, and answering it
    with an empty result teaches it the call succeeded.
    """
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"no tool named {name!r}")

    result = tool(arguments)
    if isinstance(result, list):
        return {"schemes": [item.model_dump() for item in result]}
    return result.model_dump()


def tool_schemas() -> list[dict[str, Any]]:
    """The contract in the shape an OpenAI-compatible `tools[]` payload wants."""
    return [
        _schema("check_eligibility", check_eligibility.__doc__, EligibilityRequest),
        _schema("find_schemes", find_schemes.__doc__, SchemeQuery),
    ]


def _schema(name: str, description: str | None, model: type[BaseModel]) -> dict[str, Any]:
    summary = (description or "").strip().splitlines()[0]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": summary,
            "parameters": model.model_json_schema(),
        },
    }


def _validated[T: BaseModel](model: type[T], arguments: dict[str, Any]) -> T:
    """Parse arguments, converting a pydantic error into a sayable one.

    The field name and the rule are safe to repeat. The value is not: it is an
    applicant's income or landholding, and SPEC's threat model keeps those out of
    logs and spans entirely.
    """
    try:
        return model.model_validate(arguments)
    except ValidationError as exc:
        problems = ", ".join(
            f"{'.'.join(str(part) for part in error['loc'])} {error['msg'].casefold()}"
            for error in exc.errors()
        )
        raise ToolError(f"the arguments were not usable: {problems}") from exc
