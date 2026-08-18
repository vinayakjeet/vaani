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

from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    WithJsonSchema,
)


class ToolError(Exception):
    """A tool call that cannot be answered, with a reason safe to say aloud.

    Never carries the arguments back. A validation message that quotes what the
    model sent would put an applicant's income into a log line on its way to
    being spoken.
    """


def _from_spoken_number(value: Any) -> Any:
    """A quantity the model wrote as text, which is what this one does.

    Llama emits `"50000"` for a field declared `integer`, and Groq validates tool
    arguments against the declared schema on its own side before any of it reaches
    us. So a strict `integer` is not a stricter contract here, it is a refusal: the
    call is rejected with `tool_use_failed`, no text is ever generated, and the
    listener hears the filler and then nothing. Every eligibility question failed
    that way on the deployed service.

    Widening the declared type and narrowing here is the trade this makes. The
    schema meets the model where it is and the value is an `int` by the time
    anything reads it, so the eligibility rules are unchanged and still cannot be
    handed a string. Commas and rupee signs are stripped because a model asked for
    an income in a Hindi conversation writes one.
    """
    if not isinstance(value, str):
        return value

    cleaned = value.strip().replace(",", "").replace("₹", "").removeprefix("Rs.")
    cleaned = cleaned.strip()
    if not cleaned:
        return value
    try:
        # Through float, so "50000.0" is a number rather than a rejection. A model
        # that writes an income with a decimal point has still said the number.
        return float(cleaned)
    except ValueError:
        # Left alone for the model's own validator to reject with a real message.
        # Guessing at "bahut kam" would invent an income for somebody.
        return value


# A number the schema accepts as either, because the model sends either. The runtime
# type is unchanged: `BeforeValidator` runs first and `int`/`float` still enforce it.
#
# Plain aliases rather than the `type` statement. A PEP 695 alias becomes a `$ref` into
# `$defs`, and this schema is read by a provider's validator rather than by us, so a
# reference it may not resolve is a worse contract than a repeated four-line object.
Rupees = Annotated[
    int,
    Field(ge=0),
    BeforeValidator(_from_spoken_number),
    WithJsonSchema({"type": ["integer", "string"], "minimum": 0}),
]
Acres = Annotated[
    float,
    Field(ge=0),
    BeforeValidator(_from_spoken_number),
    WithJsonSchema({"type": ["number", "string"], "minimum": 0}),
]
Limit = Annotated[
    int,
    Field(ge=1, le=20),
    BeforeValidator(_from_spoken_number),
    WithJsonSchema({"type": ["integer", "string"], "minimum": 1, "maximum": 20}),
]


def _absent_when_spelled_out(value: Any) -> Any:
    """An optional the model filled in with the word for empty.

    It sends `"null"` as a string for a field it has nothing to put in, which passes
    a `str | None` unnoticed and then filters a scheme search by a state of that
    name. A quiet wrong answer rather than a refusal, so it is caught here.
    """
    if isinstance(value, str) and value.strip().casefold() in {"", "null", "none", "nan"}:
        return None
    return value


OptionalText = Annotated[str | None, BeforeValidator(_absent_when_spelled_out)]


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    annual_income_inr: Rupees
    land_holding_acres: Acres = 0.0
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
    state: OptionalText = None
    limit: Limit = 5


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
        # The valid ids, not just the complaint. `ToolError`'s message is what the
        # model reads to correct itself within `MAX_TOOL_ROUNDS`, and a scheme name
        # is not a scheme id: asked about "PM Kisan", the model called this with
        # `scheme_id="PM Kisan"` rather than `"pm-kisan"`, and a bare "no scheme
        # with id 'PM Kisan'" gave it nothing to fix, so every retry guessed again
        # and the turn burned its whole budget landing on "I could not check."
        # Listing the ids is a smaller contract change than requiring `find_schemes`
        # first, which is a workflow nothing enforces and this model did not follow
        # on its own.
        raise ToolError(
            f"no scheme with id {request.scheme_id!r}. Valid scheme_id values: "
            f"{', '.join(sorted(_BY_ID))}"
        )

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
            "parameters": _inline_defs(model.model_json_schema()),
        },
    }


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace every `$ref` to `$defs` with the definition itself, and drop `$defs`.

    `EligibilityRequest.applicant` is a nested `Applicant` model, and pydantic
    renders any nested `BaseModel` as `$ref: "#/$defs/Applicant"` rather than
    inlining it, the same shape the `Rupees`/`Acres` aliases above were built to
    avoid at the field level, one layer up. Verified live against Groq's
    `openai/gpt-oss-120b`: with the `$ref` present it invented `applicant` field
    names (`land_acres`, `land_area_acres`, ...) and dropped required ones, 5 of
    5 tries; with the same schema inlined it produced the exact declared fields,
    5 of 5 tries. The model is reading the schema it is shown, not the one a
    `$ref` points at, so the schema it is shown must not point anywhere.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                target = defs[node["$ref"].removeprefix("#/$defs/")]
                return resolve({k: v for k, v in target.items()})
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


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
