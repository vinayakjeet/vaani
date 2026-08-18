"""M4.7. Exact-match runner over eval/scenarios.json, against real tool calls.

    uv run python eval/run_eval.py

"Exact-match" here means the structured facts the pipeline actually decided, not
the spoken reply's exact words: which tool was called, with which scheme_id, and
what eligibility verdict it returned. Matching on prose would be either far too
brittle (the same correct answer can be phrased many ways) or would need its own
NLU layer whose own correctness this eval would then also be trusting blind.
`vaani.llm_turn.dispatch` is wrapped for the duration of each scenario to record
every real tool call the turn makes, which is the same function the pipeline
calls live, not a parallel implementation of it.

**This gate is not wired into CI yet, on purpose.** `eval/scenarios.json`'s
expected outcomes are authored, not adjudicated, exactly the condition this
project's own LEARNING record says is not enough to trust a scored gate: see
`eval/build_scenarios.py`'s own header. Blocking a merge on labels nobody has
reviewed would be the ShipGate mistake with a different face, a check whose
inputs (here, its own ground truth) were never actually verified. Wiring this
into CI is the last step, after a human review pass, not before it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import ChatClient  # noqa: E402
from vaani.llm_turn import StreamedTurn  # noqa: E402
from vaani.tools import ToolError  # noqa: E402
from vaani.tools import dispatch as real_dispatch  # noqa: E402

SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios.json"


@dataclass
class Call:
    name: str
    arguments: dict
    result: object
    error: str | None = None


@dataclass
class ScenarioResult:
    id: str
    category: str
    passed: bool
    reason: str
    calls: list[Call]
    reply_chars: int
    crashed: bool = False


def _recording_dispatch(calls: list[Call]):
    def _dispatch(name: str, arguments: dict) -> object:
        # A ToolError (bad scheme_id, arguments the schema rejects) is a real
        # attempt, not a non-event, and `_result_of` in llm_turn.py depends on
        # it propagating so the model sees a failure it can retry against.
        # Recording it here, before re-raising, is what lets `check()` tell
        # "check_eligibility was never attempted" apart from "it was attempted
        # and failed" instead of both reading as the tool never being called.
        try:
            result = real_dispatch(name, arguments)
        except ToolError as exc:
            calls.append(Call(name=name, arguments=dict(arguments), result=None, error=str(exc)))
            raise
        calls.append(Call(name=name, arguments=dict(arguments), result=result))
        return result

    return _dispatch


async def run_scenario(scenario: dict) -> ScenarioResult:
    calls: list[Call] = []
    turn = StreamedTurn(llm=ChatClient())

    with patch("vaani.llm_turn.dispatch", _recording_dispatch(calls)):
        reply = "".join([chunk async for chunk in turn.run(scenario["text"], history=[])])

    passed, reason = check(scenario["expected"], calls, reply)
    return ScenarioResult(
        id=scenario["id"],
        category=scenario["category"],
        passed=passed,
        reason=reason,
        calls=calls,
        reply_chars=len(reply),
    )


def check(expected: dict, calls: list[Call], reply: str) -> tuple[bool, str]:
    """Every expectation key `eval/build_scenarios.py` writes, checked against
    what actually happened. Unknown keys are ignored rather than erroring, so a
    scenario can carry documentation-only fields without breaking the runner."""
    if not reply.strip():
        return False, "reply was empty"

    # Every attempt, successful or not: a call that failed schema validation was
    # still a real attempt, and a scenario asking for `tool: None` must fail if
    # one was made even though it never produced a usable result.
    tool_names = [c.name for c in calls]
    # Only calls that actually returned something: what `scheme_id`/`eligible`
    # can be checked against.
    ok_calls = [c for c in calls if c.error is None]

    if "tool" in expected:
        wanted = expected["tool"]
        if wanted is None:
            if tool_names:
                return False, f"expected no tool call, got {tool_names}"
        elif wanted not in tool_names:
            return False, f"expected {wanted!r} to be called, got {tool_names}"

    # Distinct from "tool": None. find_schemes is a harmless lookup and cannot
    # itself supply a number, so a scenario that only cares whether a figure or
    # a verdict could have been invented should forbid check_eligibility
    # specifically rather than every tool, or it fails on a model correctly
    # verifying that nothing matches before saying so.
    if "forbidden_tool" in expected and expected["forbidden_tool"] in tool_names:
        return False, f"expected {expected['forbidden_tool']!r} not to be called, got {tool_names}"

    if "scheme_id" in expected:
        matching = [c for c in ok_calls if c.name == "check_eligibility"]
        if not matching:
            failed = [c for c in calls if c.name == "check_eligibility"]
            if failed:
                return False, f"check_eligibility was attempted and failed: {failed[-1].error}"
            return False, "expected check_eligibility, none was called"
        got = matching[-1].arguments.get("scheme_id")
        if got != expected["scheme_id"]:
            return False, f"expected scheme_id={expected['scheme_id']!r}, got {got!r}"

    if "eligible" in expected:
        matching = [c for c in ok_calls if c.name == "check_eligibility"]
        if not matching:
            failed = [c for c in calls if c.name == "check_eligibility"]
            if failed:
                return False, f"check_eligibility was attempted and failed: {failed[-1].error}"
            return False, "expected check_eligibility, none was called"
        # dispatch() returns model_dump()'d JSON, a dict, never the pydantic object.
        got_eligible = matching[-1].result.get("eligible")
        if got_eligible != expected["eligible"]:
            return False, f"expected eligible={expected['eligible']!r}, got {got_eligible!r}"

    if expected.get("scheme_id_in_results") is not None:
        wanted = expected["scheme_id_in_results"]
        matching = [c for c in ok_calls if c.name == "find_schemes"]
        if not matching:
            return False, "expected find_schemes, none was called"
        ids = [s["scheme_id"] for s in matching[-1].result.get("schemes", [])]
        if wanted not in ids:
            return False, f"expected {wanted!r} among find_schemes results, got {ids}"

    # "no_invented_scheme_name" has no automated check here: telling an invented
    # scheme name from a real one paraphrased needs reading the reply, not
    # pattern matching it. Left for the human review pass `eval/scenarios.json`
    # already needs before this gate is trustworthy; `reply_chars` in the raw
    # JSON output is there so that review has something to read against.

    if expected.get("no_eligibility_claim") and any(c.name == "check_eligibility" for c in calls):
        return False, "expected no eligibility check, but check_eligibility was called"

    # A structural proxy, not a semantic one: only check_eligibility's result can
    # supply a number (a threshold, a confidence), so only its presence means a
    # figure was sourced for the reply to state. find_schemes returns scheme
    # names and ids, never a number, so calling it is not the risk this checks
    # for. `vaani.grounding` enforces the real invariant, reply text against
    # tool-sourced figures, at synthesis time on the live pipeline; this does
    # not re-verify that, it only confirms the scenario reached the same
    # starting condition grounding depends on.
    if expected.get("no_uncalled_figure") and any(c.name == "check_eligibility" for c in calls):
        return False, f"expected no check_eligibility call: {tool_names}"

    return True, "ok"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--json", type=Path, default=Path(__file__).with_suffix(".results.json"))
    args = parser.parse_args()

    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if args.category:
        scenarios = [s for s in scenarios if s["category"] == args.category]
    if args.limit:
        scenarios = scenarios[: args.limit]

    results: list[ScenarioResult] = []
    for i, scenario in enumerate(scenarios, 1):
        # A live network blip (a DNS failure, a dropped connection) must not cost
        # every scenario already run: 50 scenarios is dozens of real calls, and
        # losing all of them to one bad one is the same mistake bench/ablation.py
        # made and was fixed for. Recorded as failed and crashed, not silently
        # dropped, so the run's own JSON shows what happened.
        try:
            result = await run_scenario(scenario)
        except Exception as exc:  # noqa: BLE001
            result = ScenarioResult(
                id=scenario["id"],
                category=scenario["category"],
                passed=False,
                reason=f"crashed: {type(exc).__name__}: {exc}",
                calls=[],
                reply_chars=0,
                crashed=True,
            )
        results.append(result)
        mark = "PASS" if result.passed else "FAIL"
        print(f"  [{i}/{len(scenarios)}] {mark} {result.id}: {result.reason}", file=sys.stderr)

    passed = sum(1 for r in results if r.passed)
    crashed = sum(1 for r in results if r.crashed)
    if results:
        print(f"\n{passed}/{len(results)} passed ({passed / len(results):.1%})")
        if crashed:
            print(f"{crashed} scenario(s) crashed rather than failed a check; see reason per row")
    else:
        print("no scenarios run")

    by_category: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)
    for cat, rows in sorted(by_category.items()):
        cat_passed = sum(1 for r in rows if r.passed)
        print(f"  {cat}: {cat_passed}/{len(rows)}")

    args.json.write_text(
        json.dumps(
            [
                {
                    "id": r.id,
                    "category": r.category,
                    "passed": r.passed,
                    "reason": r.reason,
                    "reply_chars": r.reply_chars,
                    "crashed": r.crashed,
                    "calls": [
                        {"name": c.name, "arguments": c.arguments, "error": c.error}
                        for c in r.calls
                    ],
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nRaw results: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
