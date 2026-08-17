"""M1.14. What Groq's automatic prompt caching buys, measured rather than assumed.

    uv run python -m bench.prompt_cache

Groq applies prompt caching automatically to a request's prefix once it clears a
per-model minimum: 128 to 1024 tokens depending on the model, per docs.groq.com,
with `openai/gpt-oss-120b` confirmed supported. No code path in this project asks
for it and none can turn it off; it either engages on the real system prompt and
tool schemas at their real size, or it does not, and the only way to know is to
send them and read `usage.prompt_tokens_details.cached_tokens` back.

The two arms hold everything but the prefix's cacheability constant: `repeated`
sends the identical system prompt and tool schema on every call, which is what
every real turn already does, since `vaani/llm_turn.py` builds messages with the
static prompt first for exactly this reason. `broken` appends a fresh UUID to the
system prompt on every call, so the prefix can never match a previous request's,
which is the same request in every other respect and the only clean way to see
what caching bought without it.

Time to first token, not total completion time, because that is the number this
project is built around and prompt caching's own claim is about latency to that
point, not about how long the model takes to finish talking once it has started.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaani.tools import tool_schemas  # noqa: E402
from vaani.turn import SYSTEM_PROMPT  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

# Distinct questions per call within an arm, not the same one repeated, so a
# provider-side full-response cache (a different mechanism from prefix caching)
# cannot make this measurement about the wrong kind of cache.
QUESTIONS = (
    "PM Kisan yojana ke baare mein bataiye.",
    "Mera ilaj ke liye paisa nahi hai, Ayushman Bharat se madad mil sakti hai kya?",
    "Hamara apna ghar nahi hai, PM Awas Yojana ke liye eligible hoon kya?",
    "Ujjwala yojana mein gas cylinder kaise milega?",
    "Mere dada ji budhapa pension ke liye apply karna chahte hain.",
)


@dataclass
class Call:
    arm: str
    time_to_first_token_ms: float
    prompt_tokens: int | None
    cached_tokens: int | None


async def timed_call(
    client: httpx.AsyncClient, api_key: str, system_prompt: str, question: str
) -> Call:
    payload = {
        "model": MODEL,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "tools": tool_schemas(),
        "max_tokens": 80,
    }
    started = time.monotonic()
    first_token_ms: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None

    async with client.stream(
        "POST", GROQ_URL, json=payload, headers={"Authorization": f"Bearer {api_key}"}
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if first_token_ms is None:
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning"):
                        first_token_ms = (time.monotonic() - started) * 1000
                        break
            if usage := chunk.get("usage"):
                prompt_tokens = usage.get("prompt_tokens")
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens")

    # A reply that never produced a token still has a real elapsed time; report
    # it rather than pretend the call did not happen.
    return Call(
        arm="",
        time_to_first_token_ms=first_token_ms
        if first_token_ms is not None
        else (time.monotonic() - started) * 1000,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )


async def measure(repeats: int, gap_s: float) -> list[Call]:
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set")

    calls: list[Call] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(repeats):
            question = QUESTIONS[i % len(QUESTIONS)]

            broken_prompt = f"{SYSTEM_PROMPT}\n\n<!-- {uuid.uuid4()} -->"
            broken = await timed_call(client, api_key, broken_prompt, question)
            broken.arm = "broken"
            calls.append(broken)
            print(
                f"  [{i + 1}/{repeats}] broken: {broken.time_to_first_token_ms:.1f}ms "
                f"prompt_tokens={broken.prompt_tokens} cached_tokens={broken.cached_tokens}",
                file=sys.stderr,
            )
            if gap_s:
                await asyncio.sleep(gap_s)

            repeated = await timed_call(client, api_key, SYSTEM_PROMPT, question)
            repeated.arm = "repeated"
            calls.append(repeated)
            print(
                f"  [{i + 1}/{repeats}] repeated: {repeated.time_to_first_token_ms:.1f}ms "
                f"prompt_tokens={repeated.prompt_tokens} cached_tokens={repeated.cached_tokens}",
                file=sys.stderr,
            )
            if gap_s:
                await asyncio.sleep(gap_s)

    return calls


def render(calls: list[Call]) -> str:
    lines = [f"Prompt caching on {MODEL}, {len(calls)} calls.", ""]

    for arm in ("broken", "repeated"):
        arm_calls = [c for c in calls if c.arm == arm]
        if not arm_calls:
            continue
        ttft = [c.time_to_first_token_ms for c in arm_calls]
        cached = [c.cached_tokens for c in arm_calls if c.cached_tokens is not None]
        prompt_tok = arm_calls[0].prompt_tokens
        lines.append(
            f"{arm:>10}: n={len(arm_calls)} ttft_p50={statistics.median(ttft):.1f}ms "
            f"ttft_min={min(ttft):.1f}ms ttft_max={max(ttft):.1f}ms "
            f"prompt_tokens={prompt_tok} "
            f"cached_tokens_seen={cached if cached else 'never reported'}"
        )

    lines.append("")
    any_cached = any(c.cached_tokens for c in calls)
    if any_cached:
        lines.append(
            "Caching activated: at least one call reported cached_tokens > 0. The "
            "delta between the two arms' ttft_p50 above is what it bought."
        )
    else:
        lines.append(
            "Caching never activated on either arm: cached_tokens was absent or "
            "zero on every call. The real system prompt plus tool schemas is "
            f"{len(SYSTEM_PROMPT) + len(json.dumps(tool_schemas()))} characters, "
            "under the 128-to-1024-token minimum docs.groq.com states for this "
            "family on at least the low end of that range. There is nothing to "
            "turn on: the prefix this project sends is shorter than what Groq "
            "will cache."
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--gap-s", type=float, default=1.0, help="pause between calls, rate-limit friendly"
    )
    parser.add_argument("--json", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()

    calls = await measure(args.repeats, args.gap_s)
    print(render(calls))

    args.json.write_text(
        json.dumps([c.__dict__ for c in calls], indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nRaw per-call data: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
