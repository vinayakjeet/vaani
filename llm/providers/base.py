from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Callable
from typing import Protocol

import httpx
import structlog

from llm.types import (
    ChatMessage,
    ChatResponse,
    ProviderClientError,
    ProviderConfigError,
    ProviderError,
    RateLimitError,
    StreamCompleted,
    StreamEvent,
    TextChunk,
    ToolCall,
    ToolCallsRequested,
)

logger = structlog.get_logger(__name__)


class Provider(Protocol):
    """Anything ChatClient can dispatch to. OpenAICompatibleProvider below covers
    groq/cerebras/openrouter/ollama/sarvam/gemini today; a future native provider
    with its own native message shape implements this Protocol directly instead
    of subclassing OpenAICompatibleProvider."""

    name: str

    async def chat_completion(
        self, messages: list[ChatMessage], **kwargs: object
    ) -> ChatResponse: ...

    def stream_completion(
        self, messages: list[ChatMessage], **kwargs: object
    ) -> AsyncIterator[StreamEvent]: ...


def default_usage_parser(payload: dict) -> tuple[int | None, int | None]:
    """Standard OpenAI usage shape: {"usage": {"prompt_tokens", "completion_tokens"}}."""
    usage = payload.get("usage") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

# Reasoning models spend far longer than httpx's 5 second default before the first
# byte, so leaving it unset turns ordinary slow generations into network errors.
# Generous rather than unbounded: a genuinely hung connection must still fail.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


def parse_retry_after(resp: httpx.Response) -> float | None:
    """How long to wait after a 429, from wherever the provider chose to put it.

    Three places, in order of reliability:

    1. The `Retry-After` header, which is the standard and which Gemini does not
       send.
    2. A `google.rpc.RetryInfo` entry in `error.details`, as `retryDelay: "48s"`.
    3. The human-readable error message: "Please retry in 48.63971551s."

    Falling back through all three matters more than it looks. Without a value the
    throttle uses a short default cooldown, so a limit that needs 49 seconds gets
    retried after 5, fails, and burns quota indefinitely without ever succeeding.
    Free tiers are exactly where this happens.
    """
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass

    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None

    # Gemini wraps its error object in a single-element list.
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return None

    # `error` is an object in Google's shape but a bare string in plenty of
    # others, so it cannot be assumed to have fields.
    error = payload.get("error")
    if not isinstance(error, dict):
        return None

    for detail in error.get("details") or []:
        if not isinstance(detail, dict):
            continue
        if detail.get("@type", "").endswith("RetryInfo"):
            delay = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                return float(delay)
            except ValueError:
                continue

    if match := _RETRY_IN.search(str(error.get("message", ""))):
        value = float(match.group(1))
        # Gemini reports sub-second waits in milliseconds. Reading 607ms as 607
        # seconds would stall a run for ten minutes over a delay of half a second.
        return value / 1000 if match.group(2).lower() == "ms" else value
    return None


def total_aware_usage_parser(payload: dict) -> tuple[int | None, int | None]:
    """Usage parser for providers that bill tokens absent from the itemised fields.

    Gemini's OpenAI-compatible layer reports reasoning tokens in `total_tokens`
    only. A real response looked like prompt=2, completion=9, total=197: the model
    spent 186 tokens thinking, and reading only the itemised fields undercounts
    the run by roughly eighteen times.

    So output is derived as total minus prompt whenever that exceeds the reported
    completion count. Overstating output slightly is the safe direction to be
    wrong in for a benchmark table and a cost ceiling.
    """
    usage = payload.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")

    if prompt is None or total is None:
        return prompt, completion

    derived = total - prompt
    if completion is None or derived > completion:
        return prompt, derived
    return prompt, completion


class OpenAICompatibleProvider:
    """Backs any provider exposing an OpenAI-compatible POST {base_url}/chat/completions.

    Some OpenAI-compatible layers (Gemini's, Sarvam's) diverge slightly in how usage
    is reported - pass a custom `usage_parser` for those instead of assuming
    `default_usage_parser` fits every provider verbatim.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        default_model: str,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
        usage_parser: Callable[[dict], tuple[int | None, int | None]] = default_usage_parser,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._default_model = default_model
        self._input_price_per_1m = input_price_per_1m
        self._output_price_per_1m = output_price_per_1m
        self._usage_parser = usage_parser
        self._client = client

    def _require_api_key(self) -> str:
        key = os.environ.get(self._api_key_env)
        if not key:
            raise ProviderConfigError(
                f"provider '{self.name}' requires env var {self._api_key_env} to be set"
            )
        return key

    def _cost_usd(self, tokens_in: int | None, tokens_out: int | None) -> float | None:
        if self._input_price_per_1m is None or self._output_price_per_1m is None:
            return None
        if tokens_in is None or tokens_out is None:
            return None
        return (tokens_in / 1_000_000) * self._input_price_per_1m + (
            tokens_out / 1_000_000
        ) * self._output_price_per_1m

    async def chat_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> ChatResponse:
        api_key = self._require_api_key()
        resolved_model = model or self._default_model
        payload = {
            "model": resolved_model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            **kwargs,
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        client = self._client or httpx.AsyncClient(
            base_url=self._base_url, timeout=DEFAULT_TIMEOUT
        )
        try:
            try:
                resp = await client.post("/chat/completions", json=payload, headers=headers)
            except httpx.RequestError as exc:
                raise ProviderError(f"{self.name}: network error: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()

        self._raise_for_status(resp)

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens_in, tokens_out = self._usage_parser(data)

        return ChatResponse(
            text=text,
            provider=self.name,
            model=resolved_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost_usd(tokens_in, tokens_out),
        )

    def _raise_for_status(self, resp: httpx.Response) -> None:
        """Map a response status onto the exception taxonomy.

        Only safe once the body has been read. A streamed response has to be read
        explicitly before `.text` and `.json()` exist, and `parse_retry_after`
        needs the body to find the delay Gemini puts there instead of in a header.
        """
        if resp.status_code == 429:
            raise RateLimitError(
                f"{self.name}: rate limited", retry_after=parse_retry_after(resp)
            )
        if resp.status_code >= 500:
            raise ProviderError(f"{self.name}: server error {resp.status_code}")
        if resp.status_code >= 400:
            raise ProviderClientError(f"{self.name}: client error {resp.status_code}: {resp.text}")

    async def stream_completion(
        self, messages: list[ChatMessage], model: str | None = None, **kwargs: object
    ) -> AsyncIterator[StreamEvent]:
        """Server-sent events from an OpenAI-compatible endpoint, as typed events.

        Text arrives as it is generated, which is the whole point: synthesis starts
        on the first sentence rather than the whole reply. Tool-call arguments
        arrive in fragments and are assembled here, so a caller never sees half a
        JSON object.
        """
        api_key = self._require_api_key()
        resolved_model = model or self._default_model
        payload = {
            "model": resolved_model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "stream": True,
            # Streamed responses omit usage unless it is asked for, and without it
            # a streamed call reports no tokens and no cost. The ablation compares
            # a streamed configuration against an unstreamed baseline, so a blank
            # cost column on one side would not be a missing number, it would be a
            # comparison nobody can read. Providers that reject the field surface a
            # 4xx with the body attached, which is diagnosable; pass
            # `stream_options=None` to drop it for one that does.
            "stream_options": {"include_usage": True},
            **kwargs,
        }
        if payload.get("stream_options") is None:
            payload.pop("stream_options", None)
        headers = {"Authorization": f"Bearer {api_key}"}

        client = self._client or httpx.AsyncClient(
            base_url=self._base_url, timeout=DEFAULT_TIMEOUT
        )
        try:
            try:
                async with client.stream(
                    "POST", "/chat/completions", json=payload, headers=headers
                ) as resp:
                    if resp.status_code >= 400:
                        # Nothing has been read yet, and both the status mapping
                        # and the retry-after hint need the body.
                        await resp.aread()
                        self._raise_for_status(resp)

                    async for event in self._events(resp):
                        yield event
            except httpx.RequestError as exc:
                raise ProviderError(f"{self.name}: network error: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()

    async def _events(self, resp: httpx.Response) -> AsyncIterator[StreamEvent]:
        fragments: dict[int, dict[str, str]] = {}
        produced_text = False
        finish_reason: str | None = None
        tokens_in: int | None = None
        tokens_out: int | None = None
        terminated = False

        async for raw in resp.aiter_lines():
            line = raw.strip()
            # Blank lines separate events and a leading colon is a keepalive
            # comment, which is how a provider holds the connection open while a
            # reasoning model thinks.
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue

            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                terminated = True
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                # One unreadable frame does not justify failing a turn that is
                # already audible, so it is skipped and counted rather than
                # raised. Logged without the payload: a frame carries reply text,
                # which is about a real person's eligibility. M3.2 is where this
                # becomes a metric with a reason label.
                logger.warning("provider.stream.unreadable_frame", provider=self.name)
                continue

            if error := chunk.get("error"):
                # A provider can refuse mid-stream and still have sent 200, because
                # the status was written before it had anything to refuse. Groq does
                # this for a tool call whose arguments do not match the declared
                # schema: `event: error`, one data frame, no choices, no [DONE].
                #
                # Without this the frame falls through every branch below and the
                # stream ends having yielded nothing, which reaches the truncation
                # guard and is reported as a dropped connection. A schema rejection
                # and a dead socket then look identical, and the provider's own
                # explanation of which one it was has already been discarded. That
                # cost a live turn and an afternoon: every reply arrived as filler
                # and then silence, with "stream ended before any content" in the log.
                #
                # The code and the message, never `failed_generation`. That key holds
                # the arguments the model tried to send, which on this pipeline is an
                # applicant's income, and it is the one part of the body that must not
                # travel. `ProviderClientError` is already excluded from what the
                # session logs, for the same reason one step further out.
                raise ProviderClientError(
                    f"{self.name}: {error.get('code') or error.get('type') or 'error'}: "
                    f"{error.get('message', '')}"
                )

            if chunk.get("usage"):
                tokens_in, tokens_out = self._usage_parser(chunk)

            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                delta = choice.get("delta") or {}
                if content := delta.get("content"):
                    produced_text = True
                    yield TextChunk(text=content)

                for fragment in delta.get("tool_calls") or []:
                    slot = fragments.setdefault(
                        fragment.get("index", 0), {"id": "", "name": "", "arguments": ""}
                    )
                    if identifier := fragment.get("id"):
                        slot["id"] = identifier
                    function = fragment.get("function") or {}
                    if name := function.get("name"):
                        slot["name"] = name
                    # Arguments are concatenated, never replaced. A provider sends
                    # them a few characters at a time and the last fragment is
                    # usually just a closing brace.
                    slot["arguments"] += function.get("arguments") or ""

        if not terminated and finish_reason is None and not produced_text:
            # The connection ended mid-stream with no terminator and no finish
            # reason, so the reply is truncated. Saying nothing here would hand the
            # caller a `StreamCompleted` and let half an answer pass for a whole one,
            # which is the failure the fault helper was built to find: a dropped
            # stream and a finished one looked identical.
            #
            # The text already yielded stays yielded. It has been spoken by now and
            # cannot be un-said; what must not happen is the caller believing there
            # was no more.
            raise ProviderError(f"{self.name}: stream ended before any content")

        if not terminated and finish_reason is None:
            # Text arrived and then the stream stopped without saying it was done. The
            # words already spoken cannot be un-said, so raising here would trade a
            # partial answer for silence, which every degradation rule in SPEC exists
            # to prevent. It is reported as truncated instead, so a caller and a
            # waterfall can both tell it apart from a clean finish.
            logger.warning("provider.stream.truncated", provider=self.name)
            finish_reason = "truncated"

        if fragments:
            yield ToolCallsRequested(
                calls=[
                    ToolCall(id=slot["id"], name=slot["name"], arguments=slot["arguments"])
                    for _, slot in sorted(fragments.items())
                ]
            )

        yield StreamCompleted(
            finish_reason=finish_reason,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost_usd(tokens_in, tokens_out),
        )
