from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str

    # Set only on the assistant message that asked for tools, and on the `tool`
    # messages answering it. Serialised with `exclude_none`, so a provider that
    # has never seen these fields receives exactly the payload it did before.
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: float = 0.0


class ToolCall(BaseModel):
    """One tool the model asked for, with its arguments exactly as it sent them.

    `arguments` stays a raw JSON string rather than a parsed dict. Models emit
    malformed JSON often enough that parsing here would make a provider-layer
    concern out of it, and the caller has to validate the parsed result against a
    schema regardless. Parsing once, where the validation lives, is one place for
    both failures instead of two.
    """

    id: str
    name: str
    arguments: str


class TextChunk(BaseModel):
    """A fragment of the reply. Sentence segmentation happens downstream."""

    text: str


class ToolCallsRequested(BaseModel):
    """The model stopped to call tools. Emitted once, with the calls assembled.

    Providers deliver tool-call arguments in fragments across many chunks, so
    forwarding the fragments would make every caller reassemble them and get the
    indexing subtly wrong in its own way.
    """

    calls: list[ToolCall]


class StreamCompleted(BaseModel):
    """The last event of every stream, carrying what only the end knows."""

    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None


StreamEvent = TextChunk | ToolCallsRequested | StreamCompleted


class ProviderError(Exception):
    """Transient provider failure (5xx, network error) - safe to retry."""


class RateLimitError(ProviderError):
    """429 from a provider. May carry a Retry-After hint in seconds."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderClientError(Exception):
    """Non-retryable 4xx (bad auth, bad request, unknown model, ...)."""


class ProviderConfigError(Exception):
    """Provider is misconfigured: unknown name, bad quotas.yaml entry, or missing API key."""
