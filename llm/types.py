from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: float = 0.0


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
