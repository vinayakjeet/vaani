from __future__ import annotations

import random
from collections.abc import Callable
from typing import TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from llm.types import ProviderError

T = TypeVar("T")


def backoff_seconds(attempt: int, min_wait: float = 1.0, max_wait: float = 30.0) -> float:
    """The same curve `retry_with_backoff` uses, for callers that cannot use it.

    A streaming call cannot be retried by a decorator. Once a token has been
    handed to the caller the request is no longer repeatable, and a decorator only
    sees the whole call succeed or fail, so the streaming path in `ChatClient`
    runs its own loop and asks for the wait here. Written out rather than reaching
    into tenacity's internals, and deliberately identical to
    `wait_random_exponential`: jitter uniform between zero and the exponential
    bound, so a burst of callers does not retry in lockstep.
    """
    return random.uniform(0, min(max_wait, min_wait * 2**attempt))


def retry_with_backoff(
    max_attempts: int = 5, min_wait: float = 1.0, max_wait: float = 30.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Exponential backoff + jitter, retrying only transient ProviderError/RateLimitError.

    ProviderClientError (bad auth, bad request, ...) is never retried - retrying a
    request that's wrong by construction just burns quota faster.
    """
    return retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=min_wait, max=max_wait),
        reraise=True,
    )
