from __future__ import annotations

import asyncio
import time
from typing import Protocol


class ThrottleBackend(Protocol):
    """429-aware queueing gate: calls to a cooling-down provider wait instead of piling on."""

    async def is_open(self, provider: str) -> float:
        """Seconds to wait before `provider` is open again (0 if open now)."""
        ...

    async def trip(self, provider: str, retry_after: float | None) -> None:
        """Record a cooldown for `provider`, e.g. after a 429 response."""
        ...


class InMemoryThrottle:
    """Per-process cooldown gate.

    Correct for free-tier single-replica deploys (Render free / HF Spaces / one
    container). If a project ever runs multiple replicas, swap this for a
    redis-backed ThrottleBackend implementation - the Protocol above is the seam,
    don't build the redis version until it's actually needed.
    """

    def __init__(self, default_cooldown: float = 5.0) -> None:
        self._default_cooldown = default_cooldown
        self._cooldown_until: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_open(self, provider: str) -> float:
        async with self._lock:
            wait = self._cooldown_until.get(provider, 0.0) - time.monotonic()
            return max(wait, 0.0)

    async def trip(self, provider: str, retry_after: float | None) -> None:
        async with self._lock:
            cooldown = retry_after if retry_after is not None else self._default_cooldown
            self._cooldown_until[provider] = time.monotonic() + cooldown
