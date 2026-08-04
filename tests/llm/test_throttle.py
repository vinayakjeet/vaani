from __future__ import annotations

import asyncio

import pytest

import llm.throttle as throttle_module
from llm.throttle import InMemoryThrottle


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(throttle_module.time, "monotonic", clock)
    return clock


async def test_is_open_reports_zero_before_any_trip():
    throttle = InMemoryThrottle()
    assert await throttle.is_open("groq") == 0.0


async def test_trip_with_retry_after_sets_exact_cooldown(fake_clock):
    throttle = InMemoryThrottle(default_cooldown=5.0)
    await throttle.trip("groq", retry_after=10.0)
    assert await throttle.is_open("groq") == 10.0
    fake_clock.t = 10.0
    assert await throttle.is_open("groq") == 0.0


async def test_trip_without_retry_after_uses_default_cooldown(fake_clock):
    throttle = InMemoryThrottle(default_cooldown=5.0)
    await throttle.trip("groq", retry_after=None)
    assert await throttle.is_open("groq") == 5.0


async def test_trip_is_per_provider(fake_clock):
    throttle = InMemoryThrottle()
    await throttle.trip("groq", retry_after=10.0)
    assert await throttle.is_open("cerebras") == 0.0


async def test_concurrent_trips_do_not_corrupt_state(fake_clock):
    """Guards the check-then-act gap: many coroutines tripping the same provider
    concurrently must leave one consistent cooldown, never a torn/partial value."""
    throttle = InMemoryThrottle()
    await asyncio.gather(*(throttle.trip("groq", retry_after=float(i)) for i in range(1, 6)))
    wait = await throttle.is_open("groq")
    assert wait in {1.0, 2.0, 3.0, 4.0, 5.0}


async def test_concurrent_is_open_calls_agree_during_cooldown(fake_clock):
    throttle = InMemoryThrottle()
    await throttle.trip("groq", retry_after=10.0)
    waits = await asyncio.gather(*(throttle.is_open("groq") for _ in range(10)))
    assert all(w == 10.0 for w in waits)
