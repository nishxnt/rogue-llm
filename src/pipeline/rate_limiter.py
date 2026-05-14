"""Async rate limiting for Groq target-model calls."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class TokenBucketRateLimiter:
    """Async token bucket with bounded burst capacity."""

    def __init__(
        self,
        *,
        rate_per_minute: int = 30,
        burst: int = 2,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")

        self.rate_per_second = rate_per_minute / 60.0
        self.capacity = float(burst)
        self._tokens = float(burst)
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or asyncio.sleep
        self._updated_at = self._clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until one request token is available."""
        while True:
            async with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated_at)
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
                self._updated_at = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                missing = 1.0 - self._tokens
                wait_seconds = missing / self.rate_per_second

            await self._sleeper(wait_seconds)


class ConcurrencyLimiter:
    """Small wrapper around ``asyncio.Semaphore`` for runner readability."""

    def __init__(self, concurrency: int = 5) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self._semaphore = asyncio.Semaphore(concurrency)

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(self, *_: object) -> None:
        self._semaphore.release()
