import pytest

from src.pipeline.rate_limiter import ConcurrencyLimiter, TokenBucketRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.asyncio
async def test_token_bucket_allows_configured_burst() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate_per_minute=30, burst=2, clock=clock, sleeper=clock.sleep)

    await limiter.acquire()
    await limiter.acquire()

    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_token_bucket_waits_after_burst_is_consumed() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate_per_minute=30, burst=2, clock=clock, sleeper=clock.sleep)

    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()

    assert clock.sleeps == [2.0]


@pytest.mark.asyncio
async def test_token_bucket_refills_over_time() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(rate_per_minute=30, burst=2, clock=clock, sleeper=clock.sleep)

    await limiter.acquire()
    await limiter.acquire()
    clock.now += 4.0
    await limiter.acquire()
    await limiter.acquire()

    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_concurrency_limiter_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError):
        ConcurrencyLimiter(0)


@pytest.mark.asyncio
async def test_concurrency_limiter_context_manager() -> None:
    limiter = ConcurrencyLimiter(1)
    entered = False

    async with limiter:
        entered = True

    assert entered is True
