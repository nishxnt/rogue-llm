from typing import Any

import pytest

from src.pipeline.retry import RetryExhaustedError, retry_transient


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.headers = headers or {}


@pytest.mark.asyncio
async def test_retry_uses_backoff_sequence_before_success() -> None:
    calls = 0
    sleeps: list[float] = []

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 4:
            raise FakeHTTPError(429)
        return "ok"

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = await retry_transient(flaky, sleeper=sleep)

    assert result == "ok"
    assert calls == 4
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_retry_honors_retry_after_header() -> None:
    calls = 0
    sleeps: list[float] = []

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FakeHTTPError(429, headers={"Retry-After": "3.5"})
        return "ok"

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = await retry_transient(flaky, sleeper=sleep)

    assert result == "ok"
    assert sleeps == [3.5]


@pytest.mark.asyncio
async def test_retry_exhausts_after_three_retries() -> None:
    sleeps: list[float] = []

    async def always_fails() -> str:
        raise FakeHTTPError(500)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(RetryExhaustedError) as exc_info:
        await retry_transient(always_fails, sleeper=sleep)

    assert isinstance(exc_info.value.last_error, FakeHTTPError)
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_non_retryable_error_raises_without_sleep() -> None:
    sleeps: list[float] = []

    async def fails() -> str:
        raise ValueError("bad input")

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(RetryExhaustedError) as exc_info:
        await retry_transient(fails, sleeper=sleep)

    assert isinstance(exc_info.value.last_error, ValueError)
    assert sleeps == []


@pytest.mark.asyncio
async def test_retry_after_can_be_read_from_response_headers() -> None:
    class Response:
        headers = {"retry-after": "2"}

    class ResponseError(Exception):
        status_code = 503
        response: Any = Response()

    sleeps: list[float] = []
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ResponseError("unavailable")
        return "ok"

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    assert await retry_transient(flaky, sleeper=sleep) == "ok"
    assert sleeps == [2.0]
