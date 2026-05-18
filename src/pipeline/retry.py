"""Retry helpers for transient Groq API failures."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")

DEFAULT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


class RetryExhaustedError(RuntimeError):
    """Raised after all retry attempts are exhausted."""

    def __init__(self, message: str, *, last_error: BaseException) -> None:
        super().__init__(message)
        self.last_error = last_error


async def retry_transient(
    operation: Callable[[], Awaitable[T]],
    *,
    backoff_seconds: tuple[float, ...] = DEFAULT_BACKOFF_SECONDS,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run an async operation with capped retry for 429 and 5xx failures."""
    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            if not _is_retryable(exc) or attempt >= len(backoff_seconds):
                raise RetryExhaustedError(
                    f"operation failed after {attempt} retries: {exc}",
                    last_error=exc,
                ) from exc

            retry_after = _retry_after_seconds(exc)
            wait_seconds = retry_after if retry_after is not None else backoff_seconds[attempt]
            attempt += 1
            await sleeper(wait_seconds)


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    return isinstance(status_code, int) and 500 <= status_code <= 599


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
    if headers is None:
        return None

    value = None
    if hasattr(headers, "get"):
        value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        retry_after = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, retry_after)
