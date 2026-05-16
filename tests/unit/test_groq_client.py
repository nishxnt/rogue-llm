from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from groq import RateLimitError

from src.pipeline.groq_client import (
    GroqClientManager,
    GroqCredential,
    combined_remaining_requests_per_day,
    combined_remaining_tokens_per_minute,
)


def _rate_limit_error(headers: dict[str, str] | None = None) -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


class FakeSyncCompletions:
    def __init__(self, actions: list[object]) -> None:
        self._actions = list(actions)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class FakeAsyncCompletions:
    def __init__(self, actions: list[object]) -> None:
        self._actions = list(actions)
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class FakeRawResponse:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

    async def parse(self) -> object:
        return _response("ok")


class FakeSyncClient:
    def __init__(self, actions: list[object]) -> None:
        completions = FakeSyncCompletions(actions)
        self.chat = SimpleNamespace(completions=completions)


class FakeAsyncClient:
    def __init__(self, actions: list[object], raw_actions: list[object] | None = None) -> None:
        completions = FakeAsyncCompletions(actions)
        raw_completions = FakeAsyncCompletions(raw_actions or [])
        self.chat = SimpleNamespace(completions=completions)
        self.with_raw_response = SimpleNamespace(
            chat=SimpleNamespace(completions=raw_completions),
        )


def test_single_key_sync_behaves_as_before() -> None:
    primary_client = FakeSyncClient([_response("primary-ok")])
    manager = GroqClientManager(
        credentials=[GroqCredential("primary", "pk1")],
        sync_client_factory=lambda *, api_key: primary_client,
    )

    response = manager.create_chat_completion(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "ping"}],
    )

    assert response.choices[0].message.content == "primary-ok"
    assert len(primary_client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_dual_key_async_falls_back_to_secondary_on_primary_429() -> None:
    primary_client = FakeAsyncClient([_rate_limit_error()])
    secondary_client = FakeAsyncClient([_response("secondary-ok")])
    manager = GroqClientManager(
        credentials=[
            GroqCredential("primary", "pk1"),
            GroqCredential("secondary", "pk2"),
        ],
        async_client_factory=lambda *, api_key: {
            "pk1": primary_client,
            "pk2": secondary_client,
        }[api_key],
    )

    response = await manager.acreate_chat_completion(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "ping"}],
    )

    assert response.choices[0].message.content == "secondary-ok"
    assert len(primary_client.chat.completions.calls) == 1
    assert len(secondary_client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_dual_key_async_raises_original_primary_429_if_both_keys_exhausted() -> None:
    primary_error = _rate_limit_error()
    secondary_error = _rate_limit_error()
    manager = GroqClientManager(
        credentials=[
            GroqCredential("primary", "pk1"),
            GroqCredential("secondary", "pk2"),
        ],
        async_client_factory=lambda *, api_key: {
            "pk1": FakeAsyncClient([primary_error]),
            "pk2": FakeAsyncClient([secondary_error]),
        }[api_key],
    )

    with pytest.raises(RateLimitError) as exc_info:
        await manager.acreate_chat_completion(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "ping"}],
        )

    assert exc_info.value is primary_error


@pytest.mark.asyncio
async def test_probe_rate_limits_reads_headers_per_key() -> None:
    manager = GroqClientManager(
        credentials=[
            GroqCredential("primary", "pk1"),
            GroqCredential("secondary", "pk2"),
        ],
        async_client_factory=lambda *, api_key: {
            "pk1": FakeAsyncClient(
                [],
                raw_actions=[
                    FakeRawResponse(
                        {
                            "x-ratelimit-remaining-requests": "945",
                            "x-ratelimit-reset-requests": "1h",
                            "x-ratelimit-remaining-tokens": "12000",
                            "x-ratelimit-reset-tokens": "6m",
                        }
                    )
                ],
            ),
            "pk2": FakeAsyncClient(
                [],
                raw_actions=[
                    _rate_limit_error(
                        {
                            "x-ratelimit-remaining-requests": "944",
                            "x-ratelimit-reset-requests": "2h",
                            "x-ratelimit-remaining-tokens": "8000",
                            "x-ratelimit-reset-tokens": "11m",
                        }
                    )
                ],
            ),
        }[api_key],
    )

    budgets = await manager.probe_rate_limits(model="openai/gpt-oss-120b")

    assert [
        (
            budget.key_name,
            budget.remaining_requests_per_day,
            budget.reset_requests,
            budget.remaining_tokens_per_minute,
            budget.reset_tokens,
        )
        for budget in budgets
    ] == [
        ("primary", 945, "1h", 12000, "6m"),
        ("secondary", 944, "2h", 8000, "11m"),
    ]
    assert combined_remaining_requests_per_day(budgets) == 1889
    assert combined_remaining_tokens_per_minute(budgets) == 20000
