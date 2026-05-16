"""Shared Groq client helpers for key rotation and rate-limit probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import structlog
from groq import AsyncGroq, Groq, RateLimitError
from httpx import Headers

from src.config import Settings, get_settings

log = structlog.get_logger()

PRE_FLIGHT_MIN_COMBINED_RPD = 50
PRE_FLIGHT_MIN_COMBINED_TPM = 8_000


@dataclass(frozen=True)
class GroqCredential:
    """One configured Groq API key."""

    name: str
    api_key: str


@dataclass(frozen=True)
class GroqPreflightBudget:
    """Rate-limit snapshot for one configured Groq key."""

    key_name: str
    remaining_requests_per_day: int | None
    reset_requests: str | None
    remaining_tokens_per_minute: int | None
    reset_tokens: str | None
    raw_headers: dict[str, str]


class SyncGroqFactory(Protocol):
    def __call__(self, *, api_key: str) -> Any:
        """Return a configured synchronous Groq client."""


class AsyncGroqFactory(Protocol):
    def __call__(self, *, api_key: str) -> Any:
        """Return a configured asynchronous Groq client."""


def configured_groq_credentials(settings: Settings | None = None) -> list[GroqCredential]:
    """Return configured Groq API keys in primary-then-secondary order."""
    active_settings = settings or get_settings()
    credentials = [
        GroqCredential("primary", active_settings.groq_api_key.get_secret_value()),
    ]
    secondary = active_settings.groq_api_key_2
    if secondary is not None:
        secondary_value = secondary.get_secret_value().strip()
        if secondary_value:
            credentials.append(GroqCredential("secondary", secondary_value))
    return credentials


class GroqClientManager:
    """Groq chat-completion helper with transparent secondary-key fallback."""

    def __init__(
        self,
        *,
        credentials: list[GroqCredential] | None = None,
        sync_client_factory: SyncGroqFactory = Groq,
        async_client_factory: AsyncGroqFactory = AsyncGroq,
    ) -> None:
        self._credentials = credentials or configured_groq_credentials()
        self._sync_client_factory = sync_client_factory
        self._async_client_factory = async_client_factory
        self._sync_clients: dict[str, Any] = {}
        self._async_clients: dict[str, Any] = {}

    def create_chat_completion(self, **kwargs: Any) -> Any:
        """Create one sync chat completion, retrying on the secondary key after 429."""
        original_error: RateLimitError | None = None
        for index, credential in enumerate(self._credentials):
            try:
                response = self._sync_client(credential).chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if original_error is None:
                    original_error = exc
                if index == len(self._credentials) - 1:
                    raise original_error from exc
                log.warning(
                    "Groq rate limit on key, retrying next key",
                    key_name=credential.name,
                    model=str(kwargs.get("model", "unknown")),
                )
                continue
            log.info(
                "Groq chat completion succeeded",
                key_name=credential.name,
                model=str(kwargs.get("model", "unknown")),
            )
            return response
        if original_error is None:  # pragma: no cover
            raise RuntimeError("No Groq credentials configured")
        raise original_error

    async def acreate_chat_completion(self, **kwargs: Any) -> Any:
        """Create one async chat completion, retrying on the secondary key after 429."""
        original_error: RateLimitError | None = None
        for index, credential in enumerate(self._credentials):
            try:
                response = await self._async_client(credential).chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if original_error is None:
                    original_error = exc
                if index == len(self._credentials) - 1:
                    raise original_error from exc
                log.warning(
                    "Groq rate limit on key, retrying next key",
                    key_name=credential.name,
                    model=str(kwargs.get("model", "unknown")),
                )
                continue
            log.info(
                "Groq chat completion succeeded",
                key_name=credential.name,
                model=str(kwargs.get("model", "unknown")),
            )
            return response
        if original_error is None:  # pragma: no cover
            raise RuntimeError("No Groq credentials configured")
        raise original_error

    async def probe_rate_limits(self, *, model: str) -> list[GroqPreflightBudget]:
        """Make one tiny call per key and return the exposed Groq rate-limit headers."""
        budgets: list[GroqPreflightBudget] = []
        for credential in self._credentials:
            client = self._async_client(credential)
            try:
                raw_response = await client.with_raw_response.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "ping"}],
                )
                await raw_response.parse()
                headers = raw_response.headers
            except RateLimitError as exc:
                headers = (
                    exc.response.headers
                    if getattr(exc, "response", None) is not None
                    else Headers()
                )
            budgets.append(_budget_from_headers(credential.name, headers))
        return budgets

    def _sync_client(self, credential: GroqCredential) -> Any:
        if credential.name not in self._sync_clients:
            self._sync_clients[credential.name] = self._sync_client_factory(
                api_key=credential.api_key
            )
        return self._sync_clients[credential.name]

    def _async_client(self, credential: GroqCredential) -> Any:
        if credential.name not in self._async_clients:
            self._async_clients[credential.name] = self._async_client_factory(
                api_key=credential.api_key
            )
        return self._async_clients[credential.name]


def combined_remaining_requests_per_day(budgets: list[GroqPreflightBudget]) -> int:
    """Return the summed remaining daily-request budget across configured keys."""
    return sum(budget.remaining_requests_per_day or 0 for budget in budgets)


def combined_remaining_tokens_per_minute(budgets: list[GroqPreflightBudget]) -> int:
    """Return the summed remaining TPM budget across configured keys."""
    return sum(budget.remaining_tokens_per_minute or 0 for budget in budgets)


def _budget_from_headers(key_name: str, headers: Headers) -> GroqPreflightBudget:
    raw_headers = {
        key.lower(): value
        for key, value in headers.items()
        if key.lower().startswith("x-ratelimit-")
    }
    return GroqPreflightBudget(
        key_name=key_name,
        remaining_requests_per_day=_parse_int(raw_headers.get("x-ratelimit-remaining-requests")),
        reset_requests=raw_headers.get("x-ratelimit-reset-requests"),
        remaining_tokens_per_minute=_parse_int(raw_headers.get("x-ratelimit-remaining-tokens")),
        reset_tokens=raw_headers.get("x-ratelimit-reset-tokens"),
        raw_headers=raw_headers,
    )


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
