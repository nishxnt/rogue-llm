"""Layer 2 safety classifier backed by GPT-OSS-Safeguard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config import get_settings
from src.guardrails.reasons import GuardrailBlock
from src.pipeline.groq_client import GroqClientManager

SafetyDecision = Literal["allow", "block"]
_CLASSIFIER_MAX_TOKENS = 512
_CLASSIFIER_JSON_RETRIES = 3


class AsyncSafetyClient(Protocol):
    async def acreate_chat_completion(self, **kwargs: object) -> object:
        """Create one structured safety classification completion."""

    async def aclose(self) -> None:
        """Close any held async resources."""


class ChatCompletionResponse(Protocol):
    choices: list[Any]


class SafetyPolicyMetadata(BaseModel):
    name: str
    version: str
    updated_at: str
    model: str
    default_threshold: float = Field(ge=0.0, le=1.0)


class SafetyPolicyExample(BaseModel):
    category: str
    decision: SafetyDecision
    prompt: str
    rationale: str


class SafetyPolicy(BaseModel):
    metadata: SafetyPolicyMetadata
    assistant_should: list[str]
    assistant_must_not: list[str]
    categories: dict[str, str]
    examples: list[SafetyPolicyExample]


class SafetyClassification(BaseModel):
    decision: SafetyDecision
    confidence: float = Field(ge=0.0, le=1.0)
    matched_policy_category: str
    reason: str

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "high": 0.9,
                "medium": 0.5,
                "low": 0.1,
            }
            if normalized in mapping:
                return mapping[normalized]
            if normalized.endswith("%"):
                normalized = normalized[:-1]
            try:
                return float(normalized)
            except ValueError:
                return value
        return value


class SafetyClassifier:
    """Groq-backed Layer 2 safety classifier with load-once YAML policy."""

    def __init__(
        self,
        *,
        policy_path: Path | str,
        model: str | None = None,
        threshold: float = 0.7,
        client_manager: AsyncSafetyClient | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.policy_path = Path(policy_path)
        self.policy = load_policy(self.policy_path)
        self.model = model or get_settings().safety_model
        self.threshold = threshold
        self._client_manager = client_manager or GroqClientManager()
        self._policy_text = _policy_to_prompt(self.policy)

    async def classify(self, prompt: str) -> SafetyClassification:
        """Return the raw Layer 2 classification."""
        last_error: Exception | None = None
        for _ in range(_CLASSIFIER_JSON_RETRIES):
            try:
                response = cast(
                    "ChatCompletionResponse",
                    await self._request_structured_classification(prompt),
                )
                content = str(response.choices[0].message.content or "")
                return SafetyClassification.model_validate(json.loads(content))
            except Exception as exc:
                if not _is_classifier_contract_failure(exc):
                    raise
                last_error = exc
        if last_error is None:  # pragma: no cover - defensive only
            raise RuntimeError("classifier failed without an exception")
        raise last_error

    async def inspect(self, prompt: str) -> GuardrailBlock | None:
        """Return a Layer 2 block if the classifier exceeds the block threshold."""
        try:
            classification = await self.classify(prompt)
        except Exception as exc:
            if not _is_classifier_contract_failure(exc):
                raise
            return GuardrailBlock(
                decision="classifier_unavailable_blocked",
                decision_layer=2,
                evidence={
                    "reason": "classifier_unavailable_after_retries",
                    "failure_mode": _classifier_failure_mode(exc),
                    "retry_count": _CLASSIFIER_JSON_RETRIES,
                    "threshold": self.threshold,
                    "model": self.model,
                    "fallback_mode": "fail_closed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        if classification.decision != "block" or classification.confidence <= self.threshold:
            return None
        return GuardrailBlock(
            decision="blocked_l2",
            decision_layer=2,
            evidence={
                "classifier_confidence": classification.confidence,
                "matched_policy_category": classification.matched_policy_category,
                "reason": classification.reason,
                "threshold": self.threshold,
                "model": self.model,
            },
        )

    def fingerprint(self) -> str:
        """Return a stable fingerprint for cache invalidation."""
        policy_hash = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        return hashlib.sha256(f"{self.model}|{self.threshold}|{policy_hash}".encode()).hexdigest()

    async def aclose(self) -> None:
        await self._client_manager.aclose()

    async def _request_structured_classification(self, prompt: str) -> object:
        return await self._client_manager.acreate_chat_completion(
            model=self.model,
            temperature=0.0,
            max_tokens=_CLASSIFIER_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a safety classifier for a cybersecurity-domain RAG assistant. "
                        "Follow the supplied policy exactly. Return JSON only with keys "
                        "decision, confidence, matched_policy_category, and reason."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Policy:\n{self._policy_text}\n\n"
                        f"User prompt:\n{prompt}\n\n"
                        "Classify whether this prompt should be allowed or blocked."
                    ),
                },
            ],
        )


def load_policy(path: Path | str) -> SafetyPolicy:
    with Path(path).open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return SafetyPolicy.model_validate(loaded)


def _policy_to_prompt(policy: SafetyPolicy) -> str:
    lines: list[str] = [
        f"name: {policy.metadata.name}",
        f"version: {policy.metadata.version}",
        "assistant_should:",
        *[f"- {item}" for item in policy.assistant_should],
        "assistant_must_not:",
        *[f"- {item}" for item in policy.assistant_must_not],
        "categories:",
    ]
    for key, value in policy.categories.items():
        lines.append(f"- {key}: {value}")
    lines.append("examples:")
    for example in policy.examples:
        lines.extend(
            [
                f"- category: {example.category}",
                f"  decision: {example.decision}",
                f"  prompt: {example.prompt}",
                f"  rationale: {example.rationale}",
            ]
        )
    return "\n".join(lines)


def _is_classifier_contract_failure(exc: Exception) -> bool:
    if isinstance(exc, (json.JSONDecodeError, ValidationError)):
        return True
    return _is_json_validate_failed(exc)


def _classifier_failure_mode(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "json_decode_failed"
    if isinstance(exc, ValidationError):
        return "structured_output_validation_failed"
    return "json_validate_failed"


def _is_json_validate_failed(exc: Exception) -> bool:
    details = str(exc).lower()
    return "json_validate_failed" in details
