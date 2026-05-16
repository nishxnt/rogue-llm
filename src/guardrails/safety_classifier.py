"""Layer 2 safety classifier backed by GPT-OSS-Safeguard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml
from pydantic import BaseModel, Field

from src.config import get_settings
from src.guardrails.reasons import GuardrailBlock
from src.pipeline.groq_client import GroqClientManager

SafetyDecision = Literal["allow", "block"]


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
        response = cast(
            "ChatCompletionResponse",
            await self._client_manager.acreate_chat_completion(
                model=self.model,
                temperature=0.0,
                max_tokens=256,
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
            ),
        )
        content = str(response.choices[0].message.content or "")
        return SafetyClassification.model_validate(json.loads(content))

    async def inspect(self, prompt: str) -> GuardrailBlock | None:
        """Return a Layer 2 block if the classifier exceeds the block threshold."""
        classification = await self.classify(prompt)
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
