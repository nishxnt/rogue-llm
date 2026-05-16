"""Shared guardrail decision models and refusal strings."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

GuardrailDecision = Literal[
    "allowed",
    "blocked_l1",
    "blocked_l2",
    "blocked_l3_pii",
    "blocked_l3_leak",
    "blocked_l3_unsafe_pattern",
    "modified_l3",
]

DecisionLayer = Literal[1, 2, 3]

REFUSAL_MESSAGES: dict[GuardrailDecision, str] = {
    "blocked_l1": "I can't help with that request.",
    "blocked_l2": "I'm not able to assist with that.",
    "blocked_l3_pii": "I can't share that information.",
    "blocked_l3_leak": "I can't disclose that.",
    "blocked_l3_unsafe_pattern": "I can't provide that response.",
}


class GuardrailDecisionRecord(BaseModel):
    """Per-attack guardrail decision record persisted for Phase 5 analysis."""

    attack_id: str | None = None
    decision: GuardrailDecision
    decision_layer: DecisionLayer | None = None
    evidence: dict[str, object] = Field(default_factory=dict)
    base_target_called: bool
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class GuardrailBlock(BaseModel):
    """Internal short-circuit signal emitted by one guardrail layer."""

    decision: GuardrailDecision
    decision_layer: DecisionLayer
    evidence: dict[str, object] = Field(default_factory=dict)


def refusal_message_for(decision: GuardrailDecision) -> str:
    """Return the external refusal text for a blocking guardrail decision."""
    try:
        return REFUSAL_MESSAGES[decision]
    except KeyError as exc:  # pragma: no cover - defensive only
        raise ValueError(f"no refusal message configured for decision={decision}") from exc
