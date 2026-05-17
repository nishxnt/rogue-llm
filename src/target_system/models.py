"""Shared Pydantic models for the target system.

Kept separate so both rag_chatbot and conversation can import them
without creating a circular dependency.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    content: str
    source: str
    doc_id: str
    score: float


class Response(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: float
    tokens_used: int
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    guardrail_decision: (
        Literal[
            "allowed",
            "blocked_l1",
            "blocked_l2",
            "classifier_unavailable_blocked",
            "classifier_unavailable_passthrough",
            "blocked_l3_pii",
            "blocked_l3_leak",
            "blocked_l3_unsafe_pattern",
            "modified_l3",
        ]
        | None
    ) = None
    guardrail_decision_layer: int | None = None
    guardrail_evidence: dict[str, object] = Field(default_factory=dict)
    base_target_called: bool | None = None
    guardrail_timestamp: str | None = None
