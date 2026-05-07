"""Shared Pydantic models for the target system.

Kept separate so both rag_chatbot and conversation can import them
without creating a circular dependency.
"""

from __future__ import annotations

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
