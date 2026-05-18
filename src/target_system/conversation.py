"""Conversation abstraction for the RAG target system.

Designed for multi-turn extension: v1 uses single-turn queries, but
to_langchain_messages() provides the hook for v1.1 multi-turn support
without a refactor.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from src.target_system.models import Response  # noqa: TCH001


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    response: Response | None = None


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    turns: list[Turn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def add_user_turn(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content))

    def add_assistant_turn(self, content: str, response: Response) -> None:
        self.turns.append(Turn(role="assistant", content=content, response=response))

    def to_langchain_messages(self) -> list[BaseMessage]:
        """Return conversation history as LangChain messages for multi-turn use."""
        msgs: list[BaseMessage] = []
        for turn in self.turns:
            if turn.role == "user":
                msgs.append(HumanMessage(content=turn.content))
            else:
                msgs.append(AIMessage(content=turn.content))
        return msgs
