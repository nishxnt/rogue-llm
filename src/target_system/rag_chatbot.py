"""RAG chatbot — the target system for adversarial testing.

Public interface:
    chatbot = RAGChatbot()
    response = chatbot.query("What is SQL injection?")
    response = await chatbot.aquery("What is SQL injection?")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.target_system.conversation import Conversation
from src.target_system.prompts import SYSTEM_PROMPT

log = structlog.get_logger()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_INDEX_DIR = _PROJECT_ROOT / "data" / "index" / "faiss_index"


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


def _format_docs(docs: list[Any]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class RAGChatbot:
    """Deliberately naive RAG chatbot — the red-team target system."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model,
            encode_kwargs={"batch_size": 32, "show_progress_bar": False},
        )
        self._vectorstore = FAISS.load_local(
            str(_INDEX_DIR),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        self._llm = ChatGroq(
            model_name=self._settings.target_model,
            temperature=0.1,
            api_key=self._settings.groq_api_key.get_secret_value(),
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self._chain: Any = (
            {
                "context": self._retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | self._prompt
            | self._llm
            | StrOutputParser()
        )
        log.info(
            "RAGChatbot initialised",
            target_model=self._settings.target_model,
            index_path=str(_INDEX_DIR),
        )

    def _build_response(
        self,
        question: str,
        answer: str,
        start_time: float,
        raw_docs: list[Any],
        raw_scores: list[tuple[Any, float]],
        conversation_id: str,
        tokens_used: int,
    ) -> Response:
        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", ""),
                doc_id=doc.metadata.get("doc_id", ""),
                score=float(score),
            )
            for doc, score in raw_scores
        ]
        return Response(
            answer=answer,
            retrieved_chunks=chunks,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            tokens_used=tokens_used,
            conversation_id=conversation_id,
        )

    def query(self, prompt: str, conversation: Conversation | None = None) -> Response:
        """Run a single-turn query and return a structured Response."""
        conv = conversation or Conversation()
        conv.add_user_turn(prompt)
        start = time.perf_counter()

        # Retrieve with scores so we can populate RetrievedChunk.score.
        docs_with_scores: list[tuple[Any, float]] = self._vectorstore.similarity_search_with_score(
            prompt, k=4
        )
        docs = [doc for doc, _ in docs_with_scores]
        context = _format_docs(docs)

        messages = self._prompt.format_messages(context=context, question=prompt)
        raw = self._llm.invoke(messages)
        answer: str = str(raw.content)
        tokens_used: int = 0
        usage = getattr(raw, "response_metadata", {}).get("usage", {}) or {}
        tokens_used = int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

        response = self._build_response(
            question=prompt,
            answer=answer,
            start_time=start,
            raw_docs=docs,
            raw_scores=docs_with_scores,
            conversation_id=conv.id,
            tokens_used=tokens_used,
        )
        conv.add_assistant_turn(answer, response)
        return response

    async def aquery(self, prompt: str, conversation: Conversation | None = None) -> Response:
        """Async single-turn query."""
        conv = conversation or Conversation()
        conv.add_user_turn(prompt)
        start = time.perf_counter()

        docs_with_scores: list[tuple[Any, float]] = self._vectorstore.similarity_search_with_score(
            prompt, k=4
        )
        docs = [doc for doc, _ in docs_with_scores]
        context = _format_docs(docs)

        messages = self._prompt.format_messages(context=context, question=prompt)
        raw = await self._llm.ainvoke(messages)
        answer_async: str = str(raw.content)
        tokens_used_async = 0
        usage = getattr(raw, "response_metadata", {}).get("usage", {}) or {}
        tokens_used_async = int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

        response = self._build_response(
            question=prompt,
            answer=answer_async,
            start_time=start,
            raw_docs=docs,
            raw_scores=docs_with_scores,
            conversation_id=conv.id,
            tokens_used=tokens_used_async,
        )
        conv.add_assistant_turn(answer_async, response)
        return response
