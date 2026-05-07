"""Integration tests for RAGChatbot — requires live GROQ_API_KEY and built index.

Run with:
    uv run pytest -m integration tests/integration/test_rag_chatbot.py -v

These tests are skipped by default (pytest.ini addopts: -m 'not integration').
"""

from __future__ import annotations

from typing import Any

import pytest

from src.target_system.models import Response
from src.target_system.rag_chatbot import RAGChatbot

pytestmark = pytest.mark.integration

_CASES: list[dict[str, Any]] = [
    {
        "question": "What is SQL injection?",
        "expect_in_answer": ["sql", "injection"],
        "expect_chunk_term": "injection",
    },
    {
        "question": "What does CVE stand for?",
        "expect_in_answer": ["common", "vulnerabilities"],
        "expect_chunk_term": None,
    },
    {
        "question": "What is prompt injection in LLMs?",
        "expect_in_answer": ["prompt", "injection"],
        "expect_chunk_term": "prompt",
    },
    {
        "question": "What is cross-site scripting?",
        "expect_in_answer": ["script"],
        "expect_chunk_term": "script",
    },
    {
        "question": "Explain CVSS scoring",
        "expect_in_answer": ["score", "severity"],
        "expect_chunk_term": None,
    },
]


@pytest.fixture(scope="module")
def chatbot() -> RAGChatbot:
    return RAGChatbot()


@pytest.mark.parametrize("case", _CASES, ids=[c["question"][:40] for c in _CASES])
def test_known_question_returns_non_empty_answer(chatbot: RAGChatbot, case: dict) -> None:
    response = chatbot.query(case["question"])
    assert isinstance(response, Response)
    assert len(response.answer.strip()) > 20


@pytest.mark.parametrize("case", _CASES, ids=[c["question"][:40] for c in _CASES])
def test_known_question_retrieves_k_chunks(chatbot: RAGChatbot, case: dict) -> None:
    response = chatbot.query(case["question"])
    assert len(response.retrieved_chunks) == 4


@pytest.mark.parametrize(
    "case",
    [c for c in _CASES if c["expect_chunk_term"]],
    ids=[c["question"][:40] for c in _CASES if c["expect_chunk_term"]],
)
def test_known_question_retrieves_relevant_chunk(chatbot: RAGChatbot, case: dict) -> None:
    response = chatbot.query(case["question"])
    term = case["expect_chunk_term"].lower()
    chunk_texts = " ".join(c.content.lower() for c in response.retrieved_chunks)
    assert term in chunk_texts, f"Expected '{term}' in retrieved chunks"


@pytest.mark.parametrize("case", _CASES, ids=[c["question"][:40] for c in _CASES])
def test_known_question_answer_contains_expected_terms(chatbot: RAGChatbot, case: dict) -> None:
    response = chatbot.query(case["question"])
    answer_lower = response.answer.lower()
    for term in case["expect_in_answer"]:
        assert term in answer_lower, f"Expected '{term}' in answer"


def test_response_has_positive_latency(chatbot: RAGChatbot) -> None:
    response = chatbot.query("What is a firewall?")
    assert response.latency_ms > 0


def test_response_conversation_id_is_non_empty(chatbot: RAGChatbot) -> None:
    response = chatbot.query("What is authentication?")
    assert len(response.conversation_id) > 0


def test_chunk_scores_are_between_zero_and_one(chatbot: RAGChatbot) -> None:
    response = chatbot.query("What is access control?")
    for chunk in response.retrieved_chunks:
        assert 0.0 <= chunk.score <= 1.0, f"Score out of range: {chunk.score}"
