"""Unit tests for RAGChatbot, Response, RetrievedChunk, and Conversation.

All LLM and vector store calls are mocked — no API key required.
"""

from __future__ import annotations

import unittest.mock as mock
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.target_system.conversation import Conversation, Turn
from src.target_system.models import Response, RetrievedChunk
from src.target_system.rag_chatbot import RAGChatbot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_doc(content: str, source: str = "nvd", doc_id: str = "CVE-2024-00001") -> Any:
    doc = mock.MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "doc_id": doc_id}
    return doc


def _make_chatbot_with_mocks() -> tuple[RAGChatbot, mock.MagicMock, mock.MagicMock]:
    """Return (chatbot, mock_vectorstore, mock_llm) with embeddings and FAISS patched."""
    with (
        mock.patch("src.target_system.rag_chatbot.HuggingFaceEmbeddings"),
        mock.patch("src.target_system.rag_chatbot.FAISS") as mock_faiss_cls,
    ):
        mock_vs = mock.MagicMock()
        mock_faiss_cls.load_local.return_value = mock_vs

        fake_docs = [
            _make_fake_doc("SQL injection exploits unvalidated input.", "nvd", "CVE-2024-00001")
        ]
        mock_vs.similarity_search_with_score.return_value = [(fake_docs[0], 0.92)]
        mock_vs.as_retriever.return_value = mock.MagicMock()

        mock_settings = mock.MagicMock()
        mock_settings.target_model = "llama-3.1-8b-instant"
        mock_settings.embedding_model = "all-MiniLM-L6-v2"
        mock_settings.groq_api_key.get_secret_value.return_value = "test-key"

        with mock.patch("src.target_system.rag_chatbot.ChatGroq") as mock_groq_cls:
            mock_llm = mock.MagicMock()
            mock_groq_cls.return_value = mock_llm

            chatbot = RAGChatbot(settings=mock_settings)
            chatbot._vectorstore = mock_vs
            chatbot._llm = mock_llm

    return chatbot, mock_vs, mock_llm


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


def test_query_returns_response_with_answer() -> None:
    chatbot, mock_vs, mock_llm = _make_chatbot_with_mocks()

    fake_ai_msg = mock.MagicMock(spec=AIMessage)
    fake_ai_msg.content = "SQL injection is an attack that exploits..."
    fake_ai_msg.response_metadata = {"usage": {"prompt_tokens": 120, "completion_tokens": 80}}
    mock_llm.invoke.return_value = fake_ai_msg

    response = chatbot.query("What is SQL injection?")

    assert isinstance(response, Response)
    assert len(response.answer) > 0
    assert "SQL injection" in response.answer


def test_query_returns_retrieved_chunks() -> None:
    chatbot, mock_vs, mock_llm = _make_chatbot_with_mocks()

    fake_ai_msg = mock.MagicMock(spec=AIMessage)
    fake_ai_msg.content = "Answer about SQL injection"
    fake_ai_msg.response_metadata = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    mock_llm.invoke.return_value = fake_ai_msg

    response = chatbot.query("What is SQL injection?")

    assert len(response.retrieved_chunks) == 1
    assert isinstance(response.retrieved_chunks[0], RetrievedChunk)
    assert response.retrieved_chunks[0].source == "nvd"
    assert response.retrieved_chunks[0].doc_id == "CVE-2024-00001"
    assert response.retrieved_chunks[0].score == pytest.approx(0.92)


def test_query_populates_tokens_used() -> None:
    chatbot, mock_vs, mock_llm = _make_chatbot_with_mocks()

    fake_ai_msg = mock.MagicMock(spec=AIMessage)
    fake_ai_msg.content = "An answer"
    fake_ai_msg.response_metadata = {"usage": {"prompt_tokens": 200, "completion_tokens": 100}}
    mock_llm.invoke.return_value = fake_ai_msg

    response = chatbot.query("Some question?")
    assert response.tokens_used == 300


def test_query_latency_ms_is_positive() -> None:
    chatbot, mock_vs, mock_llm = _make_chatbot_with_mocks()

    fake_ai_msg = mock.MagicMock(spec=AIMessage)
    fake_ai_msg.content = "An answer"
    fake_ai_msg.response_metadata = {}
    mock_llm.invoke.return_value = fake_ai_msg

    response = chatbot.query("Some question?")
    assert response.latency_ms > 0


# ---------------------------------------------------------------------------
# Conversation — single-turn
# ---------------------------------------------------------------------------


def test_conversation_single_turn_messages() -> None:
    conv = Conversation()
    mock_response = mock.MagicMock(spec=Response)

    conv.add_user_turn("What is XSS?")
    conv.add_assistant_turn("XSS is cross-site scripting.", mock_response)

    messages = conv.to_langchain_messages()
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "What is XSS?"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "XSS is cross-site scripting."


def test_conversation_multi_turn_message_ordering() -> None:
    conv = Conversation()
    mock_response = mock.MagicMock(spec=Response)

    conv.add_user_turn("First question")
    conv.add_assistant_turn("First answer", mock_response)
    conv.add_user_turn("Second question")
    conv.add_assistant_turn("Second answer", mock_response)

    messages = conv.to_langchain_messages()
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], HumanMessage)
    assert isinstance(messages[3], AIMessage)
    assert messages[2].content == "Second question"


def test_conversation_turn_stores_response() -> None:
    conv = Conversation()
    mock_response = mock.MagicMock(spec=Response)

    conv.add_user_turn("A question")
    conv.add_assistant_turn("An answer", mock_response)

    assistant_turn = conv.turns[1]
    assert isinstance(assistant_turn, Turn)
    assert assistant_turn.response is mock_response


def test_conversation_id_is_unique() -> None:
    c1 = Conversation()
    c2 = Conversation()
    assert c1.id != c2.id


def test_conversation_empty_messages() -> None:
    conv = Conversation()
    assert conv.to_langchain_messages() == []
