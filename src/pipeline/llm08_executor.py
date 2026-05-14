"""Special execution path for LLM08 vector and embedding attacks."""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.target_system.models import Response, RetrievedChunk

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.documents import Document


class LLM08Execution(BaseModel):
    """Target response plus LLM08-specific retrieval diagnostics.

    ``llm08_retrieved_docs`` stores FAISS L2 ``distance`` values, where lower
    means more similar. It also includes an approximate ``similarity`` value
    computed as ``1 / (1 + distance)`` for downstream scoring convenience.
    """

    response: Response
    llm08_retrieved_docs: list[dict[str, Any]]
    llm08_checks: dict[str, Any]


class InMemoryFAISSContext:
    """Deep-copy a FAISS vectorstore so mutations never touch disk state."""

    def __init__(self, vectorstore: Any) -> None:
        self._source = vectorstore
        self.vectorstore: Any | None = None

    def __enter__(self) -> Any:
        self.vectorstore = copy.deepcopy(self._source)
        return self.vectorstore

    def __exit__(self, *_: object) -> None:
        self.vectorstore = None


class LLM08Executor:
    """Execute structured LLM08 entries against the target RAG system."""

    def __init__(self, target_system: Any, *, top_k: int = 4) -> None:
        self.target_system = target_system
        self.top_k = top_k

    async def execute(self, attack: dict[str, Any]) -> LLM08Execution:
        """Run an LLM08 attack and return response plus retrieval diagnostics."""
        strategy = str(attack["attack_strategy"])
        if strategy == "embedding_poisoning":
            return await self._execute_embedding_poisoning(attack)
        return await self._execute_standard_retrieval(attack)

    async def _execute_embedding_poisoning(self, attack: dict[str, Any]) -> LLM08Execution:
        query = str(attack.get("target_query") or attack["prompt_text"])
        source_vectorstore = self.target_system._vectorstore
        with InMemoryFAISSContext(source_vectorstore) as vectorstore:
            poisoned_metadata = dict(attack.get("poisoned_doc_metadata") or {})
            poisoned_doc_id = str(poisoned_metadata.get("doc_id") or f"{attack['id']}-poisoned")
            poisoned_metadata["doc_id"] = poisoned_doc_id
            poisoned_metadata.setdefault("source", "synthetic_poisoned")
            vectorstore.add_texts(
                [str(attack["poisoned_doc_content"])],
                metadatas=[poisoned_metadata],
            )
            response = await self._query_with_vectorstore(query, vectorstore)

        docs = _docs_from_response(response)
        return LLM08Execution(
            response=response,
            llm08_retrieved_docs=docs,
            llm08_checks={
                "strategy": "embedding_poisoning",
                "poisoned_doc_id": poisoned_doc_id,
                "poisoned_doc_retrieved": any(doc["doc_id"] == poisoned_doc_id for doc in docs),
            },
        )

    async def _execute_standard_retrieval(self, attack: dict[str, Any]) -> LLM08Execution:
        query = str(
            attack.get("adversarial_query")
            or attack.get("probe_query")
            or attack.get("inversion_query")
            or attack["prompt_text"]
        )
        response = await self.target_system.aquery(query)
        docs = _docs_from_response(response)
        checks: dict[str, Any] = {"strategy": attack["attack_strategy"]}
        if attack["attack_strategy"] == "similarity_collision":
            target_doc_id = str(attack.get("target_doc_id", ""))
            checks["target_doc_id"] = target_doc_id
            checks["target_doc_retrieved"] = any(doc["doc_id"] == target_doc_id for doc in docs)
        return LLM08Execution(
            response=response,
            llm08_retrieved_docs=docs,
            llm08_checks=checks,
        )

    async def _query_with_vectorstore(self, query: str, vectorstore: Any) -> Response:
        start = time.perf_counter()
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=self.top_k)
        docs = [doc for doc, _ in docs_with_scores]
        context = _format_docs(docs)

        prompt = self.target_system._prompt
        llm = self.target_system._llm
        messages = prompt.format_messages(context=context, question=query)
        raw = await llm.ainvoke(messages)
        usage = getattr(raw, "response_metadata", {}).get("usage", {}) or {}
        tokens_used = int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

        return Response(
            answer=str(raw.content),
            retrieved_chunks=[
                RetrievedChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", ""),
                    doc_id=doc.metadata.get("doc_id", ""),
                    score=float(score),
                )
                for doc, score in docs_with_scores
            ],
            latency_ms=(time.perf_counter() - start) * 1000,
            tokens_used=tokens_used,
        )


def snapshot_mtimes(index_dir: Path) -> dict[Path, int]:
    """Return mtime snapshots for files under a FAISS index directory."""
    return {path: path.stat().st_mtime_ns for path in sorted(index_dir.iterdir()) if path.is_file()}


def assert_mtimes_unchanged(before: dict[Path, int]) -> None:
    """Raise if any previously snapshotted index file changed on disk."""
    after = {path: path.stat().st_mtime_ns for path in before}
    if after != before:
        changed = [str(path) for path, mtime in before.items() if after.get(path) != mtime]
        raise AssertionError(f"FAISS index files changed on disk: {changed}")


def _docs_from_response(response: Response) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": chunk.doc_id,
            "source": chunk.source,
            "distance": chunk.score,
            "similarity": 1 / (1 + chunk.score),
            "content": chunk.content,
        }
        for chunk in response.retrieved_chunks
    ]


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
