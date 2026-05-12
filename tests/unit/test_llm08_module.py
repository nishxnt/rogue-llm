from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from src.attack_generation import llm08_module

if TYPE_CHECKING:
    from pathlib import Path


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        words = {
            word.strip(".,:;!?()[]{}\"'").lower()
            for word in text.split()
            if word.strip(".,:;!?()[]{}\"'")
        }
        dimensions = [
            {"sql", "injection", "database", "query"},
            {"xss", "cross-site", "scripting", "html", "rendered"},
            {"prompt", "injection", "rag", "instructions"},
            {"sensitive", "information", "logs", "backups"},
            {"emailgpt", "service", "logic"},
            {"gl.inet", "admintoken", "cookie", "crontab"},
            {"chunk", "quote", "retrieved", "document"},
        ]
        return [float(len(words & dimension)) for dimension in dimensions]


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate_poisoned_document(
        self,
        target_query: str,
        attempt: int,
        anchor_terms: str,
    ) -> str:
        self.calls += 1
        return (
            f"{target_query} Authoritative synthetic guidance for benchmark testing. "
            f"Anchors: {anchor_terms}. This document intentionally contains one wrong "
            "defensive recommendation without operational misuse details."
        )

    def generate_collision_query(
        self,
        target_doc_id: str,
        target_source: str,
        target_excerpt: str,
        attempt: int,
        anchor_terms: str,
    ) -> str:
        self.calls += 1
        return (
            f"{target_excerpt} Please summarize this as a benign governance question "
            f"for {target_source}, using anchors {anchor_terms}."
        )


class FakeDocstore:
    def __init__(self, docs: dict[str, Any]) -> None:
        self._docs = docs

    def search(self, docstore_id: str) -> Any:
        return self._docs[docstore_id]


class FakeVectorstore:
    def __init__(self) -> None:
        docs = {
            "a": SimpleNamespace(
                page_content=(
                    "SQL injection database query vulnerability mitigation prepared statements "
                    "parameterized database access"
                ),
                metadata={"source": "nvd", "doc_id": "CVE-2023-41287"},
            ),
            "b": SimpleNamespace(
                page_content=(
                    "EmailGPT prompt injection service logic takeover direct prompt instruction "
                    "handling"
                ),
                metadata={"source": "nvd", "doc_id": "CVE-2024-5184"},
            ),
            "c": SimpleNamespace(
                page_content=(
                    "GL.iNet AdminToken cookie crontab remote code execution device management"
                ),
                metadata={"source": "nvd", "doc_id": "CVE-2023-50922"},
            ),
        }
        self.docstore: FakeDocstore = FakeDocstore(docs)
        self.index_to_docstore_id: dict[int, str] = {0: "a", 1: "b", 2: "c"}


def _build_with_fakes() -> list[dict[str, Any]]:
    return llm08_module.build_llm08_attacks(
        embeddings=FakeEmbeddings(),
        llm_client=FakeLLM(),
        vectorstore=FakeVectorstore(),
    )


def test_embedding_poisoning_returns_high_similarity() -> None:
    attacks = _build_with_fakes()
    poisoning = [a for a in attacks if a["attack_strategy"] == "embedding_poisoning"]

    assert len(poisoning) == 4
    assert all(a["intended_similarity"] > 0.65 for a in poisoning)


def test_similarity_collision_returns_paraphrase_above_threshold() -> None:
    attacks = _build_with_fakes()
    collisions = [a for a in attacks if a["attack_strategy"] == "similarity_collision"]

    assert len(collisions) == 3
    assert all(a["measured_similarity"] > 0.70 for a in collisions)
    assert {a["target_doc_id"] for a in collisions} == {
        "CVE-2023-41287",
        "CVE-2024-5184",
        "CVE-2023-50922",
    }


def test_module_returns_11_entries() -> None:
    attacks = _build_with_fakes()

    assert len(attacks) == 11
    assert {a["id"] for a in attacks} == {f"LLM08-{i:04d}" for i in range(1, 12)}
    assert [a["attack_strategy"] for a in attacks].count("embedding_inversion") == 2


def test_no_index_mutation(monkeypatch: Any, tmp_path: Path) -> None:
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    index_files = [index_dir / "index.faiss", index_dir / "index.pkl"]
    for path in index_files:
        path.write_text("unchanged", encoding="utf-8")

    before = {path.name: os.stat(path).st_mtime_ns for path in index_files}

    def fake_load_vectorstore(index_dir_arg: Path, embeddings: Any) -> FakeVectorstore:
        assert index_dir_arg == index_dir
        return FakeVectorstore()

    monkeypatch.setattr(llm08_module, "_load_vectorstore", fake_load_vectorstore)

    attacks = llm08_module.build_llm08_attacks(
        embeddings=FakeEmbeddings(),
        llm_client=FakeLLM(),
        index_dir=index_dir,
    )

    after = {path.name: os.stat(path).st_mtime_ns for path in index_files}
    assert len(attacks) == 11
    assert after == before
