"""Unit tests for the baseline evaluation helper workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import typer
from langchain_core.messages import AIMessage

from src.evaluation import baseline

if TYPE_CHECKING:
    from pathlib import Path


def test_load_chunks_reads_nvd_and_owasp_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb_dir = tmp_path / "kb"
    (kb_dir / "owasp_llm_top10").mkdir(parents=True)
    (kb_dir / "owasp_web_top10").mkdir(parents=True)
    (kb_dir / "nvd_cves.jsonl").write_text(
        json.dumps(
            {
                "id": "CVE-2026-0001",
                "description": "A" * 90,
            }
        )
        + "\n"
        + json.dumps({"id": "CVE-2026-0002", "description": "short"})
        + "\n",
        encoding="utf-8",
    )
    (kb_dir / "owasp_llm_top10" / "LLM01.md").write_text("B" * 100, encoding="utf-8")
    (kb_dir / "owasp_web_top10" / "A01.md").write_text("C" * 100, encoding="utf-8")
    monkeypatch.setattr(baseline, "_KB_DIR", kb_dir)

    chunks = baseline._load_chunks()

    assert {chunk["source"] for chunk in chunks} == {"nvd", "owasp_llm", "owasp_web"}
    assert {chunk["doc_id"] for chunk in chunks} == {"CVE-2026-0001", "LLM01", "A01"}


def test_stratified_sample_includes_each_source() -> None:
    chunks = [
        {"source": "nvd", "content": "a", "doc_id": "1"},
        {"source": "nvd", "content": "b", "doc_id": "2"},
        {"source": "owasp_llm", "content": "c", "doc_id": "3"},
        {"source": "owasp_web", "content": "d", "doc_id": "4"},
    ]

    sampled = baseline._stratified_sample(chunks, 3)

    assert len(sampled) == 3
    assert {chunk["source"] for chunk in sampled} == {"nvd", "owasp_llm", "owasp_web"}


class _FakeLLM:
    def __init__(self, content: str | Exception) -> None:
        self.content = content

    def invoke(self, messages: object) -> AIMessage:
        if isinstance(self.content, Exception):
            raise self.content
        return AIMessage(content=self.content)


def test_generate_qa_parses_json_fenced_output() -> None:
    chunk = {"content": "context", "source": "nvd", "doc_id": "CVE-1"}
    llm = _FakeLLM(
        '```json\n{"question": "What happened?", "ground_truth": "A supported answer."}\n```'
    )

    qa = baseline._generate_qa(chunk, llm, 7)

    assert qa == {
        "id": "semi-007",
        "question": "What happened?",
        "ground_truth": "A supported answer.",
        "reference_doc_id": "CVE-1",
        "source": "nvd",
        "construction": "semi-synthetic",
    }


def test_generate_qa_returns_none_on_invalid_output() -> None:
    chunk = {"content": "context", "source": "nvd", "doc_id": "CVE-1"}

    assert baseline._generate_qa(chunk, _FakeLLM("{}"), 1) is None
    assert baseline._generate_qa(chunk, _FakeLLM(RuntimeError("boom")), 1) is None


def test_load_checkpoint_ignores_invalid_and_unscored_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "faithfulness_score": 0.5}),
                json.dumps({"id": "b", "faithfulness_score": None}),
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "_CHECKPOINT_PATH", checkpoint)

    assert baseline._load_checkpoint() == {"a": {"id": "a", "faithfulness_score": 0.5}}


@dataclass
class _Chunk:
    content: str
    doc_id: str


class _Chatbot:
    def query(self, question: str) -> SimpleNamespace:
        return SimpleNamespace(
            answer=f"answer to {question}",
            retrieved_chunks=[_Chunk(content="supporting context", doc_id="DOC-1")],
        )


class _Scorer:
    async def ascore(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> SimpleNamespace:
        assert user_input
        assert response
        assert retrieved_contexts == ["supporting context"]
        return SimpleNamespace(value=0.75)


async def test_score_all_resumes_and_writes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baseline, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(baseline, "_CHECKPOINT_PATH", tmp_path / "checkpoint.jsonl")

    qa_records = [
        {"id": "done", "question": "Already scored?"},
        {"id": "new", "question": "Score me?"},
    ]
    pre_scored = {"done": {"id": "done", "question": "Already scored?", "faithfulness_score": 1.0}}

    scored = await baseline._score_all(qa_records, _Chatbot(), _Scorer(), pre_scored)

    assert [row["id"] for row in scored] == ["done", "new"]
    assert scored[0]["faithfulness_score"] == 1.0
    assert scored[1]["faithfulness_score"] == 0.75
    assert scored[1]["retrieved_doc_ids"] == ["DOC-1"]

    checkpoint_rows = (tmp_path / "checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(checkpoint_rows) == 1
    assert json.loads(checkpoint_rows[0])["id"] == "new"


def test_run_generate_rejects_missing_knowledge_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baseline, "_KB_DIR", tmp_path / "missing")
    monkeypatch.setattr(baseline, "get_settings", lambda: SimpleNamespace())

    with pytest.raises(typer.Exit) as exc_info:
        baseline._run_generate()

    assert exc_info.value.exit_code == 1
