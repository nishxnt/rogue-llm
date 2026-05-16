from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.documents import Document

from src.pipeline.attack_runner import AttackRunner
from src.pipeline.rate_limiter import TokenBucketRateLimiter
from src.target_system.models import Response, RetrievedChunk

if TYPE_CHECKING:
    from pathlib import Path

    from src.guardrails.reasons import GuardrailDecision


class FakeTarget:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def aquery(self, prompt: str) -> Response:
        self.calls.append(prompt)
        return Response(
            answer=f"answer: {prompt}",
            retrieved_chunks=[
                RetrievedChunk(content="chunk one", source="fixture", doc_id="doc-1", score=0.1)
            ],
            latency_ms=12.5,
            tokens_used=42,
        )


class FakeGuardrailedTarget(FakeTarget):
    def __init__(
        self,
        *,
        decision: GuardrailDecision = "allowed",
        base_target_called: bool = True,
    ) -> None:
        super().__init__()
        self.decision = decision
        self._base_target_called = base_target_called

    def set_attack_context(self, attack_id: str) -> None:
        self.current_attack_id = attack_id

    async def aquery(self, prompt: str) -> Response:
        response = await super().aquery(prompt)
        response.guardrail_decision = self.decision
        response.guardrail_decision_layer = 2 if self.decision != "allowed" else None
        response.guardrail_evidence = {
            "source": "test",
            "reason": "classifier_unavailable_after_retries",
        }
        response.base_target_called = self._base_target_called
        response.guardrail_timestamp = "2026-05-16T00:00:00+00:00"
        return response


class FakeVectorstore:
    def __init__(self, docs: list[Document] | None = None) -> None:
        self.docs = docs or [
            Document(
                page_content="original chunk",
                metadata={"source": "fixture", "doc_id": "original-doc"},
            )
        ]

    def __deepcopy__(self, memo: dict[int, Any]) -> FakeVectorstore:
        return FakeVectorstore(copy.deepcopy(self.docs, memo))

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]]) -> list[str]:
        for text, metadata in zip(texts, metadatas, strict=True):
            self.docs.insert(0, Document(page_content=text, metadata=metadata))
        return [metadata["doc_id"] for metadata in metadatas]

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        return [(doc, float(i)) for i, doc in enumerate(self.docs[:k])]


class FakePrompt:
    def format_messages(self, *, context: str, question: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"{question}\n{context}"}]


class FakeLLM:
    async def ainvoke(self, messages: list[dict[str, str]]) -> Any:
        return SimpleNamespace(
            content=f"llm08 response {len(messages)}",
            response_metadata={"usage": {"prompt_tokens": 3, "completion_tokens": 4}},
        )


class FakeLLM08Target(FakeTarget):
    def __init__(self) -> None:
        super().__init__()
        self._vectorstore = FakeVectorstore()
        self._prompt = FakePrompt()
        self._llm = FakeLLM()


class FakeFailure(Exception):
    status_code = 500


class FailingTarget:
    async def aquery(self, prompt: str) -> Response:
        raise FakeFailure(prompt)


class NoopLimiter(TokenBucketRateLimiter):
    def __init__(self) -> None:
        pass

    async def acquire(self) -> None:
        return None


async def _no_sleep(_: float) -> None:
    return None


def _write_dataset(path: Path, count: int = 2) -> None:
    rows = [
        {
            "id": f"LLM01-{i:04d}",
            "owasp_category": "LLM01:2025",
            "attack_strategy": "direct_override",
            "prompt_text": f"prompt {i}",
        }
        for i in range(1, count + 1)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.asyncio
async def test_attack_runner_writes_structured_results_and_cache(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    cache_path = tmp_path / "cache.sqlite"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    target = FakeTarget()
    runner = AttackRunner(
        target_system=target,
        dataset_path=dataset_path,
        cache_path=cache_path,
        results_root=results_root,
        rate_limiter=NoopLimiter(),
    )

    try:
        results = await runner.run()
        second_results = await runner.run()
    finally:
        runner.close()

    assert len(results) == 2
    assert results[0].attack_id == "LLM01-0001"
    assert results[0].target_response == "answer: prompt 1"
    assert results[0].retrieved_chunks == ["chunk one"]
    assert results[0].retrieved_doc_ids == ["doc-1"]
    assert results[0].cache_hit is False
    assert [result.cache_hit for result in second_results] == [True, True]
    assert target.calls == ["prompt 1", "prompt 2"]
    assert cache_path.exists()
    result_files = sorted(results_root.glob("run_*/results.jsonl"))
    assert result_files
    assert len(result_files[-1].read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.asyncio
async def test_attack_runner_writes_guardrail_decisions_when_present(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    cache_path = tmp_path / "cache.sqlite"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path, count=1)
    runner = AttackRunner(
        target_system=FakeGuardrailedTarget(),
        dataset_path=dataset_path,
        cache_path=cache_path,
        results_root=results_root,
        rate_limiter=NoopLimiter(),
    )

    try:
        await runner.run()
    finally:
        runner.close()

    decision_files = sorted(results_root.glob("run_*/guardrail_decisions.jsonl"))
    assert decision_files
    row = json.loads(decision_files[-1].read_text(encoding="utf-8").splitlines()[0])
    assert row["attack_id"] == "LLM01-0001"
    assert row["decision"] == "allowed"
    assert row["base_target_called"] is True


@pytest.mark.asyncio
async def test_attack_runner_writes_classifier_unavailable_guardrail_decision(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    cache_path = tmp_path / "cache.sqlite"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path, count=1)
    runner = AttackRunner(
        target_system=FakeGuardrailedTarget(
            decision="classifier_unavailable_blocked",
            base_target_called=False,
        ),
        dataset_path=dataset_path,
        cache_path=cache_path,
        results_root=results_root,
        rate_limiter=NoopLimiter(),
    )

    try:
        await runner.run()
    finally:
        runner.close()

    decision_files = sorted(results_root.glob("run_*/guardrail_decisions.jsonl"))
    assert decision_files
    row = json.loads(decision_files[-1].read_text(encoding="utf-8").splitlines()[0])
    assert row["attack_id"] == "LLM01-0001"
    assert row["decision"] == "classifier_unavailable_blocked"
    assert row["decision_layer"] == 2
    assert row["base_target_called"] is False
    assert row["evidence"]["reason"] == "classifier_unavailable_after_retries"


@pytest.mark.asyncio
async def test_attack_runner_records_infrastructure_failure(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(dataset_path, count=1)
    runner = AttackRunner(
        target_system=FailingTarget(),
        dataset_path=dataset_path,
        cache_path=tmp_path / "cache.sqlite",
        results_root=tmp_path / "results",
        rate_limiter=NoopLimiter(),
        retry_sleeper=_no_sleep,
    )

    try:
        results = await runner.run()
    finally:
        runner.close()

    assert len(results) == 1
    assert results[0].status == "infrastructure_failure"
    assert results[0].error_type == "FakeFailure"


@pytest.mark.asyncio
async def test_run_with_sample_is_stratified(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    rows = []
    for category in ("LLM01:2025", "LLM02:2025", "LLM03:2025"):
        for i in range(2):
            rows.append(
                {
                    "id": f"{category[:5]}-{i:04d}",
                    "owasp_category": category,
                    "attack_strategy": "test",
                    "prompt_text": f"{category} prompt {i}",
                }
            )
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    runner = AttackRunner(
        target_system=FakeTarget(),
        dataset_path=dataset_path,
        cache_path=tmp_path / "cache.sqlite",
        results_root=tmp_path / "results",
        rate_limiter=NoopLimiter(),
    )

    try:
        results = await runner.run_with_sample(3)
    finally:
        runner.close()

    assert len(results) == 3
    assert {result.owasp_category for result in results} == {
        "LLM01:2025",
        "LLM02:2025",
        "LLM03:2025",
    }


@pytest.mark.asyncio
async def test_attack_runner_adds_llm08_retrieval_diagnostics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    attack = {
        "id": "LLM08-0001",
        "owasp_category": "LLM08:2025",
        "attack_strategy": "embedding_poisoning",
        "prompt_text": "What is the canonical mitigation?",
        "target_query": "What is the canonical mitigation?",
        "poisoned_doc_content": "poisoned benchmark-only guidance",
        "poisoned_doc_metadata": {"source": "fake", "doc_id": "POISON-001"},
    }
    dataset_path.write_text(json.dumps(attack) + "\n", encoding="utf-8")
    runner = AttackRunner(
        target_system=FakeLLM08Target(),
        dataset_path=dataset_path,
        cache_path=tmp_path / "cache.sqlite",
        results_root=tmp_path / "results",
        rate_limiter=NoopLimiter(),
        retry_sleeper=_no_sleep,
    )

    try:
        results = await runner.run()
    finally:
        runner.close()

    assert len(results) == 1
    dumped = results[0].model_dump()
    assert dumped["llm08_checks"]["poisoned_doc_retrieved"] is True
    assert dumped["llm08_retrieved_docs"][0]["doc_id"] == "POISON-001"
    assert "distance" in dumped["llm08_retrieved_docs"][0]
    assert "similarity" in dumped["llm08_retrieved_docs"][0]
    assert "score" not in dumped["llm08_retrieved_docs"][0]
