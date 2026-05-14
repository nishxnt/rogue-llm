import json
from pathlib import Path

import pytest

from src.pipeline.attack_runner import AttackRunner
from src.pipeline.rate_limiter import TokenBucketRateLimiter
from src.target_system.models import Response, RetrievedChunk


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
