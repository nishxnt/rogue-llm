import json
from pathlib import Path

import pytest

from src.pipeline.attack_runner import AttackRunner
from src.pipeline.rate_limiter import TokenBucketRateLimiter
from src.target_system.models import Response, RetrievedChunk

pytestmark = pytest.mark.integration


class MockTarget:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def aquery(self, prompt: str) -> Response:
        self.calls.append(prompt)
        return Response(
            answer=f"mock response for {prompt}",
            retrieved_chunks=[
                RetrievedChunk(
                    content=f"retrieved context for {prompt}",
                    source="mock",
                    doc_id=f"doc-{len(self.calls)}",
                    score=0.2,
                )
            ],
            latency_ms=5.0,
            tokens_used=17,
        )


class NoopLimiter(TokenBucketRateLimiter):
    def __init__(self) -> None:
        pass

    async def acquire(self) -> None:
        return None


def _write_dataset(path: Path) -> None:
    rows = [
        {
            "id": f"LLM01-{i:04d}",
            "owasp_category": "LLM01:2025",
            "attack_strategy": "direct_override",
            "prompt_text": f"attack prompt {i}",
        }
        for i in range(1, 6)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.asyncio
async def test_runner_e2e_writes_results_and_reuses_cache(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    cache_path = tmp_path / "results_cache.sqlite"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    target = MockTarget()
    runner = AttackRunner(
        target_system=target,
        dataset_path=dataset_path,
        cache_path=cache_path,
        results_root=results_root,
        concurrency=2,
        rate_limiter=NoopLimiter(),
    )

    try:
        first = await runner.run()
        second = await runner.run()
    finally:
        runner.close()

    assert len(first) == 5
    assert len(second) == 5
    assert len(target.calls) == 5
    assert all(result.status == "success" for result in first)
    assert all(result.cache_hit is False for result in first)
    assert all(result.cache_hit is True for result in second)
    assert cache_path.exists()

    result_files = sorted(results_root.glob("run_*/results.jsonl"))
    assert result_files
    rows = [json.loads(line) for line in result_files[-1].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert rows[0]["attack_id"] == "LLM01-0001"
    assert rows[0]["retrieved_chunks"] == ["retrieved context for attack prompt 1"]
