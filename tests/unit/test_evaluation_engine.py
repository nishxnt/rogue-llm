from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.evaluation.engine import AttackEvaluationInput, EvaluationEngine, MetricResult

if TYPE_CHECKING:
    from pathlib import Path


class CountingMetric:
    name = "counting"
    judge_model = "deterministic"
    judge_version = "v1"

    def __init__(self) -> None:
        self.calls = 0

    async def score(self, attack: AttackEvaluationInput) -> MetricResult:
        self.calls += 1
        return MetricResult(attack_id=attack.attack_id, metric_name=self.name, score=0.25)


def _write_results(path: Path) -> None:
    row = {
        "attack_id": "LLM01-0001",
        "owasp_category": "LLM01:2025",
        "attack_prompt": "attack",
        "target_response": "response",
        "retrieved_chunks": ["context"],
        "retrieved_doc_ids": ["doc-1"],
        "latency_ms": 1,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_evaluation_engine_scores_and_uses_metric_cache(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    _write_results(results_path)
    metric = CountingMetric()
    engine = EvaluationEngine(
        results_path=results_path,
        cache_path=tmp_path / "cache.sqlite",
        metrics=[metric],
        output_root=tmp_path / "out",
    )

    try:
        first = await engine.run()
        second = await engine.run()
    finally:
        engine.close()

    assert [score.score for score in first] == [0.25]
    assert [score.score for score in second] == [0.25]
    assert metric.calls == 1
    score_files = sorted((tmp_path / "out").glob("run_*/scores.jsonl"))
    assert score_files
