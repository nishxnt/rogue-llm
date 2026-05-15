from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.evaluation import cross_validator
from src.evaluation.cross_validator import (
    cross_validate_run,
    stratified_cross_validation_sample,
)
from src.evaluation.engine import AttackEvaluationInput, MetricResult

if TYPE_CHECKING:
    from pathlib import Path


class FakeMetric:
    name = "faithfulness"
    judge_model = "fake-cross-family"
    judge_version = "test-v1"

    async def score(self, attack: AttackEvaluationInput) -> MetricResult:
        score = 0.7 if attack.attack_id.endswith("1") else 0.1
        return MetricResult(
            attack_id=attack.attack_id,
            metric_name=self.name,
            score=score,
            judge_model=self.judge_model,
            judge_version=self.judge_version,
        )


def _attack(attack_id: str, category: str) -> AttackEvaluationInput:
    return AttackEvaluationInput(
        attack_id=attack_id,
        owasp_category=category,
        attack_prompt="prompt",
        target_response="response",
        retrieved_chunks=["context"],
    )


def test_stratified_cross_validation_sample_is_deterministic_and_balanced() -> None:
    attacks = [
        _attack("LLM01-0001", "LLM01:2025"),
        _attack("LLM01-0002", "LLM01:2025"),
        _attack("LLM02-0001", "LLM02:2025"),
        _attack("LLM02-0002", "LLM02:2025"),
        _attack("LLM03-0001", "LLM03:2025"),
        _attack("LLM03-0002", "LLM03:2025"),
    ]

    first = stratified_cross_validation_sample(attacks, sample_size=3, seed=7)
    second = stratified_cross_validation_sample(attacks, sample_size=3, seed=7)

    assert [attack.attack_id for attack in first] == [attack.attack_id for attack in second]
    assert {attack.owasp_category for attack in first} == {
        "LLM01:2025",
        "LLM02:2025",
        "LLM03:2025",
    }


@pytest.mark.asyncio
async def test_cross_validate_run_compares_primary_and_cross_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_path = tmp_path / "results.jsonl"
    primary_scores_path = tmp_path / "scores.jsonl"
    cache_path = tmp_path / "cache.sqlite"

    attacks = [
        _attack("LLM01-0001", "LLM01:2025"),
        _attack("LLM02-0002", "LLM02:2025"),
    ]
    results_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "attack_id": attack.attack_id,
                    "owasp_category": attack.owasp_category,
                    "attack_prompt": attack.attack_prompt,
                    "target_response": attack.target_response,
                    "retrieved_chunks": attack.retrieved_chunks,
                }
            )
            for attack in attacks
        ),
        encoding="utf-8",
    )
    primary_scores = [
        MetricResult(attack_id="LLM01-0001", metric_name="faithfulness", score=0.8),
        MetricResult(attack_id="LLM02-0002", metric_name="faithfulness", score=0.6),
    ]
    primary_scores_path.write_text(
        "\n".join(score.model_dump_json() for score in primary_scores),
        encoding="utf-8",
    )

    monkeypatch.setattr(cross_validator, "build_metric_suite", lambda **_: [FakeMetric()])

    report, path = await cross_validate_run(
        results_path=results_path,
        primary_scores_path=primary_scores_path,
        cache_path=cache_path,
        output_root=tmp_path,
        sample_size=2,
        metric_names=("faithfulness",),
        agreement_tolerance=0.2,
        concurrency=1,
    )

    assert path.exists()
    assert report.sample_size == 2
    assert report.metric_summaries[0].metric_name == "faithfulness"
    assert report.metric_summaries[0].compared_count == 2
    assert report.metric_summaries[0].agreement_count == 1
    assert report.metric_summaries[0].agreement_rate == 0.5
