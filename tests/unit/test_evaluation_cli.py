from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from src.evaluation import cli, cross_validator
from src.evaluation.cli import app
from src.evaluation.engine import AttackEvaluationInput, MetricResult
from src.pipeline.groq_client import GroqPreflightBudget

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class FakeMetric:
    name = "refusal"
    judge_model = "deterministic"
    judge_version = "test-v1"

    async def score(self, attack: AttackEvaluationInput) -> MetricResult:
        return MetricResult(
            attack_id=attack.attack_id,
            metric_name=self.name,
            score=0.0,
            judge_model=self.judge_model,
            judge_version=self.judge_version,
        )


def _write_results(path: Path) -> None:
    rows = [
        {
            "attack_id": "LLM01-0001",
            "owasp_category": "LLM01:2025",
            "attack_prompt": "prompt 1",
            "target_response": "response 1",
            "retrieved_chunks": ["context 1"],
        },
        {
            "attack_id": "LLM02-0001",
            "owasp_category": "LLM02:2025",
            "attack_prompt": "prompt 2",
            "target_response": "response 2",
            "retrieved_chunks": ["context 2"],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_sample_cli_scores_stratified_subset_without_live_judges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results.jsonl"
    cache = tmp_path / "cache.sqlite"
    output_root = tmp_path / "out"
    _write_results(results)
    monkeypatch.setattr(cli, "build_metric_suite", lambda **_: [FakeMetric()])

    async def fake_probe(**_: object) -> list[GroqPreflightBudget]:
        return [
            GroqPreflightBudget("primary", 25_000, "5m"),
            GroqPreflightBudget("secondary", 25_000, "5m"),
        ]

    monkeypatch.setattr(
        cli,
        "probe_groq_token_budgets",
        fake_probe,
    )

    result = CliRunner().invoke(
        app,
        [
            "sample",
            "--results",
            str(results),
            "--n",
            "1",
            "--cache",
            str(cache),
            "--output-root",
            str(output_root),
            "--deterministic-only",
        ],
    )

    assert result.exit_code == 0
    assert "sample: scored 1 attack(s)" in result.output
    assert "System Risk Score:" in result.output
    assert list(output_root.glob("run_*/scores.jsonl"))
    assert list(output_root.glob("run_*/risk_scores.json"))


def test_resume_cli_aborts_when_combined_preflight_budget_is_too_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results.jsonl"
    cache = tmp_path / "cache.sqlite"
    output_root = tmp_path / "out"
    _write_results(results)
    monkeypatch.setattr(cli, "build_metric_suite", lambda **_: [FakeMetric()])

    async def fake_probe(**_: object) -> list[GroqPreflightBudget]:
        return [
            GroqPreflightBudget("primary", 612, "5m"),
            GroqPreflightBudget("secondary", 0, "9m"),
        ]

    monkeypatch.setattr(
        cli,
        "probe_groq_token_budgets",
        fake_probe,
    )

    result = CliRunner().invoke(
        app,
        [
            "resume",
            "--results",
            str(results),
            "--cache",
            str(cache),
            "--output-root",
            str(output_root),
        ],
    )

    assert result.exit_code == 1
    assert "Insufficient TPD budget across configured keys." in result.output
    assert "Primary: 612 tokens." in result.output
    assert "Secondary: 0 tokens." in result.output
    assert not list(output_root.glob("run_*/scores.jsonl"))


def test_cross_validate_cli_prints_metric_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results.jsonl"
    scores = tmp_path / "scores.jsonl"
    cache = tmp_path / "cache.sqlite"
    output_root = tmp_path / "out"
    _write_results(results)
    scores.write_text(
        MetricResult(
            attack_id="LLM01-0001",
            metric_name="faithfulness",
            score=0.0,
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(cross_validator, "build_metric_suite", lambda **_: [FakeMetric()])

    result = CliRunner().invoke(
        app,
        [
            "cross-validate",
            "--results",
            str(results),
            "--scores",
            str(scores),
            "--cache",
            str(cache),
            "--output-root",
            str(output_root),
            "--sample-size",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Cross-validation report:" in result.output
