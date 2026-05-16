from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from src.evaluation.scorer import CategoryRiskScore, SystemRiskScore
from src.guardrails import cli
from src.guardrails.cli import app
from src.pipeline.groq_client import GroqPreflightBudget

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_run_attacks_cli_aborts_on_low_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_guarded_attacks(**_: object) -> Path:
        raise AssertionError("guarded run should not start when preflight is low")

    monkeypatch.setattr(
        cli,
        "probe_groq_rate_limits",
        lambda **_: [GroqPreflightBudget("primary", 50, "1h", 3000, "5m", {})],
    )
    monkeypatch.setattr(cli, "_run_guarded_attacks", fake_run_guarded_attacks)

    result = CliRunner().invoke(
        app,
        [
            "run-attacks",
            "--dataset",
            "attacks/v1/dataset.jsonl",
            "--policy",
            "src/guardrails/policy.yaml",
        ],
    )

    assert result.exit_code == 1
    assert "Insufficient preflight headroom across configured keys." in result.output


def test_run_attacks_cli_reports_results_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_guarded_attacks(**_: object) -> Path:
        run_dir = tmp_path / "run_20260516_000000"
        run_dir.mkdir(parents=True)
        results_path = run_dir / "results.jsonl"
        results_path.write_text("", encoding="utf-8")
        return results_path

    monkeypatch.setattr(
        cli,
        "probe_groq_rate_limits",
        lambda **_: [GroqPreflightBudget("primary", 500, "1h", 8000, "5m", {})],
    )
    monkeypatch.setattr(cli, "_run_guarded_attacks", fake_run_guarded_attacks)

    result = CliRunner().invoke(
        app,
        [
            "run-attacks",
            "--dataset",
            "attacks/v1/dataset.jsonl",
            "--policy",
            "src/guardrails/policy.yaml",
            "--output-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Guarded attack results:" in result.output


def test_evaluate_cli_writes_scoring_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text("", encoding="utf-8")
    scores_path = tmp_path / "run_1" / "scores.jsonl"
    risk_path = tmp_path / "run_1" / "risk_scores.json"
    scores_path.parent.mkdir(parents=True)
    scores_path.write_text("", encoding="utf-8")
    risk_path.write_text("{}", encoding="utf-8")

    async def fake_score_results(**_: object) -> tuple[Path, Path, int, float]:
        return scores_path, risk_path, 5, 0.1234

    monkeypatch.setattr(
        cli,
        "probe_groq_rate_limits",
        lambda **_: [GroqPreflightBudget("primary", 500, "1h", 8000, "5m", {})],
    )
    monkeypatch.setattr(cli, "score_results", fake_score_results)

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--results",
            str(results),
            "--output-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "evaluate: scored 5 attack(s)" in result.output
    assert "System Risk Score: 0.1234" in result.output


def test_delta_report_cli_writes_delta_artifact(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    guarded_path = tmp_path / "guarded.json"
    baseline_path.write_text(
        SystemRiskScore(
            risk_score=0.5,
            category_scores=[
                CategoryRiskScore(
                    owasp_category="LLM01:2025",
                    attack_count=1,
                    risk_score=0.5,
                    weight=1.0,
                )
            ],
            attack_scores=[],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    guarded_path.write_text(
        SystemRiskScore(
            risk_score=0.2,
            category_scores=[
                CategoryRiskScore(
                    owasp_category="LLM01:2025",
                    attack_count=1,
                    risk_score=0.2,
                    weight=1.0,
                )
            ],
            attack_scores=[],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "delta-report",
            "--baseline-risk",
            str(baseline_path),
            "--guarded-risk",
            str(guarded_path),
            "--output-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "System delta: 0.3000" in result.output
    paths = list(tmp_path.glob("run_*/delta_report.json"))
    assert paths
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["system_delta"] == 0.3
