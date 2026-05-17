from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.reporting.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


def test_reporting_cli_full_generates_json_and_html(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "full",
            "--unguarded-risk",
            str(REPO_ROOT / "results/run_20260516_131022/risk_scores.json"),
            "--guarded-results",
            str(REPO_ROOT / "results/run_20260516_164921/results.jsonl"),
            "--guarded-decisions",
            str(REPO_ROOT / "results/run_20260516_164921/guardrail_decisions.jsonl"),
            "--guarded-scores",
            str(REPO_ROOT / "results/run_20260517_115140/scores.jsonl"),
            "--guarded-risk",
            str(REPO_ROOT / "results/run_20260517_115140/risk_scores.json"),
            "--residual-analysis",
            str(REPO_ROOT / "results/run_20260517_115451/residual_analysis.json"),
            "--cross-validation",
            str(REPO_ROOT / "results/cross_validation_20260516_132118/cross_validation.json"),
            "--output-root",
            str(tmp_path),
            "--report-tag",
            "v0.5.0-phase5",
        ],
    )

    assert result.exit_code == 0, result.stdout
    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "risk_report.json").exists()
    assert (run_dirs[0] / "risk_report.html").exists()
