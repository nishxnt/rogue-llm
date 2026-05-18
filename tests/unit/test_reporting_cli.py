from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from src.reporting.cli import app
from tests.unit.reporting_fixture_paths import (
    CROSS_VALIDATION,
    GUARDED_DECISIONS,
    GUARDED_RESULTS,
    GUARDED_RISK,
    GUARDED_SCORES,
    RESIDUAL_ANALYSIS,
    UNGUARDED_RISK,
)

if TYPE_CHECKING:
    from pathlib import Path

RUNNER = CliRunner()


def test_reporting_cli_full_generates_json_and_html(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "full",
            "--unguarded-risk",
            str(UNGUARDED_RISK),
            "--guarded-results",
            str(GUARDED_RESULTS),
            "--guarded-decisions",
            str(GUARDED_DECISIONS),
            "--guarded-scores",
            str(GUARDED_SCORES),
            "--guarded-risk",
            str(GUARDED_RISK),
            "--residual-analysis",
            str(RESIDUAL_ANALYSIS),
            "--cross-validation",
            str(CROSS_VALIDATION),
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
