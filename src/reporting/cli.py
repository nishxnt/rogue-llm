"""Typer CLI for Phase 6 risk report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.reporting.html_renderer import render_risk_report_html, render_risk_report_html_from_json
from src.reporting.report_builder import build_risk_report, write_risk_report

app = typer.Typer(help="Phase 6 reporting and standalone HTML generation.")

PathOption = Annotated[Path, typer.Option(file_okay=True, dir_okay=False, readable=True)]
OutputRootOption = Annotated[Path, typer.Option("--output-root")]
OutputPathOption = Annotated[Path | None, typer.Option("--output-path")]
TagOption = Annotated[str, typer.Option("--report-tag")]

_DEFAULT_OUTPUT_ROOT = Path("results")
_DEFAULT_TAG = "v0.5.0-phase5"


@app.command("build-report")
def build_report(
    unguarded_risk: PathOption,
    guarded_results: PathOption,
    guarded_decisions: PathOption,
    guarded_scores: PathOption,
    guarded_risk: PathOption,
    residual_analysis: PathOption,
    cross_validation: PathOption,
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    report_tag: TagOption = _DEFAULT_TAG,
) -> None:
    """Build risk_report.json from existing Phase 4 and Phase 5 artifacts."""
    report = build_risk_report(
        unguarded_risk_path=unguarded_risk,
        guarded_results_path=guarded_results,
        guarded_decisions_path=guarded_decisions,
        guarded_scores_path=guarded_scores,
        guarded_risk_path=guarded_risk,
        residual_analysis_path=residual_analysis,
        cross_validation_path=cross_validation,
        report_tag=report_tag,
    )
    path = write_risk_report(report, output_root=output_root)
    typer.echo(f"Risk report JSON: {path}")


@app.command("render-html")
def render_html(
    risk_report: Annotated[
        Path, typer.Option("--risk-report", file_okay=True, dir_okay=False, readable=True)
    ],
    output_path: OutputPathOption = None,
) -> None:
    """Render standalone HTML from a previously generated risk_report.json."""
    path = render_risk_report_html_from_json(
        risk_report_path=risk_report,
        output_path=output_path,
    )
    typer.echo(f"Risk report HTML: {path}")


@app.command("full")
def full(
    unguarded_risk: PathOption,
    guarded_results: PathOption,
    guarded_decisions: PathOption,
    guarded_scores: PathOption,
    guarded_risk: PathOption,
    residual_analysis: PathOption,
    cross_validation: PathOption,
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    report_tag: TagOption = _DEFAULT_TAG,
) -> None:
    """Build JSON and render standalone HTML in one pass."""
    report = build_risk_report(
        unguarded_risk_path=unguarded_risk,
        guarded_results_path=guarded_results,
        guarded_decisions_path=guarded_decisions,
        guarded_scores_path=guarded_scores,
        guarded_risk_path=guarded_risk,
        residual_analysis_path=residual_analysis,
        cross_validation_path=cross_validation,
        report_tag=report_tag,
    )
    json_path = write_risk_report(report, output_root=output_root)
    html_path = render_risk_report_html(report, output_path=json_path.with_suffix(".html"))
    typer.echo(f"Risk report JSON: {json_path}")
    typer.echo(f"Risk report HTML: {html_path}")


if __name__ == "__main__":
    app()
