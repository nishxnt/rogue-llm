from __future__ import annotations

from typing import TYPE_CHECKING

from src.reporting.html_renderer import render_risk_report_html, render_risk_report_html_from_json
from src.reporting.report_builder import RiskReport, build_risk_report, write_risk_report
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


def _build_report() -> RiskReport:
    return build_risk_report(
        unguarded_risk_path=UNGUARDED_RISK,
        guarded_results_path=GUARDED_RESULTS,
        guarded_decisions_path=GUARDED_DECISIONS,
        guarded_scores_path=GUARDED_SCORES,
        guarded_risk_path=GUARDED_RISK,
        residual_analysis_path=RESIDUAL_ANALYSIS,
        cross_validation_path=CROSS_VALIDATION,
        report_tag="v0.5.0-phase5",
    )


def test_render_risk_report_html_embeds_assets_inline(tmp_path: Path) -> None:
    report = _build_report()

    html_path = render_risk_report_html(report, output_path=tmp_path / "risk_report.html")
    html = html_path.read_text(encoding="utf-8")

    assert html_path.name == "risk_report.html"
    assert "<style>" in html
    assert "<script>" in html
    assert "<script src=" not in html
    assert "Per-Category Risk Heatmap" in html
    assert "OWASP Web Chunking Finding" in html
    assert "LLM06 Weakness" in html
    assert "Residual Vulnerabilities" in html


def test_render_risk_report_html_from_json_round_trips(tmp_path: Path) -> None:
    report = _build_report()
    json_path = write_risk_report(report, output_root=tmp_path)

    html_path = render_risk_report_html_from_json(risk_report_path=json_path)

    assert html_path.exists()
    assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
