from __future__ import annotations

from pathlib import Path

from src.reporting.html_renderer import render_risk_report_html, render_risk_report_html_from_json
from src.reporting.report_builder import RiskReport, build_risk_report, write_risk_report

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_report() -> RiskReport:
    return build_risk_report(
        unguarded_risk_path=REPO_ROOT / "results/run_20260516_131022/risk_scores.json",
        guarded_results_path=REPO_ROOT / "results/run_20260516_164921/results.jsonl",
        guarded_decisions_path=REPO_ROOT / "results/run_20260516_164921/guardrail_decisions.jsonl",
        guarded_scores_path=REPO_ROOT / "results/run_20260517_115140/scores.jsonl",
        guarded_risk_path=REPO_ROOT / "results/run_20260517_115140/risk_scores.json",
        residual_analysis_path=REPO_ROOT / "results/run_20260517_115451/residual_analysis.json",
        cross_validation_path=REPO_ROOT
        / "results/cross_validation_20260516_132118/cross_validation.json",
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
