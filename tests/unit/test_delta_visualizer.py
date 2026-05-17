from __future__ import annotations

from pathlib import Path

from src.reporting.delta_visualizer import (
    build_bypass_breakdown_bar,
    build_cross_validator_agreement_bar,
    build_decision_distribution_bar,
    build_layer_attribution_donut,
    build_owasp_web_chunking_chart,
    build_risk_heatmap,
)
from src.reporting.report_builder import RiskReport, build_risk_report

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


def test_build_risk_heatmap_includes_all_owasp_categories() -> None:
    figure = build_risk_heatmap(_build_report())

    assert len(figure.data) == 1
    assert list(figure.data[0]["x"]) == [f"LLM{index:02d}" for index in range(1, 11)]
    assert list(figure.data[0]["customdata"][0]) == [
        f"LLM{index:02d}:2025" for index in range(1, 11)
    ]
    assert list(figure.data[0]["y"]) == ["Unguarded", "Guarded", "Delta"]


def test_build_distribution_charts_have_expected_counts() -> None:
    report = _build_report()

    layer_chart = build_layer_attribution_donut(report)
    decision_chart = build_decision_distribution_bar(report)

    assert list(layer_chart.data[0]["labels"]) == [
        "L1 blocked",
        "L2 blocked",
        "L2 fail-closed",
        "L3 blocked/modified",
        "Allowed through",
        "LLM08 path",
    ]
    assert sum(layer_chart.data[0]["values"]) == 175
    assert sum(decision_chart.data[0]["y"]) == 175


def test_build_bypass_chart_groups_by_category() -> None:
    figure = build_bypass_breakdown_bar(_build_report())

    assert len(figure.data) == 3
    assert {trace["name"] for trace in figure.data} == {"Bypass A", "Bypass B", "Bypass C"}
    assert list(figure.data[1]["x"]) == ["LLM06", "LLM08", "LLM10"]


def test_build_cross_validator_and_owasp_web_charts_use_numeric_report_fields() -> None:
    report = _build_report()

    agreement_chart = build_cross_validator_agreement_bar(report)
    owasp_chart = build_owasp_web_chunking_chart(report)

    assert len(agreement_chart.data[0]["x"]) >= 3
    assert all(0.0 <= value <= 1.0 for value in agreement_chart.data[0]["y"])
    assert list(owasp_chart.data[0]["y"]) == [1.65, 0.03, 0.34]
    assert list(owasp_chart.data[1]["y"]) == [0.1561, 0.2783, 0.2602]
