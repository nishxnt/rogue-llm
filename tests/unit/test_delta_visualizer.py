from __future__ import annotations

from src.reporting.delta_visualizer import (
    build_bypass_breakdown_bar,
    build_cross_validator_agreement_bar,
    build_decision_distribution_bar,
    build_layer_attribution_donut,
    build_owasp_web_chunking_chart,
    build_risk_heatmap,
)
from src.reporting.report_builder import RiskReport, build_risk_report
from tests.unit.reporting_fixture_paths import (
    CROSS_VALIDATION,
    GUARDED_DECISIONS,
    GUARDED_RESULTS,
    GUARDED_RISK,
    GUARDED_SCORES,
    RESIDUAL_ANALYSIS,
    UNGUARDED_RISK,
)


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
    assert sum(layer_chart.data[0]["values"]) == report.run_metadata.guarded_attack_count
    assert sum(decision_chart.data[0]["y"]) == report.run_metadata.guarded_attack_count


def test_build_bypass_chart_groups_by_category() -> None:
    figure = build_bypass_breakdown_bar(_build_report())

    assert len(figure.data) == 3
    assert {trace["name"] for trace in figure.data} == {"Bypass A", "Bypass B", "Bypass C"}
    assert list(figure.data[1]["x"]) == ["LLM01", "LLM06", "LLM08", "LLM10"]


def test_build_cross_validator_and_owasp_web_charts_use_numeric_report_fields() -> None:
    report = _build_report()

    agreement_chart = build_cross_validator_agreement_bar(report)
    owasp_chart = build_owasp_web_chunking_chart(report)

    assert len(agreement_chart.data[0]["x"]) >= 3
    assert all(0.0 <= value <= 1.0 for value in agreement_chart.data[0]["y"])
    assert list(owasp_chart.data[0]["y"]) == [1.65, 0.03, 0.34]
    assert list(owasp_chart.data[1]["y"]) == [0.1561, 0.2783, 0.2602]
