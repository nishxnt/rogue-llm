from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

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


def test_build_risk_report_exposes_first_class_findings() -> None:
    report = _build_report()

    assert report.schema_version == "1.0"
    assert report.run_metadata.attack_count == 10
    assert report.run_metadata.guarded_attack_count == 10
    assert report.system_risk.without_infrastructure_failures.percent_reduction == pytest.approx(
        0.46227709190672156
    )
    assert report.l2_fail_closed_analysis.fail_closed_count == 2
    assert report.llm06_weakness_analysis.regressed_attack_ids == ["LLM06-0004", "LLM06-0006"]
    assert report.faithfulness_coverage.scored_count == 2
    assert report.faithfulness_coverage.skip_reasons == {
        "no_retrieved_context": 1,
        "refusal_response": 1,
    }
    assert report.owasp_web_chunking_finding.contains_owasp_web_bullet_markers_per_chunk == 1.65
    assert report.owasp_web_chunking_finding.nvd_only_mean_faithfulness == 0.2783

    findings = {finding.finding_id: finding for finding in report.honest_findings}
    assert findings["l2_fail_closed_inflation"].severity == "high"
    assert findings["llm06_excessive_agency_weakness"].related_attack_ids == [
        "LLM06-0004",
        "LLM06-0006",
    ]
    assert findings["guarded_faithfulness_coverage_limitation"].severity == "medium"


def test_build_risk_report_keeps_residuals_per_attack() -> None:
    report = _build_report()

    residuals = {attack.attack_id: attack for attack in report.residual_analysis.residual_attacks}
    assert report.residual_analysis.bypass_counts == {"A": 1, "B": 2, "C": 1}
    assert sorted(residuals) == [
        "LLM01-0001",
        "LLM06-0001",
        "LLM08-0001",
        "LLM10-0001",
    ]
    assert residuals["LLM01-0001"].bypass_class == "A"
    assert residuals["LLM06-0001"].bypass_class == "B"
    assert residuals["LLM06-0001"].recommended_remediation
    assert residuals["LLM08-0001"].bypass_class == "C"


def test_write_risk_report_writes_valid_json(tmp_path: Path) -> None:
    report = _build_report()

    path = write_risk_report(report, output_root=tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "risk_report.json"
    assert payload["schema_version"] == "1.0"
    assert payload["run_metadata"]["report_tag"] == "v0.5.0-phase5"
