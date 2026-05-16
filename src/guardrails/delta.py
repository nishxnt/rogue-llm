"""Delta reporting helpers for Phase 5 guarded vs unguarded comparisons."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from src.evaluation.scorer import CategoryRiskScore, SystemRiskScore


class CategoryRiskDelta(BaseModel):
    """Risk-score delta for one OWASP category."""

    owasp_category: str
    baseline_risk_score: float
    guarded_risk_score: float
    delta: float


class DeltaReport(BaseModel):
    """System and category deltas between guarded and unguarded runs."""

    baseline_risk_score: float
    guarded_risk_score: float
    system_delta: float
    category_deltas: list[CategoryRiskDelta]


def build_delta_report(
    baseline: SystemRiskScore,
    guarded: SystemRiskScore,
) -> DeltaReport:
    """Compute system/category risk reduction from unguarded to guarded."""
    baseline_by_category = {score.owasp_category: score for score in baseline.category_scores}
    guarded_by_category = {score.owasp_category: score for score in guarded.category_scores}
    categories = sorted(set(baseline_by_category) | set(guarded_by_category))
    category_deltas: list[CategoryRiskDelta] = []
    for category in categories:
        baseline_score = baseline_by_category.get(category, _empty_category(category))
        guarded_score = guarded_by_category.get(category, _empty_category(category))
        category_deltas.append(
            CategoryRiskDelta(
                owasp_category=category,
                baseline_risk_score=baseline_score.risk_score,
                guarded_risk_score=guarded_score.risk_score,
                delta=baseline_score.risk_score - guarded_score.risk_score,
            )
        )
    return DeltaReport(
        baseline_risk_score=baseline.risk_score,
        guarded_risk_score=guarded.risk_score,
        system_delta=baseline.risk_score - guarded.risk_score,
        category_deltas=category_deltas,
    )


def write_delta_report(
    report: DeltaReport,
    *,
    output_root: Path | str,
) -> Path:
    """Write one delta report artifact under a timestamped results run directory."""
    run_dir = Path(output_root) / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "delta_report.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_risk_report(path: Path | str) -> SystemRiskScore:
    """Load a risk_scores.json artifact."""
    return SystemRiskScore.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _empty_category(category: str) -> CategoryRiskScore:
    return CategoryRiskScore(
        owasp_category=category,
        attack_count=0,
        risk_score=0.0,
        weight=1.0,
    )
