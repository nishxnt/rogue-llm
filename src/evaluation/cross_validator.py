"""Cross-family judge validation for Phase 4 metric scores."""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from src.evaluation.config import CROSS_VALIDATOR_MODEL, DEFAULT_CONCURRENCY
from src.evaluation.engine import (
    AttackEvaluationInput,
    EvaluationEngine,
    EvaluationMetric,
    MetricResult,
)
from src.evaluation.metric_suite import LLM_GRADED_METRIC_NAMES, build_metric_suite

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = structlog.get_logger()

DEFAULT_SAMPLE_FRACTION = 0.10
DEFAULT_AGREEMENT_TOLERANCE = 0.20
DEFAULT_RANDOM_SEED = 20260515


class CrossValidationComparison(BaseModel):
    """Primary-vs-cross judge comparison for one attack metric."""

    attack_id: str
    owasp_category: str
    metric_name: str
    primary_score: float | None = None
    cross_score: float | None = None
    absolute_delta: float | None = None
    agreed: bool = False
    skipped: bool = False
    reason: str | None = None


class CrossValidationMetricSummary(BaseModel):
    """Agreement summary for one LLM-graded metric."""

    metric_name: str
    compared_count: int
    agreement_count: int
    agreement_rate: float
    mean_absolute_delta: float | None = None
    skipped_count: int = 0


class CrossValidationReport(BaseModel):
    """Structured cross-validation report artifact."""

    generated_at: str
    primary_scores_path: str
    results_path: str
    cache_path: str
    cross_validator_model: str
    sample_fraction: float
    sample_size: int
    agreement_tolerance: float
    metric_summaries: list[CrossValidationMetricSummary]
    comparisons: list[CrossValidationComparison] = Field(default_factory=list)


async def cross_validate_run(
    *,
    results_path: Path | str,
    primary_scores_path: Path | str,
    cache_path: Path | str,
    output_root: Path | str = "results",
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    sample_size: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    judge_model: str = CROSS_VALIDATOR_MODEL,
    metric_names: Sequence[str] = LLM_GRADED_METRIC_NAMES,
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[CrossValidationReport, Path]:
    """Run cross-family validation over a stratified attack sample."""
    if sample_fraction <= 0.0 or sample_fraction > 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive when provided")
    if agreement_tolerance < 0.0:
        raise ValueError("agreement_tolerance must be non-negative")

    metric_name_set = set(metric_names)
    primary_scores = _load_primary_scores(primary_scores_path, metric_name_set)
    metrics = [
        metric
        for metric in build_metric_suite(judge_model=judge_model, live_llm_judges=True)
        if metric.name in metric_name_set
    ]

    engine = EvaluationEngine(
        results_path=results_path,
        cache_path=cache_path,
        metrics=metrics,
        output_root=output_root,
        concurrency=concurrency,
    )
    try:
        attacks = engine.load_attack_results()
        sampled_attacks = stratified_cross_validation_sample(
            attacks,
            sample_fraction=sample_fraction,
            sample_size=sample_size,
            seed=seed,
        )
        cross_scores = await _score_sample(engine, sampled_attacks, metrics)
    finally:
        await _close_metrics(metrics)
        engine.close()

    comparisons = _compare_scores(
        sampled_attacks=sampled_attacks,
        metric_names=metric_names,
        primary_scores=primary_scores,
        cross_scores=cross_scores,
        agreement_tolerance=agreement_tolerance,
    )
    report = CrossValidationReport(
        generated_at=datetime.now(UTC).isoformat(),
        primary_scores_path=str(primary_scores_path),
        results_path=str(results_path),
        cache_path=str(cache_path),
        cross_validator_model=judge_model,
        sample_fraction=sample_fraction,
        sample_size=len(sampled_attacks),
        agreement_tolerance=agreement_tolerance,
        metric_summaries=_summarize(comparisons, metric_names),
        comparisons=comparisons,
    )
    path = _write_report(report, output_root)
    return report, path


def stratified_cross_validation_sample(
    attacks: Sequence[AttackEvaluationInput],
    *,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    sample_size: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[AttackEvaluationInput]:
    """Return a deterministic category-stratified sample."""
    if not attacks:
        return []
    target_size = sample_size or max(1, round(len(attacks) * sample_fraction))
    target_size = min(target_size, len(attacks))

    by_category: dict[str, list[AttackEvaluationInput]] = defaultdict(list)
    for attack in attacks:
        by_category[attack.owasp_category].append(attack)

    rng = random.Random(seed)
    for group in by_category.values():
        group.sort(key=lambda attack: attack.attack_id)
        rng.shuffle(group)

    sampled: list[AttackEvaluationInput] = []
    categories = sorted(by_category)
    while len(sampled) < target_size:
        progressed = False
        for category in categories:
            if by_category[category]:
                sampled.append(by_category[category].pop(0))
                progressed = True
                if len(sampled) == target_size:
                    break
        if not progressed:
            break

    return sorted(sampled, key=lambda attack: attack.attack_id)


def _load_primary_scores(
    primary_scores_path: Path | str,
    metric_names: set[str],
) -> dict[tuple[str, str], MetricResult]:
    path = Path(primary_scores_path)
    if not path.exists():
        raise FileNotFoundError(f"primary scores not found: {path}")
    scores: dict[tuple[str, str], MetricResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = MetricResult.model_validate(json.loads(line))
        if result.metric_name in metric_names:
            scores[(result.attack_id, result.metric_name)] = result
    return scores


async def _score_sample(
    engine: EvaluationEngine,
    sampled_attacks: Sequence[AttackEvaluationInput],
    metrics: Sequence[EvaluationMetric],
) -> dict[tuple[str, str], MetricResult]:
    semaphore = asyncio.Semaphore(engine.concurrency)

    async def score_one(metric: EvaluationMetric, attack: AttackEvaluationInput) -> MetricResult:
        async with semaphore:
            try:
                result = await engine._score_with_cache(metric, attack)
            except Exception as exc:  # pragma: no cover - exercised by live runs
                metric_name = str(getattr(metric, "name", "unknown"))
                judge_model = str(getattr(metric, "judge_model", "unknown"))
                judge_version = str(getattr(metric, "judge_version", "unknown"))
                log.warning(
                    "Cross-validation metric failed",
                    attack_id=attack.attack_id,
                    metric=metric_name,
                    error=str(exc),
                )
                return MetricResult(
                    attack_id=attack.attack_id,
                    metric_name=metric_name,
                    score=None,
                    skipped=True,
                    reason=f"cross_validator_error:{type(exc).__name__}",
                    evidence={"error": str(exc)},
                    judge_model=judge_model,
                    judge_version=judge_version,
                )
            return result

    tasks = [score_one(metric, attack) for attack in sampled_attacks for metric in metrics]
    results = await asyncio.gather(*tasks)
    return {(result.attack_id, result.metric_name): result for result in results}


def _compare_scores(
    *,
    sampled_attacks: Sequence[AttackEvaluationInput],
    metric_names: Sequence[str],
    primary_scores: Mapping[tuple[str, str], MetricResult],
    cross_scores: Mapping[tuple[str, str], MetricResult],
    agreement_tolerance: float,
) -> list[CrossValidationComparison]:
    comparisons: list[CrossValidationComparison] = []
    categories_by_attack = {attack.attack_id: attack.owasp_category for attack in sampled_attacks}
    for attack in sampled_attacks:
        for metric_name in metric_names:
            primary = primary_scores.get((attack.attack_id, metric_name))
            cross = cross_scores.get((attack.attack_id, metric_name))
            comparisons.append(
                _comparison(
                    attack_id=attack.attack_id,
                    owasp_category=categories_by_attack[attack.attack_id],
                    metric_name=metric_name,
                    primary=primary,
                    cross=cross,
                    agreement_tolerance=agreement_tolerance,
                )
            )
    return comparisons


def _comparison(
    *,
    attack_id: str,
    owasp_category: str,
    metric_name: str,
    primary: MetricResult | None,
    cross: MetricResult | None,
    agreement_tolerance: float,
) -> CrossValidationComparison:
    if primary is None:
        return CrossValidationComparison(
            attack_id=attack_id,
            owasp_category=owasp_category,
            metric_name=metric_name,
            skipped=True,
            reason="missing_primary_score",
        )
    if cross is None:
        return CrossValidationComparison(
            attack_id=attack_id,
            owasp_category=owasp_category,
            metric_name=metric_name,
            primary_score=primary.score,
            skipped=True,
            reason="missing_cross_score",
        )
    if primary.skipped or cross.skipped or primary.score is None or cross.score is None:
        agreed = primary.skipped and cross.skipped and primary.reason == cross.reason
        return CrossValidationComparison(
            attack_id=attack_id,
            owasp_category=owasp_category,
            metric_name=metric_name,
            primary_score=primary.score,
            cross_score=cross.score,
            agreed=agreed,
            skipped=True,
            reason=cross.reason or primary.reason or "skipped_score",
        )

    delta = abs(primary.score - cross.score)
    return CrossValidationComparison(
        attack_id=attack_id,
        owasp_category=owasp_category,
        metric_name=metric_name,
        primary_score=primary.score,
        cross_score=cross.score,
        absolute_delta=delta,
        agreed=delta <= agreement_tolerance,
    )


def _summarize(
    comparisons: Sequence[CrossValidationComparison],
    metric_names: Sequence[str],
) -> list[CrossValidationMetricSummary]:
    summaries: list[CrossValidationMetricSummary] = []
    for metric_name in metric_names:
        metric_comparisons = [item for item in comparisons if item.metric_name == metric_name]
        comparable = [item for item in metric_comparisons if not item.skipped]
        deltas = [item.absolute_delta for item in comparable if item.absolute_delta is not None]
        agreement_count = sum(1 for item in comparable if item.agreed)
        compared_count = len(comparable)
        summaries.append(
            CrossValidationMetricSummary(
                metric_name=metric_name,
                compared_count=compared_count,
                agreement_count=agreement_count,
                agreement_rate=(agreement_count / compared_count) if compared_count else 0.0,
                mean_absolute_delta=(sum(deltas) / len(deltas)) if deltas else None,
                skipped_count=len(metric_comparisons) - compared_count,
            )
        )
    return summaries


async def _close_metrics(metrics: Sequence[EvaluationMetric]) -> None:
    for metric in metrics:
        aclose = getattr(metric, "aclose", None)
        if callable(aclose):
            await aclose()
            continue
        close = getattr(metric, "close", None)
        if callable(close):
            close()


def _write_report(report: CrossValidationReport, output_root: Path | str) -> Path:
    run_dir = Path(output_root) / f"cross_validation_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "cross_validation.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    log.info("Cross-validation report written", path=str(path))
    return path
