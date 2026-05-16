"""Typer CLI for Phase 4 evaluation and cross-validation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from src.evaluation.config import (
    CROSS_VALIDATOR_MODEL,
    DEFAULT_CONCURRENCY,
    PRIMARY_JUDGE_MODEL,
)
from src.evaluation.cross_validator import (
    DEFAULT_AGREEMENT_TOLERANCE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_FRACTION,
    cross_validate_run,
    stratified_cross_validation_sample,
)
from src.evaluation.engine import (
    AttackEvaluationInput,
    EvaluationEngine,
    EvaluationMetric,
    MetricResult,
    metric_input_hash,
)
from src.evaluation.metric_suite import LLM_GRADED_METRIC_NAMES, build_metric_suite
from src.evaluation.scorer import score_run
from src.pipeline.groq_client import (
    PRE_FLIGHT_MIN_COMBINED_RPD,
    PRE_FLIGHT_MIN_COMBINED_TPM,
    GroqClientManager,
    GroqPreflightBudget,
    combined_remaining_requests_per_day,
    combined_remaining_tokens_per_minute,
)

app = typer.Typer(help="Phase 4 evaluation scoring.")

ResultsPathOption = Annotated[
    Path,
    typer.Option(
        "--results",
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Phase 3 results.jsonl path.",
    ),
]
ScoresPathOption = Annotated[
    Path,
    typer.Option(
        "--scores",
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Primary scores.jsonl path.",
    ),
]
CachePathOption = Annotated[
    Path,
    typer.Option("--cache", help="SQLite result cache path."),
]
OutputRootOption = Annotated[
    Path,
    typer.Option("--output-root", help="Directory for score/risk artifacts."),
]
ConcurrencyOption = Annotated[
    int,
    typer.Option("--concurrency", min=1, help="Maximum concurrent metric calls."),
]
JudgeModelOption = Annotated[
    str,
    typer.Option("--judge-model", help="Primary judge model for LLM-graded metrics."),
]
CrossJudgeModelOption = Annotated[
    str,
    typer.Option("--judge-model", help="Cross-validator judge model."),
]
SampleSizeOption = Annotated[
    int,
    typer.Option("--n", "--sample", min=1, help="Number of attacks to score."),
]
SampleFractionOption = Annotated[
    float,
    typer.Option("--sample-fraction", min=0.01, max=1.0, help="Cross-validation sample fraction."),
]
SeedOption = Annotated[int, typer.Option("--seed", help="Deterministic sampling seed.")]
DeterministicOnlyOption = Annotated[
    bool,
    typer.Option(
        "--deterministic-only",
        help="Disable live LLM judges for hybrid injection/refusal metrics.",
    ),
]
SkipPreflightOption = Annotated[
    bool,
    typer.Option(
        "--skip-preflight",
        help="Skip Groq token-budget preflight probe before scoring.",
    ),
]

_DEFAULT_CACHE_PATH = Path("cache/results_cache.sqlite")
_DEFAULT_OUTPUT_ROOT = Path("results")


@app.command()
def full(
    results: ResultsPathOption,
    cache: CachePathOption = _DEFAULT_CACHE_PATH,
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    concurrency: ConcurrencyOption = DEFAULT_CONCURRENCY,
    judge_model: JudgeModelOption = PRIMARY_JUDGE_MODEL,
    deterministic_only: DeterministicOnlyOption = False,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Score every attack result and write scores.jsonl plus risk_scores.json."""
    _run_score_command(
        results_path=results,
        cache_path=cache,
        output_root=output_root,
        concurrency=concurrency,
        judge_model=judge_model,
        live_llm_judges=not deterministic_only,
        sample_size=None,
        seed=DEFAULT_RANDOM_SEED,
        mode="full",
        skip_preflight=skip_preflight,
    )


@app.command()
def sample(
    results: ResultsPathOption,
    n: SampleSizeOption,
    cache: CachePathOption = _DEFAULT_CACHE_PATH,
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    concurrency: ConcurrencyOption = DEFAULT_CONCURRENCY,
    judge_model: JudgeModelOption = PRIMARY_JUDGE_MODEL,
    seed: SeedOption = DEFAULT_RANDOM_SEED,
    deterministic_only: DeterministicOnlyOption = False,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Score a deterministic stratified sample for development iteration."""
    _run_score_command(
        results_path=results,
        cache_path=cache,
        output_root=output_root,
        concurrency=concurrency,
        judge_model=judge_model,
        live_llm_judges=not deterministic_only,
        sample_size=n,
        seed=seed,
        mode="sample",
        skip_preflight=skip_preflight,
    )


@app.command()
def resume(
    results: ResultsPathOption,
    cache: CachePathOption = _DEFAULT_CACHE_PATH,
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    concurrency: ConcurrencyOption = DEFAULT_CONCURRENCY,
    judge_model: JudgeModelOption = PRIMARY_JUDGE_MODEL,
    deterministic_only: DeterministicOnlyOption = False,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Resume scoring from cache, filling missing metric calls."""
    _run_score_command(
        results_path=results,
        cache_path=cache,
        output_root=output_root,
        concurrency=concurrency,
        judge_model=judge_model,
        live_llm_judges=not deterministic_only,
        sample_size=None,
        seed=DEFAULT_RANDOM_SEED,
        mode="resume",
        skip_preflight=skip_preflight,
    )


@app.command("cross-validate")
def cross_validate(
    results: ResultsPathOption,
    scores: ScoresPathOption,
    cache: CachePathOption = _DEFAULT_CACHE_PATH,
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    concurrency: ConcurrencyOption = DEFAULT_CONCURRENCY,
    judge_model: CrossJudgeModelOption = CROSS_VALIDATOR_MODEL,
    sample_fraction: SampleFractionOption = DEFAULT_SAMPLE_FRACTION,
    sample_size: Annotated[int | None, typer.Option("--sample-size", min=1)] = None,
    seed: SeedOption = DEFAULT_RANDOM_SEED,
    agreement_tolerance: Annotated[
        float,
        typer.Option("--agreement-tolerance", min=0.0, max=1.0),
    ] = DEFAULT_AGREEMENT_TOLERANCE,
) -> None:
    """Run cross-family validation for LLM-graded metrics."""
    report, path = asyncio.run(
        cross_validate_run(
            results_path=results,
            primary_scores_path=scores,
            cache_path=cache,
            output_root=output_root,
            sample_fraction=sample_fraction,
            sample_size=sample_size,
            seed=seed,
            judge_model=judge_model,
            metric_names=LLM_GRADED_METRIC_NAMES,
            agreement_tolerance=agreement_tolerance,
            concurrency=concurrency,
        )
    )
    typer.echo(f"Cross-validation report: {path}")
    for summary in report.metric_summaries:
        typer.echo(
            f"{summary.metric_name}: agreement={summary.agreement_rate:.3f} "
            f"compared={summary.compared_count} skipped={summary.skipped_count}"
        )


def _run_score_command(
    *,
    results_path: Path,
    cache_path: Path,
    output_root: Path,
    concurrency: int,
    judge_model: str,
    live_llm_judges: bool,
    sample_size: int | None,
    seed: int,
    mode: str,
    skip_preflight: bool,
) -> None:
    if not skip_preflight:
        preflight = probe_groq_rate_limits(model=judge_model)
        combined_rpd = combined_remaining_requests_per_day(preflight)
        combined_tpm = combined_remaining_tokens_per_minute(preflight)
        for budget in preflight:
            typer.echo(
                f"Preflight {budget.key_name}: "
                f"remaining_requests_per_day={budget.remaining_requests_per_day if budget.remaining_requests_per_day is not None else 'unknown'} "
                f"reset_requests={budget.reset_requests or 'unknown'} "
                f"remaining_tokens_per_minute={budget.remaining_tokens_per_minute if budget.remaining_tokens_per_minute is not None else 'unknown'} "
                f"reset_tokens={budget.reset_tokens or 'unknown'}"
            )
        typer.echo(
            "Preflight summary: "
            f"combined_rpd_remaining={combined_rpd} "
            f"combined_tpm_remaining={combined_tpm} "
            "tpd_remaining=not queryable (will manifest as 429 mid-run)"
        )
        if combined_rpd < PRE_FLIGHT_MIN_COMBINED_RPD or combined_tpm < PRE_FLIGHT_MIN_COMBINED_TPM:
            retry_after = _format_retry_after(preflight)
            typer.echo(
                "Insufficient preflight headroom across configured keys. "
                f"Combined RPD: {combined_rpd}. "
                f"Primary TPM: {_remaining_tpm_label(preflight, 'primary')} tokens. "
                f"Secondary TPM: {_remaining_tpm_label(preflight, 'secondary')} tokens. "
                f"Combined TPM: {combined_tpm}. "
                f"Retry after: {retry_after}."
            )
            raise typer.Exit(code=1)
    scores_path, risk_path, attack_count, system_risk = asyncio.run(
        score_results(
            results_path=results_path,
            cache_path=cache_path,
            output_root=output_root,
            concurrency=concurrency,
            judge_model=judge_model,
            live_llm_judges=live_llm_judges,
            sample_size=sample_size,
            seed=seed,
        )
    )
    typer.echo(f"{mode}: scored {attack_count} attack(s)")
    typer.echo(f"Scores: {scores_path}")
    typer.echo(f"Risk: {risk_path}")
    typer.echo(f"System Risk Score: {system_risk:.4f}")


def probe_groq_rate_limits(model: str) -> list[GroqPreflightBudget]:
    """Probe the configured Groq keys and return the exposed rate-limit headers."""
    manager = GroqClientManager()
    try:
        return manager.probe_rate_limits_sync(model=model)
    finally:
        manager.close()


async def score_results(
    *,
    results_path: Path,
    cache_path: Path,
    output_root: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    judge_model: str = PRIMARY_JUDGE_MODEL,
    live_llm_judges: bool = True,
    sample_size: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[Path, Path, int, float]:
    """Score Phase 3 results and write score/risk artifacts."""
    metrics = build_metric_suite(judge_model=judge_model, live_llm_judges=live_llm_judges)
    engine = EvaluationEngine(
        results_path=results_path,
        cache_path=cache_path,
        metrics=metrics,
        output_root=output_root,
        concurrency=concurrency,
    )
    try:
        attacks = engine.load_attack_results()
        if sample_size is not None:
            attacks = stratified_cross_validation_sample(
                attacks, sample_size=sample_size, seed=seed
            )
        scores = await _score_attacks(engine, attacks)
        scores_path = engine.write_scores(scores)
    finally:
        await close_metrics(metrics)
        engine.close()

    risk = score_run(attacks, scores)
    risk_path = scores_path.parent / "risk_scores.json"
    risk_path.write_text(risk.model_dump_json(indent=2), encoding="utf-8")
    return scores_path, risk_path, len(attacks), risk.risk_score


async def close_metrics(metrics: list[EvaluationMetric]) -> None:
    """Close async metric resources that outlive a single score call."""
    for metric in metrics:
        aclose = getattr(metric, "aclose", None)
        if callable(aclose):
            await aclose()
            continue
        close = getattr(metric, "close", None)
        if callable(close):
            close()


async def _score_attacks(
    engine: EvaluationEngine,
    attacks: list[AttackEvaluationInput],
) -> list[MetricResult]:
    semaphore = asyncio.Semaphore(engine.concurrency)

    async def score_one(metric: EvaluationMetric, attack: AttackEvaluationInput) -> MetricResult:
        async with semaphore:
            try:
                return await engine._score_with_cache(metric, attack)
            except Exception as exc:  # pragma: no cover - exercised by live judge failures
                if _is_transient_metric_error(exc):
                    raise
                result = MetricResult(
                    attack_id=attack.attack_id,
                    metric_name=metric.name,
                    score=None,
                    skipped=True,
                    reason=f"metric_error:{type(exc).__name__}",
                    evidence={"error": str(exc)},
                    judge_model=metric.judge_model,
                    judge_version=metric.judge_version,
                )
                engine.cache.set_metric_score(
                    attack_id=attack.attack_id,
                    metric_name=metric.name,
                    judge_model=metric.judge_model,
                    judge_version=metric.judge_version,
                    input_hash=metric_input_hash(metric.name, attack),
                    score=result.model_dump(),
                )
                return result

    tasks = [score_one(metric, attack) for attack in attacks for metric in engine.metrics]
    return list(await asyncio.gather(*tasks))


def _is_transient_metric_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    transient_markers = (
        "RateLimitError",
        "rate_limit",
        "rate limit",
        "429",
        "tokens per minute",
        "TPM",
        "TPD",
    )
    return any(marker in text for marker in transient_markers)


def _remaining_tpm_label(budgets: list[GroqPreflightBudget], key_name: str) -> str:
    for budget in budgets:
        if budget.key_name == key_name:
            return (
                str(budget.remaining_tokens_per_minute)
                if budget.remaining_tokens_per_minute is not None
                else "unknown"
            )
    return "not configured"


def _format_retry_after(budgets: list[GroqPreflightBudget]) -> str:
    parts = [
        f"{budget.key_name}={budget.reset_tokens}" for budget in budgets if budget.reset_tokens
    ]
    return ", ".join(parts) if parts else "unknown"


@app.command(hidden=True)
def _commands() -> None:
    """Keep Typer in multi-command mode so command names are explicit."""


if __name__ == "__main__":
    app()
