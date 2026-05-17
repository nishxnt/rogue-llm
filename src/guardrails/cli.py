"""Typer CLI for Phase 5 guardrail execution and delta evaluation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from src.config import get_settings
from src.evaluation.cli import score_results
from src.guardrails.delta import build_delta_report, load_risk_report, write_delta_report
from src.guardrails.guardrail_target import GuardrailTarget
from src.guardrails.residual_analysis import (
    analyze_residual_vulnerabilities,
    write_timestamped_residual_analysis,
)
from src.pipeline.attack_runner import AttackRunner
from src.pipeline.groq_client import (
    GroqClientManager,
    GroqPreflightBudget,
    combined_remaining_requests_per_day,
    combined_remaining_tokens_per_minute,
)
from src.target_system.rag_chatbot import RAGChatbot

app = typer.Typer(help="Phase 5 guardrail execution and delta evaluation.")

DatasetPathOption = Annotated[
    Path,
    typer.Option("--dataset", file_okay=True, dir_okay=False, readable=True),
]
PolicyPathOption = Annotated[
    Path,
    typer.Option("--policy", file_okay=True, dir_okay=False, readable=True),
]
ResultsPathOption = Annotated[
    Path,
    typer.Option("--results", file_okay=True, dir_okay=False, readable=True),
]
RiskPathOption = Annotated[
    Path,
    typer.Option("--risk", file_okay=True, dir_okay=False, readable=True),
]
ThresholdOption = Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)]
CachePathOption = Annotated[Path, typer.Option("--cache")]
OutputRootOption = Annotated[Path, typer.Option("--output-root")]
ConcurrencyOption = Annotated[int, typer.Option("--concurrency", min=1)]
SampleOption = Annotated[int | None, typer.Option("--sample", min=1)]
SkipPreflightOption = Annotated[
    bool,
    typer.Option("--skip-preflight", help="Skip Groq preflight check before live calls."),
]
JudgeModelOption = Annotated[str | None, typer.Option("--judge-model")]

_DEFAULT_DATASET_PATH = Path("attacks/v1/dataset.jsonl")
_DEFAULT_POLICY_PATH = Path("src/guardrails/policy.yaml")
_DEFAULT_CACHE_PATH = Path("cache/results_cache.sqlite")
_DEFAULT_OUTPUT_ROOT = Path("results")
_MIN_COMBINED_RPD = 100
_MIN_COMBINED_TPM = 4_000


@app.command("run-attacks")
def run_attacks(
    dataset: DatasetPathOption = _DEFAULT_DATASET_PATH,
    policy: PolicyPathOption = _DEFAULT_POLICY_PATH,
    cache: CachePathOption = _DEFAULT_CACHE_PATH,
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    concurrency: ConcurrencyOption = 1,
    sample: SampleOption = None,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Run the full attack dataset against the guardrail-wrapped target."""
    settings = get_settings()
    if not skip_preflight:
        budgets = probe_groq_rate_limits(model=settings.safety_model)
        _emit_preflight(budgets)
        _abort_if_preflight_low(budgets)
    results_path = asyncio.run(
        _run_guarded_attacks(
            dataset_path=dataset,
            policy_path=policy,
            cache_path=cache,
            output_root=output_root,
            concurrency=concurrency,
            sample=sample,
        )
    )
    typer.echo(f"Guarded attack results: {results_path}")


@app.command()
def evaluate(
    results: ResultsPathOption,
    cache: CachePathOption = _DEFAULT_CACHE_PATH,
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    concurrency: ConcurrencyOption = 1,
    judge_model: JudgeModelOption = None,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Run the Phase 4 evaluation engine against guarded results."""
    resolved_judge_model = judge_model or get_settings().judge_model
    if not skip_preflight:
        budgets = probe_groq_rate_limits(model=resolved_judge_model)
        _emit_preflight(budgets)
        _abort_if_preflight_low(budgets)
    scores_path, risk_path, attack_count, system_risk = asyncio.run(
        score_results(
            results_path=results,
            cache_path=cache,
            output_root=output_root,
            concurrency=concurrency,
            judge_model=resolved_judge_model,
            live_llm_judges=True,
        )
    )
    typer.echo(f"evaluate: scored {attack_count} attack(s)")
    typer.echo(f"Scores: {scores_path}")
    typer.echo(f"Risk: {risk_path}")
    typer.echo(f"System Risk Score: {system_risk:.4f}")


@app.command("delta-report")
def delta_report(
    baseline_risk: Annotated[
        Path,
        typer.Option("--baseline-risk", file_okay=True, dir_okay=False, readable=True),
    ],
    guarded_risk: Annotated[
        Path,
        typer.Option("--guarded-risk", file_okay=True, dir_okay=False, readable=True),
    ],
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
) -> None:
    """Compare guarded and unguarded risk reports and write delta_report.json."""
    baseline = load_risk_report(baseline_risk)
    guarded = load_risk_report(guarded_risk)
    report = build_delta_report(baseline, guarded)
    path = write_delta_report(report, output_root=output_root)
    typer.echo(f"Delta report: {path}")
    typer.echo(f"System delta: {report.system_delta:.4f}")
    for category in report.category_deltas:
        typer.echo(f"{category.owasp_category}: delta={category.delta:.4f}")


@app.command("residual-report")
def residual_report(
    guarded_results: Annotated[
        Path,
        typer.Option("--guarded-results", file_okay=True, dir_okay=False, readable=True),
    ],
    guarded_decisions: Annotated[
        Path,
        typer.Option("--guarded-decisions", file_okay=True, dir_okay=False, readable=True),
    ],
    guarded_risk: Annotated[
        Path,
        typer.Option("--guarded-risk", file_okay=True, dir_okay=False, readable=True),
    ],
    unguarded_risk: Annotated[
        Path,
        typer.Option("--unguarded-risk", file_okay=True, dir_okay=False, readable=True),
    ],
    output_root: OutputRootOption = _DEFAULT_OUTPUT_ROOT,
    threshold: ThresholdOption = 0.5,
) -> None:
    """Write residual vulnerability analysis from guarded and unguarded artifacts."""
    report = analyze_residual_vulnerabilities(
        guarded_results_path=guarded_results,
        guarded_risk_path=guarded_risk,
        guarded_decisions_path=guarded_decisions,
        unguarded_risk_path=unguarded_risk,
        threshold=threshold,
    )
    json_path, md_path = write_timestamped_residual_analysis(report, output_root=output_root)
    typer.echo(f"Residual analysis: {json_path}")
    typer.echo(f"Residual summary: {md_path}")
    typer.echo(f"Residual attacks: {report.residual_count}")
    typer.echo(
        "Bypass counts: "
        f"A={report.bypass_counts.get('A', 0)} "
        f"B={report.bypass_counts.get('B', 0)} "
        f"C={report.bypass_counts.get('C', 0)}"
    )


def probe_groq_rate_limits(model: str) -> list[GroqPreflightBudget]:
    """Probe the configured Groq keys and return exposed rate-limit headers."""
    manager = GroqClientManager()
    try:
        return manager.probe_rate_limits_sync(model=model)
    finally:
        manager.close()


async def _run_guarded_attacks(
    *,
    dataset_path: Path,
    policy_path: Path,
    cache_path: Path,
    output_root: Path,
    concurrency: int,
    sample: int | None,
) -> Path:
    target = GuardrailTarget(
        base_rag_chatbot=RAGChatbot(),
        policy_path=policy_path,
    )
    runner = AttackRunner(
        target_system=target,
        dataset_path=dataset_path,
        cache_path=cache_path,
        results_root=output_root,
        concurrency=concurrency,
    )
    try:
        if sample is None:
            await runner.run()
        else:
            await runner.run_with_sample(sample)
        run_dir = max(output_root.glob("run_*"), key=lambda path: path.name)
        return run_dir / "results.jsonl"
    finally:
        await target.aclose()
        runner.close()


def _emit_preflight(budgets: list[GroqPreflightBudget]) -> None:
    for budget in budgets:
        typer.echo(
            f"Preflight {budget.key_name}: "
            f"remaining_requests_per_day={budget.remaining_requests_per_day if budget.remaining_requests_per_day is not None else 'unknown'} "
            f"remaining_tokens_per_minute={budget.remaining_tokens_per_minute if budget.remaining_tokens_per_minute is not None else 'unknown'} "
            f"reset_requests={budget.reset_requests or 'unknown'} "
            f"reset_tokens={budget.reset_tokens or 'unknown'}"
        )
    typer.echo(
        "Preflight summary: "
        f"combined_rpd_remaining={combined_remaining_requests_per_day(budgets)} "
        f"combined_tpm_remaining={combined_remaining_tokens_per_minute(budgets)} "
        "tpd_remaining=not queryable (will manifest as 429 mid-run)"
    )


def _abort_if_preflight_low(budgets: list[GroqPreflightBudget]) -> None:
    combined_rpd = combined_remaining_requests_per_day(budgets)
    combined_tpm = combined_remaining_tokens_per_minute(budgets)
    if combined_rpd >= _MIN_COMBINED_RPD and combined_tpm >= _MIN_COMBINED_TPM:
        return
    typer.echo(
        "Insufficient preflight headroom across configured keys. "
        f"Combined RPD: {combined_rpd}. Combined TPM: {combined_tpm}. "
        "TPD remaining is not queryable and will manifest as 429 mid-run."
    )
    raise typer.Exit(code=1)


@app.command(hidden=True)
def _commands() -> None:
    """Keep Typer in multi-command mode so command names remain explicit."""


if __name__ == "__main__":
    app()
