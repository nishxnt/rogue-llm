#!/usr/bin/env python3
"""Build one CI PR safety comment from sample guarded artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.evaluation.scorer import SystemRiskScore
from src.reporting.report_builder import RiskReport, _aggregate_risk_modes

COMMENT_MARKER = "<!-- rogue-llm-risk-comment -->"
RESIDUAL_THRESHOLD = 0.5
SMOKE_NOTE = (
    "This is a CI regression check, not a full safety evaluation. "
    "Full 175-attack runs are performed manually after merge to dev; "
    "see results/baseline/risk_report.json for the canonical baseline."
)


def main() -> None:
    args = _parse_args()
    if args.status == "tpd_exhausted":
        body = _tpd_skip_comment(run_url=args.run_url)
    elif args.status == "workflow_error":
        body = _workflow_error_comment(run_url=args.run_url)
    else:
        body = _success_comment(
            risk_path=args.risk,
            results_path=args.results,
            baseline_path=args.baseline,
            run_url=args.run_url,
            sample_size=args.sample_size,
        )
    args.output.write_text(body, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        choices=["success", "tpd_exhausted", "workflow_error"],
        default="success",
    )
    parser.add_argument("--risk", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _tpd_skip_comment(*, run_url: str) -> str:
    return "\n".join(
        [
            COMMENT_MARKER,
            "### CI safety smoke check (25-attack subset)",
            "",
            "<details>",
            "<summary>Provider-limited run</summary>",
            "",
            f"{SMOKE_NOTE}",
            "",
            "CI safety evaluation skipped due to provider TPD exhaustion.",
            "",
            f"Workflow run: [artifacts and logs]({run_url})",
            "</details>",
            "",
        ]
    )


def _workflow_error_comment(*, run_url: str) -> str:
    return "\n".join(
        [
            COMMENT_MARKER,
            "### CI safety smoke check (25-attack subset)",
            "",
            "<details>",
            "<summary>Workflow error</summary>",
            "",
            f"{SMOKE_NOTE}",
            "",
            "CI safety evaluation failed unexpectedly. Inspect the workflow logs and artifacts before trusting this PR's safety posture.",
            "",
            f"Workflow run: [artifacts and logs]({run_url})",
            "</details>",
            "",
        ]
    )


def _success_comment(
    *,
    risk_path: Path | None,
    results_path: Path | None,
    baseline_path: Path | None,
    run_url: str,
    sample_size: int,
) -> str:
    if risk_path is None or results_path is None:
        raise ValueError("risk_path and results_path are required for success comments")

    risk = SystemRiskScore.model_validate_json(risk_path.read_text(encoding="utf-8"))
    aggregated = _aggregate_risk_modes(risk_path)
    results = _load_jsonl(results_path)
    decision_counts = _decision_counts(results)
    residual_attacks = [
        score
        for score in sorted(
            risk.attack_scores,
            key=lambda score: (score.vulnerability_score, score.attack_id),
            reverse=True,
        )
        if score.vulnerability_score >= RESIDUAL_THRESHOLD
    ]
    baseline_text, baseline_by_category = _baseline_summary(baseline_path)

    lines = [
        COMMENT_MARKER,
        "### CI safety smoke check (25-attack subset)",
        "",
        "<details>",
        f"<summary>{sample_size}-attack guarded sample on this branch</summary>",
        "",
        f"Workflow run: [artifacts and logs]({run_url})",
        "",
        SMOKE_NOTE,
        "",
        f"System Risk (with infrastructure failures): `{aggregated.with_infrastructure_failures.system:.4f}`",
        f"System Risk (without infrastructure failures): `{aggregated.without_infrastructure_failures.system:.4f}`",
        "",
        baseline_text,
        "",
        "#### Decision distribution",
        "",
        "| Decision | Count |",
        "|---|---:|",
    ]
    for decision, count in decision_counts:
        lines.append(f"| `{decision}` | {count} |")

    lines.extend(
        [
            "",
            "#### Per-category sample Risk summary",
            "",
            "| Category | Attacks | Sample Risk | Δ vs baseline guarded |",
            "|---|---:|---:|---:|",
        ]
    )
    for category_name, category in aggregated.without_infrastructure_failures.categories.items():
        baseline_value = baseline_by_category.get(category_name)
        delta_cell = (
            f"{category.risk_score - baseline_value:+.4f}" if baseline_value is not None else "n/a"
        )
        attack_count = next(
            (
                score.attack_count
                for score in risk.category_scores
                if score.owasp_category == category_name
            ),
            0,
        )
        lines.append(
            f"| `{category_name}` | {attack_count} | {category.risk_score:.4f} | {delta_cell} |"
        )

    lines.extend(
        [
            "",
            "#### Residual attack IDs (sample rows scoring ≥ 0.5)",
            "",
        ]
    )
    if residual_attacks:
        for attack in residual_attacks[:10]:
            decision = _decision_for_attack(results, attack.attack_id)
            lines.append(
                f"- `{attack.attack_id}` ({attack.owasp_category}, score={attack.vulnerability_score:.2f}, decision=`{decision}`)"
            )
    else:
        lines.append("- none")

    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def _baseline_summary(
    baseline_path: Path | None,
) -> tuple[str, dict[str, float]]:
    if baseline_path is None or not baseline_path.exists():
        return "No baseline available for comparison; this is the initial run.", {}
    report = RiskReport.model_validate_json(baseline_path.read_text(encoding="utf-8"))
    baseline_by_category = {
        row.owasp_category: row.guarded_risk_score
        for row in report.per_category_risk.without_infrastructure_failures
    }
    system_guarded = report.system_risk.without_infrastructure_failures.guarded_risk_score
    return (
        "Baseline guarded Risk (without infrastructure failures): "
        f"`{system_guarded:.4f}` from `{baseline_path}`",
        baseline_by_category,
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _decision_counts(results: list[dict[str, object]]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in results:
        decision = row.get("guardrail_decision")
        if isinstance(decision, str) and decision:
            counts[decision] += 1
            continue
        counts["none"] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _decision_for_attack(results: list[dict[str, object]], attack_id: str) -> str:
    for row in results:
        if row.get("attack_id") != attack_id:
            continue
        decision = row.get("guardrail_decision")
        if isinstance(decision, str) and decision:
            return decision
        return "none"
    return "unknown"


if __name__ == "__main__":
    main()
