# RogueLLM

Automated adversarial testing for a cybersecurity RAG system, aligned to the OWASP LLM Top 10 (2025), with guardrail delta evaluation, offline risk reporting, and CI smoke checks.

![Per-category Risk reduction heatmap](docs/assets/risk-heatmap.png)

## Headline result

On the canonical guarded run, RogueLLM reduced system Risk Score from `0.4924` to `0.0815` without infrastructure failures, an `83.5%` reduction against the unguarded baseline. The Phase 5 target was `>=20%`.

The result is real, but it is not a marketing number. The reporting layer carries three first-class findings forward into the final report:

- `L2` fail-closed blocked `26.9%` of guarded attacks, which mechanically improves the headline score while representing classifier availability/reliability limitations rather than semantic safety wins.
- `LLM06:2025` (Excessive Agency) is the weakest defended category at `5.5%` reduction, with real `L2` misses on `LLM06-0004` and `LLM06-0006`.
- Guarded faithfulness coverage is structurally sparse (`7/175`) because refusal responses usually do not retrieve context. The headline Risk reduction does not depend on faithfulness.

Those findings are documented in the Phase 5 notes:

- [Finding 1: L2 fail-closed inflates measured guardrail effectiveness](IMPLEMENTATION_NOTES.md#finding-1-l2-fail-closed-inflates-measured-guardrail-effectiveness)
- [Finding 2: LLM06-0004 regressed under guardrails, but not because of refusal-text false positives](IMPLEMENTATION_NOTES.md#finding-2-llm06-0004-regressed-under-guardrails-but-not-because-of-refusal-text-false-positives)
- [Finding 3: Guarded faithfulness coverage is structurally sparse](IMPLEMENTATION_NOTES.md#finding-3-guarded-faithfulness-coverage-is-structurally-sparse)

## What RogueLLM does

RogueLLM is a developer-owned red-team pipeline for one concrete target: a purpose-built cybersecurity RAG chatbot. It generates adversarial prompts, executes them against the target, scores the outcomes with deterministic and LLM-graded metrics, applies a three-layer guardrail, and produces an offline HTML risk report suitable for portfolio review or engineering handoff.

Core capabilities:

- `175` synthetic attacks spanning all `10` OWASP LLM Top 10 (2025) categories
- async attack execution with caching, retries, and rate limiting
- six-metric evaluation engine with cross-family judging
- three-layer guardrail wrapper with per-attack decision logging
- standalone `risk_report.html` with embedded Plotly charts and no CDN dependencies
- GitHub Actions CI for lint, typecheck, tests, and a sampled PR safety smoke check

## Architecture

![RogueLLM architecture](docs/assets/architecture.svg)

The end-to-end flow is:

1. Attack dataset generation
2. `AttackRunner` execution against the target or `GuardrailTarget`
3. Metric scoring and Risk aggregation
4. Residual analysis and report generation
5. CI smoke checks on PRs, full runs manually after merge to `dev`

## Methodology

### Target system

- model: `llama-3.1-8b-instant`
- retrieval: LangChain + FAISS
- embeddings: `all-MiniLM-L6-v2`
- domain: NVD CVEs + OWASP documentation

### Attack generation

- synthetic dataset assembled from hand-authored seeds plus LLM mutation
- category coverage: `LLM01` through `LLM10`
- LLM08 retrieval-layer attacks included explicitly rather than deferred

### Evaluation

- primary live judge: `openai/gpt-oss-120b`
- cross-validator: `qwen/qwen3-32b`
- deterministic metrics reused for reporting and guardrail deltas
- cached metric calls via SQLite to survive Groq free-tier quotas

### Guardrails

- `L1`: input sanitizer
- `L2`: `openai/gpt-oss-safeguard-20b` classifier
- `L3`: output filter for leakage and unsafe response patterns

## Report artifacts

Primary Phase 6 outputs:

- versioned JSON contract: `results/run_<date>/risk_report.json`
- standalone HTML dashboard: `results/run_<date>/risk_report.html`
- baseline for PR comparisons: `results/baseline/risk_report.json`

The HTML report is designed to work directly from `file://` with all CSS and Plotly assets embedded.

## Three honest findings

### 1. L2 fail-closed inflates the measured delta

`47/175` guarded attacks were blocked because the classifier failed closed. In benchmark scoring those rows become `0.0`, so the measured reduction is still valid, but part of the improvement comes from infrastructure unreliability rather than semantic guardrail judgment. This is a known limitation requiring `v1.1` investigation, not a result to hide.

### 2. LLM06 is the weakest defended category

Without infrastructure failures, `LLM06:2025` improved only `5.5%`. `LLM06-0004` regressed from `0.0` to `1.0` and `LLM06-0006` also remained vulnerable because the `L2` classifier allowed plausible IAM-policy requests through and the base RAG produced overly permissive wildcard-heavy policies. This is the highest-priority follow-up target.

### 3. OWASP Web chunking degrades faithfulness

The Phase 4 faithfulness anomaly was not random noise. Rows containing OWASP Web chunks had materially worse faithfulness and much denser bullet-style retrieval content:

- `contains_owasp_web`: mean faithfulness `0.1561`, bullet markers/chunk `1.65`
- `nvd_only`: mean faithfulness `0.2783`, bullet markers/chunk `0.03`

That finding is preserved as a first-class section in the final report because it points to retrieval/chunking quality, not just model behavior.

See [OWASP Web Faithfulness Investigation](IMPLEMENTATION_NOTES.md#46-owasp-web-faithfulness-investigation) and the Phase 5 carry-forward findings in the report.

## Quick start

### Prerequisites

- Python `3.11`
- `uv`
- at least one `GROQ_API_KEY`
- enough Groq free-tier headroom for the commands you run

### Install

```bash
uv sync --all-groups
cp .env.example .env
```

Populate `.env` with your Groq credentials. Full guarded scoring used four keys locally; the CI smoke check uses a single key.

### Common commands

Run tests and static checks:

```bash
make lint
make test
```

Build the final report from existing artifacts only:

```bash
uv run python -m src.reporting.cli full \
  --unguarded-risk results/run_20260516_131022/risk_scores.json \
  --guarded-results results/run_20260516_164921/results.jsonl \
  --guarded-decisions results/run_20260516_164921/guardrail_decisions.jsonl \
  --guarded-scores results/run_20260517_115140/scores.jsonl \
  --guarded-risk results/run_20260517_115140/risk_scores.json \
  --residual-analysis results/run_20260517_115451/residual_analysis.json \
  --cross-validation results/cross_validation_20260516_132118/cross_validation.json \
  --output-root results
```

Run a sampled PR-style smoke check locally:

```bash
uv run python -m src.guardrails.cli run-attacks --sample 25 --skip-preflight
uv run python -m src.evaluation.cli full --results <sample-results.jsonl> --skip-preflight
```

## Reproducibility and quotas

This project was intentionally built against free-tier constraints rather than idealized cloud budgets.

- full unguarded + guarded evaluation is not a one-command CI task
- full scoring may require ~24h reset windows across Groq keys
- four-key rotation worked under real load during guarded scoring
- per-attack token attribution is still incomplete; Phase 6 reports only what existing artifacts persist

The practical implication: CI runs only a `25`-attack smoke check. Full `175`-attack evaluations are manual and feed the committed baseline artifact.

## CI and baseline workflow

### CI

- `ci.yml`: lint, typecheck, and tests on push to `dev`, push to `main`, and all PRs
- `pr-risk-comment.yml`: sampled `25`-attack PR smoke check for changes under `src/**`

The PR bot is intentionally framed as a regression detector, not a safety certification step.

### Baseline update process

`results/baseline/risk_report.json` is committed manually after a trusted full run.

Baseline refresh steps:

1. Run the full guarded evaluation manually after merge to `dev`
2. Build a fresh `risk_report.json`
3. Copy that artifact into `results/baseline/risk_report.json`
4. Commit the update with the Phase 6 reporting branch

If no baseline exists, the PR bot posts: `No baseline available for comparison; this is the initial run.`

## Repository map

```text
src/target_system/    RAG chatbot and retrieval pipeline
src/pipeline/         attack execution, caching, retries, rate limiting
src/evaluation/       metric suite, scoring, cross-validation
src/guardrails/       input sanitizer, safety classifier, output filter, residuals
src/reporting/        report builder, Plotly renderer, CLI
attacks/v1/           synthetic benchmark dataset
results/              scored runs, residuals, deltas, reports, baseline
.github/workflows/    CI and PR smoke-check workflows
```

## Status

Phase status at the end of Phase 6 work:

- target RAG system: complete
- attack dataset: complete at `175` accepted attacks
- evaluation engine: complete
- guardrail delta evaluation: complete
- offline reporting: complete
- CI + PR smoke checks: complete

The main remaining follow-up items are `v1.1` work:

- improve `L2` classifier reliability and reduce fail-closed inflation
- strengthen `LLM06` coverage for risky IAM and tool-call style prompts
- persist per-attack token attribution for cost-of-defense reporting
- improve retrieval chunking for OWASP Web content

## Specification and notes

- [PROJECT_SPEC.md](PROJECT_SPEC.md)
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- [Phase 6 report schema contract](src/reporting/schema_v1.md)
