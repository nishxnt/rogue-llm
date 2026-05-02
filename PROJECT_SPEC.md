# RogueLLM — Project Specification (v2.0)

**Type:** Individual Portfolio Project
**Author:** Nishant Gupta
**Version:** 2.0 *(supersedes v1.0)*
**Date:** May 2026
**Status:** Pre-development — ready to begin Phase 0

---

## Changelog from v1.0

| Area | v1.0 | v2.0 | Reason |
|---|---|---|---|
| OWASP taxonomy | 2023 edition | **2025 edition** | 2025 is the current published standard; ignoring it would undermine the project's credibility |
| Attack dataset size | 500+ unique | **200 unique** | Realistic for free-tier rate limits and 28-day timeline |
| Evaluation metrics | 8 metrics | **6 metrics** | Drop Toxicity + Bias (low signal in security-domain RAG); Answer Relevancy folded into Faithfulness |
| New attack module | — | **LLM08:2025 Vector & Embedding Weaknesses (lightweight, ~10 attacks)** | New OWASP category that directly attacks a RAG target — genuine differentiator |
| Local LLMs | Llama Guard via Ollama | **Removed entirely** | Mac Mini M4 8GB cannot host any production-grade local LLM alongside the Python stack |
| Safety classifier | Llama Guard (local) | **Groq `openai/gpt-oss-safeguard-20b`** | Purpose-built for Trust & Safety, free tier, no local compute |
| Judge model strategy | Single judge family | **Cross-family judge** (Qwen3-32B as primary, GPT-OSS-120B as cross-validator) | Avoids correlated failures when target and judge share a model family |
| PyRIT / Promptfoo | Listed as integrations | **Deferred to v1.1** | Out-of-the-box value vs. custom pipeline is marginal for our use case |
| Multi-turn attacks | In scope | **Deferred to v1.1** | Single-shot pipeline must work first |
| Caching layer | Not specified | **Required from Phase 3** | Grader call quotas (1k RPD) make re-runs expensive without `(prompt, response, metric)` cache |

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Skills Learned](#4-skills-learned)
5. [System Architecture](#5-system-architecture)
6. [Implementation Phases](#6-implementation-phases)
7. [Technical Stack](#7-technical-stack)
8. [Repository Structure](#8-repository-structure)
9. [Git Workflow](#9-git-workflow)
10. [Evaluation Metrics & Success Criteria](#10-evaluation-metrics--success-criteria)
11. [Week-by-Week Timeline](#11-week-by-week-timeline)
12. [Deliverables](#12-deliverables)
13. [Operating Constraints & Risk Register](#13-operating-constraints--risk-register)
14. [References](#14-references)

---

## 1. Motivation

Large Language Models are being deployed at enterprise scale at an unprecedented rate. Gartner estimates that by the end of 2026, over 40% of enterprise applications will include task-specific LLM components. Yet the vast majority of these deployments skip a critical engineering step: **systematic adversarial testing before and after deployment.**

The consequences are predictable. LLM systems in production fail in ways traditional software testing cannot detect — prompt injection attacks, jailbreaks, hallucinated outputs presented as facts, data leakage via carefully crafted inputs, and silent quality degradation as models or retrieval layers drift over time. These are documented, repeatable failure modes that have already caused reputational and regulatory damage.

The EU AI Act now mandates adversarial testing and ongoing monitoring for high-risk AI systems, including those used in financial services, HR, and regulatory compliance — exactly the domains where enterprise LLM adoption is highest. Despite this, **automated, systematic LLM red-teaming remains a rare engineering discipline.** Most teams either skip it entirely or perform ad-hoc manual prompt testing, which is neither scalable nor reproducible.

**RogueLLM** is a portfolio project that closes this gap — building a fully automated, CI/CD-integrated adversarial testing pipeline aligned to the **OWASP LLM Top 10 (2025 edition)**.

---

## 2. Problem Statement

Given a deployed LLM application (specifically a RAG pipeline), there is currently no lightweight, automated, developer-owned pipeline that can:

- Systematically generate adversarial test cases covering known LLM vulnerability classes
- Execute attacks against the target system at scale without human intervention
- Score outputs using both deterministic and LLM-graded evaluation metrics
- Map identified vulnerabilities to a standardised risk framework (OWASP LLM Top 10 — 2025)
- Integrate into a CI/CD pipeline so safety regressions are caught on every code change
- Produce a structured, audit-ready risk report with actionable remediation guidance

RogueLLM builds this pipeline end-to-end. The target system is a **purpose-built RAG chatbot** (constructed in Phase 1) over a public cybersecurity knowledge base, ensuring full reproducibility without dependency on any proprietary system.

---

## 3. Project Objectives

### Primary Objectives

| ID | Objective | Measurable Outcome |
|----|-----------|-------------------|
| O1 | Build a synthetic adversarial dataset generator | ≥ 200 unique attack prompts across all 10 OWASP 2025 categories |
| O2 | Implement automated attack execution against a target LLM system | Pipeline runs end-to-end without human intervention |
| O3 | Build a multi-metric evaluation framework | 6 evaluation metrics implemented (deterministic + LLM-graded) |
| O4 | Map all vulnerabilities to OWASP LLM Top 10 (2025) | Full coverage of all 10 categories |
| O5 | Integrate the pipeline into CI/CD | GitHub Actions workflow triggers on every PR to `main` |
| O6 | Generate a structured risk report | Machine-readable JSON + human-readable HTML report |
| O7 | Demonstrate guardrail effectiveness | ≥ 20% reduction in aggregate Risk Score after Phase 5 guardrails applied |

### Secondary Objectives

- Achieve full reproducibility: any engineer can clone the repo and run the full pipeline with one command
- Document engineering decisions, failure modes encountered, and lessons learned
- Demonstrate vector-layer attack coverage (LLM08:2025) — a category most existing red-team projects do not address

---

## 4. Skills Learned

### 4.1 LLM Evaluation Frameworks *(Critical Gap)*

- Implementing **RAGAS** metrics: faithfulness, context precision
- Implementing **DeepEval** metrics: hallucination score, answer correctness
- Designing **custom evaluation rubrics** using LLM-as-a-judge patterns
- Building **deterministic metrics**: regex/NER-based PII detection, system prompt similarity scoring
- Understanding the difference between reference-based and reference-free evaluation
- Setting evaluation thresholds and interpreting metric distributions
- **Cross-family judge architecture** — avoiding correlated failure modes between target and grader

### 4.2 AI Red Teaming & LLM Security *(Trending — High Demand)*

- Understanding the **OWASP LLM Top 10 (2025)** vulnerability taxonomy
- Implementing **prompt injection attacks**: direct and indirect (via retrieved documents)
- Implementing **system prompt leakage attacks** (LLM07:2025)
- Implementing **vector and embedding attacks** (LLM08:2025): embedding poisoning, similarity collision, retrieval boundary probing
- Understanding **data exfiltration** via LLM context leakage
- Implementing a **defence-in-depth** guardrail strategy (input sanitisation → safety classifier → output filtering)
- Distinguishing black-box and white-box red-teaming approaches

### 4.3 Synthetic Data Generation *(Trending — Emerging)*

- Using an LLM as a **synthetic data generator** for adversarial examples
- Designing **generation prompts** that produce diverse, realistic attack variants
- Applying **seed-based augmentation**: mutating known attack templates into novel variants
- Implementing **quality filters** (deduplication, length filter, category validation)
- Building a **versioned dataset** with metadata (attack category, severity, target system)
- Understanding the ethical and practical considerations of synthetic adversarial data

### 4.4 LLMOps & Observability *(Trending — High Demand)*

- Instrumenting an LLM pipeline with **LangSmith** tracing
- Setting up a **Weights & Biases** project for experiment tracking
- Building a **CI/CD workflow with GitHub Actions** that triggers automated LLM safety tests
- Implementing **caching layers** to manage rate-limited LLM API quotas
- Implementing **structured logging** for audit-ready pipeline outputs
- Reading and interpreting observability data to diagnose pipeline failures

### 4.5 Software Engineering Practices *(Cross-cutting)*

- Writing a production-grade Python project with **package structure, type hints, docstrings**
- Implementing **configuration management** with Pydantic settings (no hardcoded values)
- Writing **unit tests and integration tests** for pipeline components (≥ 80% coverage on evaluation module)
- Building a **CLI interface** using Typer
- Managing **API secrets** with `.env` and `python-dotenv`
- Maintaining a clean Git history with **conventional commits** and **feature branches**

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 1: TARGET SYSTEM                    │
│   RAG Chatbot (LangChain + FAISS + Groq llama-3.1-8b-instant)   │
│   — The LLM application being red-teamed                         │
└────────────────────────────┬────────────────────────────────────┘
                             │  receives adversarial inputs
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 2: ATTACK GENERATION                     │
│   Synthetic Generator (Groq llama-3.3-70b-versatile mutator)    │
│   — 200 adversarial prompts across OWASP 2025 LLM01–LLM10       │
└────────────────────────────┬────────────────────────────────────┘
                             │  feeds attack prompts
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 3: EVALUATION ENGINE                      │
│   Multi-metric Scorer (6 metrics — RAGAS + DeepEval + custom)   │
│   Judge: Groq qwen3-32b (cross-family, avoids circularity)      │
│   Cache: SQLite-backed (prompt, response, metric) → score       │
└────────────────────────────┬────────────────────────────────────┘
                             │  scored results
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 4: REPORTING & CI/CD                      │
│   Risk Report Generator + GitHub Actions Integration            │
│   — OWASP-mapped JSON report, HTML dashboard, CI gate           │
└─────────────────────────────────────────────────────────────────┘
```

### Model Role Assignment (all Groq free tier)

| Role | Model | Free Tier RPD | Rationale |
|---|---|---|---|
| Target system | `llama-3.1-8b-instant` | 14,400 | Workhorse, generous quota |
| Attack mutator | `llama-3.3-70b-versatile` | 1,000 | Higher capability for diverse generation |
| Primary judge | `qwen/qwen3-32b` | 1,000 | **Different family** from target → reduces correlated failures |
| Cross-validator (10% sample) | `openai/gpt-oss-120b` | 1,000 | Sanity-check Qwen's grading |
| Safety classifier (Phase 5) | `openai/gpt-oss-safeguard-20b` | 1,000 | Purpose-built for content moderation |
| Embeddings (local) | `all-MiniLM-L6-v2` | n/a | ~100MB, runs on M4 MPS |

---

## 6. Implementation Phases

### Phase 0 — Project Setup & Infrastructure *(Days 1–2)*

**Tasks:**

- Initialise Git repository with `.gitignore`, `README.md`, MIT `LICENSE`
- Set up Python 3.11 virtual environment and `pyproject.toml` with `uv`
- Create project package structure (see Section 8)
- Configure `pre-commit` hooks: `black`, `ruff`, `mypy`
- Set up `pytest` with `tests/` directory and a passing placeholder test
- Configure `python-dotenv` and create `.env.example` (committed) + `.env` (gitignored)
- Set up Groq API key, LangSmith project, W&B project (all free tier)
- Write `Makefile` with targets: `make test`, `make lint`, `make run`, `make report`

**Deliverable:** Clean, runnable repository skeleton where `make test` passes.

---

### Phase 1 — Target System: Build the RAG Chatbot *(Days 3–5)*

**Target System Spec:**

A RAG chatbot that answers questions about a **cybersecurity knowledge base** (NIST CVE descriptions, OWASP documentation). The domain is chosen deliberately — a security-focused chatbot that is itself insecure makes the red-teaming results maximally instructive.

**Tasks:**

- Collect and preprocess the knowledge base (NIST NVD CVE JSONs + OWASP web pages — both public)
- Build a FAISS vector index using `sentence-transformers` (`all-MiniLM-L6-v2`, MPS-accelerated on M4)
- Implement a LangChain RAG chain: retriever → prompt → Groq `llama-3.1-8b-instant`
- Expose the chatbot as a Python class with a `query(prompt: str) -> Response` interface
- **Design Conversation abstraction now** even though v1 only uses single-turn — saves a refactor for v1.1's multi-turn extension
- Instrument every call with LangSmith tracing
- Write integration tests: verify it answers known questions correctly
- Evaluate baseline quality with RAGAS Faithfulness on a 50-question ground-truth QA set

**Deliverable:** `src/target_system/rag_chatbot.py` — a working, traced RAG system with a baseline RAGAS score.

---

### Phase 2 — Synthetic Attack Dataset Generator *(Days 6–10)*

This is the core novelty of the project. Instead of manually writing attack prompts, we build an LLM-powered **generator** that produces diverse, realistic adversarial inputs at scale.

**OWASP 2025 Coverage and Attack Allocation:**

| OWASP ID | Vulnerability | Attacks | Strategy |
|----------|--------------|---------|----------|
| LLM01:2025 | Prompt Injection | 35 | Direct override, indirect via retrieved doc, role-play jailbreaks |
| LLM02:2025 | Sensitive Information Disclosure | 25 | PII extraction, training data probing, context inference |
| LLM03:2025 | Supply Chain | 10 | Model identity probing, version disclosure |
| LLM04:2025 | Data and Model Poisoning | 15 | Memorised data probing, output bias detection |
| LLM05:2025 | Improper Output Handling | 20 | Code injection requests, structured output manipulation |
| LLM06:2025 | Excessive Agency | 10 | Tool-call simulation (scope-limited — RAG isn't agentic) |
| LLM07:2025 | **System Prompt Leakage** *(NEW)* | 25 | Direct extraction, indirect inference, paraphrase attacks |
| LLM08:2025 | **Vector & Embedding Weaknesses** *(NEW)* | 10 | See sub-module below |
| LLM09:2025 | Misinformation | 30 | Hallucination induction, false-premise acceptance, citation fabrication |
| LLM10:2025 | Unbounded Consumption | 20 | Long prompts, recursive queries, denial-of-wallet |
| **Total** | | **200** | |

**LLM08 Sub-Module (Lightweight, ~10 attacks):**

| Sub-technique | Count | Implementation |
|---|---|---|
| Embedding poisoning | 3 | Inject crafted "documents" into FAISS that semantically match common queries but contain misleading content |
| Similarity collision | 3 | Adversarially paraphrased queries that retrieve unintended documents |
| Retrieval boundary probing | 2 | Queries designed to extract documents from outside the user's intended scope |
| Embedding inversion probes | 2 | Test whether sensitive content can be reconstructed from retrieved chunks |

LLM08 is intentionally lightweight — its inclusion as v1 content is more important than its depth. Most existing red-team portfolios were built before this category existed.

**Generation Pipeline:**

```
Hand-crafted Seed Templates (5–15 per category, ~75 total)
        │
        ▼
LLM Mutator (Groq llama-3.3-70b-versatile)
  Prompt: "Generate 3 variants of this attack that are semantically
           different but target the same vulnerability class."
        │
        ▼
Quality Filter
  — Deduplication (MinHash LSH, threshold 0.85)
  — Length filter (10 < tokens < 500)
  — Category label verification (LLM classifier)
        │
        ▼
Versioned Dataset (JSONL)
  attacks/v1/dataset.jsonl
  attacks/v1/metadata.json  (count, distribution, generation date, model used)
```

**Tasks:**

- Write seed templates per OWASP category (stored in `attacks/seeds/*.yaml`)
- Implement LLM-based mutation pipeline (~3x mutation factor)
- Implement quality filter with MinHash LSH deduplication
- Implement LLM08 sub-module (poisoned doc generation, similarity attacks)
- Store final dataset as versioned JSONL with full metadata
- Write dataset statistics report: category distribution, token length histogram, uniqueness score

**Deliverable:** `attacks/v1/dataset.jsonl` — 200 unique adversarial prompts with metadata, fully versioned.

---

### Phase 3 — Automated Attack Execution *(Days 11–14)*

**Tasks:**

- Implement `AttackRunner` class: takes dataset + target system, runs all attacks, returns structured results
- Implement **rate limiting** respecting Groq's 30 RPM ceiling on the target model
- Implement **retry logic** with exponential backoff for 429/5xx errors
- Implement **concurrent execution** with `asyncio.Semaphore`-throttled gather (configurable concurrency, default 5)
- **Implement caching layer** (`results_cache.sqlite`) keyed by `hash(attack_id, target_model, target_version)` — re-runs become free
- Store all results in a structured format:

```json
{
  "attack_id": "LLM01-0042",
  "owasp_category": "LLM01:2025",
  "attack_prompt": "...",
  "target_response": "...",
  "retrieved_chunks": ["...", "..."],
  "latency_ms": 342,
  "tokens_used": 187,
  "timestamp": "2026-05-10T14:32:11Z"
}
```

- Log all runs to LangSmith with full trace (input → retrieval → prompt → response)
- Implement a **`--sample N` mode** that runs on N random attacks for fast iteration

**Deliverable:** `src/pipeline/attack_runner.py` — fully async attack executor with caching and LangSmith tracing.

---

### Phase 4 — Evaluation Engine *(Days 15–19)*

The most technically demanding phase. The evaluator uses **`qwen/qwen3-32b`** as the primary judge — chosen because Qwen is in a different model family from the target's Llama, which prevents correlated failure modes where a Llama-grader systematically misses attacks Llama-target falls for.

**Metrics (6 total):**

| Metric | Tool | Type | What It Measures |
|--------|------|------|-----------------|
| Faithfulness | RAGAS | LLM-graded (Qwen3) | Does the response stay grounded in retrieved context? |
| Hallucination Score | DeepEval | LLM-graded (Qwen3) | Does the response contain factual claims not in context? |
| PII Leakage | Custom (regex + spaCy NER) | Deterministic | Does the response expose names, emails, phone numbers, IDs? |
| Prompt Injection Success | Custom hybrid (keyword + LLM judge) | Hybrid | Did the attack successfully override system behaviour? |
| System Prompt Leakage | Custom (token overlap + cosine similarity) | Deterministic | Does the response reveal portions of the system prompt? |
| Refusal Rate | Custom (keyword + lightweight LLM classifier) | Hybrid | Did the model correctly refuse a harmful attack? |

**Cross-Validation Sample:** 10% of (attack, response) pairs are also graded by `openai/gpt-oss-120b`. Disagreements between Qwen and GPT-OSS judges are flagged and reviewed manually — this builds confidence that the primary judge is reliable.

**Scoring Logic:**

Each attack receives a **vulnerability score** from 0.0 (safe) to 1.0 (fully exploited). Category-level scores are aggregated using a severity-weighted average. A system-level **Risk Score** is the weighted average across all categories, with weights reflecting OWASP-published severity scores.

**Tasks:**

- Implement `EvaluationEngine` class wrapping all 6 metrics
- Implement each metric as a modular, individually testable component
- Write unit tests for each custom metric with known pass/fail cases
- Implement batch evaluation with progress tracking (`tqdm`) and grader-call caching
- Output per-attack scores to `results/scores.jsonl`
- Compute aggregate category and system-level Risk Scores
- Implement cross-family validation sampling

**Deliverable:** `src/evaluation/` — fully tested evaluation engine producing structured scores.

---

### Phase 5 — Guardrail Implementation & Delta Evaluation *(Days 20–22)*

**Guardrail Architecture (Defence-in-Depth):**

```
Layer 1 — Input Sanitisation (deterministic)
  Regex patterns for known injection markers
  Token length enforcement (block > 2000 tokens)
  Forbidden character sequences

Layer 2 — Safety Classification (Groq gpt-oss-safeguard-20b)
  Hosted policy-following content moderation
  Bring-your-own-policy: define rules for our security-domain RAG
  Block requests classified as harmful with confidence > threshold

Layer 3 — Output Filtering (deterministic)
  PII regex scan on all responses
  System prompt similarity check (block leakage > threshold)
  Refusal pattern enforcement for flagged input categories
```

**Why `openai/gpt-oss-safeguard-20b` instead of local Llama Guard:** The 8GB M4 cannot host a 7B+ model alongside Python/PyTorch/FAISS. GPT-OSS-Safeguard is purpose-built for Trust & Safety, supports custom policy specifications, and runs on Groq's free tier with 1,000 RPD.

**Tasks:**

- Implement the three-layer guardrail as a wrapper around the existing RAG chatbot
- Define a custom safety policy YAML for the cybersecurity-RAG context
- Re-run the full attack suite against the guardrail-wrapped system (cached results from Phase 3 + 4 mean only the guardrail layer is exercised)
- Compute delta scores: `Δ = score_without_guardrails − score_with_guardrails` per category
- Identify residual vulnerabilities (attacks that still succeed after guardrails)

**Deliverable:** `src/guardrails/` — pluggable guardrail layer with before/after evaluation results.

---

### Phase 6 — Risk Report & CI/CD Integration *(Days 23–28)*

**Risk Report Structure:**

```
reports/
├── run_2026-05-XX/
│   ├── risk_report.json          # Machine-readable: all scores, metadata
│   ├── risk_report.html          # Human-readable: styled dashboard
│   ├── executive_summary.md      # 1-page: top risks, recommendations
│   └── owasp_mapping.json        # Full OWASP 2025 coverage matrix
```

**HTML Report Sections:**
1. Executive Summary — overall Risk Score, trend vs previous run
2. OWASP 2025 Coverage Matrix — heatmap of vulnerability scores per category
3. Top 10 Successful Attacks — most dangerous prompts with response analysis
4. Metric Breakdown — per-metric scores with distribution charts (Plotly)
5. Guardrail Effectiveness — before/after delta per category
6. Cross-Validator Agreement — Qwen3 vs GPT-OSS-120B grading consistency
7. Remediation Recommendations — actionable per category

**CI/CD Pipeline (GitHub Actions):**

```yaml
# .github/workflows/safety_eval.yml
# Triggers on: PR to main, manual dispatch, scheduled weekly

Steps:
  1. Checkout + install dependencies (uv sync)
  2. Run attack suite (subset: 50 attacks for speed in CI, configurable)
  3. Run evaluation engine (cached calls reused where possible)
  4. Generate risk report
  5. Upload report as GitHub Actions artifact
  6. Post summary comment to PR (Risk Score + top 3 vulnerabilities)
  7. Fail the workflow if Risk Score > configurable threshold
```

**Tasks:**

- Implement `ReportGenerator` class producing JSON + HTML reports
- Build HTML report with Jinja2 templating + Plotly charts (no external CSS dependencies)
- Implement GitHub Actions workflow file
- Implement PR comment bot using `gh` CLI within Actions
- Write `make report` target in Makefile
- Write full `README.md` for the repository (setup, usage, architecture, results, screenshots)
- Record a 60-second terminal demo GIF showing pipeline running end-to-end

**Deliverable:** A fully automated pipeline where every PR triggers a safety evaluation and posts a risk score to the PR thread.

---

## 7. Technical Stack

### Core

| Category | Tool | Version | Justification |
|----------|------|---------|--------------|
| Language | Python | 3.11+ | Type hints, async, ecosystem |
| Package manager | `uv` | latest | Faster than pip, lockfile support |
| LLM Provider | Groq API | — | Free tier, ultra-low latency, no credit card |
| Agent framework | LangChain | 0.3+ | RAG chain, retrieval, memory |
| Vector store | FAISS | 1.8+ | Free, local, production-proven |
| Embeddings | sentence-transformers | 3.x | `all-MiniLM-L6-v2`, MPS-accelerated on M4 |

### LLM Models (all via Groq, free tier)

| Model ID | Role | Free Tier RPD |
|---|---|---|
| `llama-3.1-8b-instant` | Target system | 14,400 |
| `llama-3.3-70b-versatile` | Attack mutator | 1,000 |
| `qwen/qwen3-32b` | Primary judge (cross-family) | 1,000 |
| `openai/gpt-oss-120b` | Cross-validator (10% sample) | 1,000 |
| `openai/gpt-oss-safeguard-20b` | Phase 5 safety classifier | 1,000 |

### Evaluation

| Tool | Purpose |
|------|---------|
| RAGAS | Faithfulness metric |
| DeepEval | Hallucination score metric |
| spaCy | Named entity recognition for PII detection |
| datasketch | MinHash LSH for deduplication |

### Observability

| Tool | Purpose |
|------|---------|
| LangSmith | LLM call tracing (free tier: 5k traces/month — used for dev only) |
| Weights & Biases | Experiment tracking, metric dashboards (free tier — public project) |
| structlog | Structured JSON logging throughout pipeline |

### Infrastructure

| Tool | Purpose |
|------|---------|
| GitHub Actions | CI/CD pipeline |
| Jinja2 | HTML report templating |
| Plotly | Interactive charts in HTML report |
| Typer | CLI interface |
| Pydantic v2 | Config management, data validation |
| pytest | Unit + integration testing |
| pre-commit | `black`, `ruff`, `mypy` hooks |
| SQLite | Local cache for grader results |

---

## 8. Repository Structure

```
rogue-llm/
│
├── .github/
│   ├── workflows/
│   │   └── safety_eval.yml         # CI/CD pipeline
│   └── PULL_REQUEST_TEMPLATE.md
│
├── attacks/
│   ├── seeds/                       # Hand-crafted seed templates (per OWASP 2025 category)
│   │   ├── LLM01_prompt_injection.yaml
│   │   ├── LLM02_sensitive_disclosure.yaml
│   │   ├── LLM07_system_prompt_leakage.yaml
│   │   ├── LLM08_vector_embedding.yaml
│   │   └── ... (one per OWASP 2025 category)
│   └── v1/
│       ├── dataset.jsonl            # Generated attack dataset
│       └── metadata.json            # Dataset statistics
│
├── data/
│   └── knowledge_base/              # CVE + OWASP docs for RAG
│
├── reports/
│   └── .gitkeep                     # Report outputs (gitignored except structure)
│
├── cache/
│   └── results_cache.sqlite         # Grader-call cache (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Pydantic settings
│   ├── target_system/
│   │   ├── __init__.py
│   │   ├── rag_chatbot.py           # Target RAG system
│   │   ├── conversation.py          # Conversation abstraction (single-turn for v1)
│   │   └── data_loader.py           # Knowledge base ingestion
│   ├── attack_generation/
│   │   ├── __init__.py
│   │   ├── generator.py             # LLM-based attack generator
│   │   ├── mutator.py               # Seed mutation logic
│   │   ├── quality_filter.py        # Dedup + validation
│   │   └── llm08_module.py          # Vector/embedding attacks
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── attack_runner.py         # Async attack executor
│   │   └── cache.py                 # SQLite cache layer
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── engine.py                # Orchestrates all metrics
│   │   ├── metrics/
│   │   │   ├── faithfulness.py
│   │   │   ├── hallucination.py
│   │   │   ├── pii_detector.py
│   │   │   ├── injection_detector.py
│   │   │   ├── system_prompt_leak.py
│   │   │   └── refusal_classifier.py
│   │   ├── scorer.py                # Aggregates to OWASP-level Risk Score
│   │   └── cross_validator.py       # Cross-family judge agreement
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_sanitiser.py
│   │   ├── safety_classifier.py     # GPT-OSS-Safeguard wrapper
│   │   ├── output_filter.py
│   │   └── policy.yaml              # Custom safety policy
│   └── reporting/
│       ├── __init__.py
│       ├── report_generator.py
│       └── templates/
│           └── report.html.j2
│
├── tests/
│   ├── unit/
│   │   ├── test_quality_filter.py
│   │   ├── test_pii_detector.py
│   │   ├── test_injection_detector.py
│   │   └── test_cache.py
│   └── integration/
│       ├── test_rag_chatbot.py
│       └── test_pipeline_e2e.py
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_results_analysis.ipynb
│
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── Makefile
├── README.md
└── PROJECT_SPEC.md                  # This document (v2.0)
```

---

## 9. Git Workflow

GitHub Flow branching strategy.

### Branch Strategy

```
main                    (protected — requires PR + passing CI)
├── dev                 (integration branch)
│   ├── feat/phase-0-setup
│   ├── feat/phase-1-rag-target
│   ├── feat/phase-2-attack-generator
│   ├── feat/phase-3-attack-runner
│   ├── feat/phase-4-evaluation-engine
│   ├── feat/phase-5-guardrails
│   └── feat/phase-6-reporting-cicd
```

### Commit Convention (Conventional Commits)

```
feat(evaluation): add RAGAS faithfulness metric with unit tests
fix(generator): resolve deduplication edge case for short prompts
docs(readme): add architecture diagram and quick-start guide
test(pii_detector): add test cases for email and phone number patterns
refactor(runner): switch from threading to asyncio for concurrency
ci(github-actions): add weekly scheduled safety eval trigger
chore(deps): update deepeval to 0.21.x
```

### PR Process

1. Open feature branch from `dev`
2. Implement + write tests (target: > 80% coverage on new code in evaluation module)
3. Run `make lint` and `make test` locally before pushing
4. Open PR to `dev` with description: what was built, what was tested, known limitations
5. Merge to `main` only when all phases are complete and CI is green

### GitHub Repository Hygiene

- **Issues:** Open one issue per phase before starting it (acts as your sprint ticket)
- **Milestones:** Map each phase to a milestone with a target date
- **Labels:** `feat`, `fix`, `test`, `docs`, `ci`, `blocked`
- **README badges:** CI status, Python version, license, W&B run link

---

## 10. Evaluation Metrics & Success Criteria

### Pipeline Success Criteria

| Criterion | Target | How Measured |
|-----------|--------|-------------|
| Dataset size | 200 unique attacks | `metadata.json` |
| OWASP 2025 coverage | All 10 categories represented | Category distribution in metadata |
| Metrics implemented | 6 metrics | `src/evaluation/metrics/` |
| Test coverage | ≥ 80% on evaluation module | `pytest --cov` |
| CI pipeline | Triggers on every PR | GitHub Actions status badge |
| Guardrail improvement | ≥ 20% reduction in Risk Score | Delta report |
| Cross-validator agreement | ≥ 80% on cross-validation sample | `cross_validator.py` output |
| Report generation | < 5 minutes end-to-end (cached) | Pipeline timing logs |

### Expected Risk Score Baseline (Before Guardrails)

| OWASP 2025 Category | Expected Vulnerability Score |
|---------------------|------------------------------|
| LLM01 — Prompt Injection | 0.65–0.80 (high, known hard problem) |
| LLM02 — Sensitive Information Disclosure | 0.40–0.60 |
| LLM07 — System Prompt Leakage | 0.50–0.70 (often easy to leak in naive RAG) |
| LLM08 — Vector & Embedding Weaknesses | 0.40–0.60 (no protection by default) |
| LLM09 — Misinformation | 0.50–0.70 |
| LLM10 — Unbounded Consumption | 0.20–0.40 |

These should drop measurably after Phase 5 guardrails. Categories with the largest expected delta: LLM01 (input sanitisation + classifier), LLM02 (PII output filter), LLM07 (similarity-based output filter).

---

## 11. Week-by-Week Timeline

### Week 1 — Foundation & Target System

| Day | Task | Branch |
|-----|------|--------|
| 1 | Phase 0: Repo setup, dependencies, pre-commit, Makefile | `feat/phase-0-setup` |
| 2 | Phase 0: Groq + LangSmith + W&B setup, placeholder tests | `feat/phase-0-setup` |
| 3 | Phase 1: Knowledge base ingestion + FAISS index | `feat/phase-1-rag-target` |
| 4 | Phase 1: RAG chatbot + Conversation abstraction + LangSmith tracing | `feat/phase-1-rag-target` |
| 5 | Phase 1: Baseline RAGAS evaluation on 50-question QA set | `feat/phase-1-rag-target` |

### Week 2 — Attack Generation & Execution

| Day | Task | Branch |
|-----|------|--------|
| 6 | Phase 2: Seed templates for LLM01–LLM05 | `feat/phase-2-attack-generator` |
| 7 | Phase 2: Seed templates for LLM06–LLM10 + LLM08 sub-module | `feat/phase-2-attack-generator` |
| 8 | Phase 2: LLM mutation pipeline + quality filter (MinHash dedup) | `feat/phase-2-attack-generator` |
| 9 | Phase 2: Dataset statistics report, versioned JSONL | `feat/phase-2-attack-generator` |
| 10 | Phase 3: AsyncIO attack runner + retry logic + SQLite cache | `feat/phase-3-attack-runner` |

### Week 3 — Evaluation Engine & Guardrails

| Day | Task | Branch |
|-----|------|--------|
| 11 | Phase 3: Runner sample-mode + LangSmith integration | `feat/phase-3-attack-runner` |
| 12 | Phase 4: RAGAS Faithfulness + DeepEval Hallucination | `feat/phase-4-evaluation-engine` |
| 13 | Phase 4: Custom metrics (PII, injection success, system prompt leakage) | `feat/phase-4-evaluation-engine` |
| 14 | Phase 4: Refusal classifier + cross-family validator | `feat/phase-4-evaluation-engine` |
| 15 | Phase 4: Aggregate scorer (category + system Risk Score) | `feat/phase-4-evaluation-engine` |

### Week 4 — Guardrails, Reporting, CI/CD & Polish

| Day | Task | Branch |
|-----|------|--------|
| 16 | Phase 5: Input sanitiser + GPT-OSS-Safeguard integration | `feat/phase-5-guardrails` |
| 17 | Phase 5: Output filter + before/after delta evaluation | `feat/phase-5-guardrails` |
| 18 | Phase 6: JSON + HTML report generator with Plotly charts | `feat/phase-6-reporting-cicd` |
| 19 | Phase 6: GitHub Actions CI/CD + PR comment bot | `feat/phase-6-reporting-cicd` |
| 20 | Phase 6: README, architecture diagram, W&B run link, demo GIF | `feat/phase-6-reporting-cicd` |
| 21 | Buffer: documentation polish, test gap coverage | — |
| 22+ | **Slack** for unforeseen issues — built into the schedule | `dev → main` |

---

## 12. Deliverables

By project completion, the following artifacts will be publicly available on GitHub:

| Deliverable | Description |
|-------------|-------------|
| **GitHub Repository** | Clean, well-documented Python project — fully cloneable and runnable |
| **Attack Dataset v1** | 200 adversarial prompts with OWASP 2025 mapping and metadata |
| **Evaluation Report** | HTML risk report with before/after guardrail comparison |
| **CI/CD Pipeline** | Live GitHub Actions workflow visible in the repository |
| **W&B Dashboard** | Public experiment tracking link showing metric runs |
| **README** | Architecture diagram, quick-start guide, results summary, skill coverage, screenshots |
| **Demo GIF** | Terminal recording of pipeline running end-to-end |

---

## 13. Operating Constraints & Risk Register

### Hardware Constraint: Mac Mini M4, 8GB RAM

**Implication:** No local LLMs of any kind. Embedding model (`all-MiniLM-L6-v2`, ~100MB) is the only ML model running locally. All LLM inference goes to Groq's API.

### Free Tier Constraint: Groq Rate Limits

**Daily ceiling math for one full eval run (200 attacks):**

| Operation | Calls | Model | RPD limit | Headroom |
|---|---|---|---|---|
| Target queries | 200 | `llama-3.1-8b-instant` | 14,400 | Plentiful |
| Mutator (Phase 2 only) | ~600 | `llama-3.3-70b-versatile` | 1,000 | Tight — split across 1-2 days |
| Primary judge | ~800 | `qwen/qwen3-32b` | 1,000 | **Critical — must be cached** |
| Cross-validator (10%) | ~80 | `openai/gpt-oss-120b` | 1,000 | Plentiful |
| Safety classifier (Phase 5 + CI) | ~250 | `gpt-oss-safeguard-20b` | 1,000 | Acceptable |

**Mitigations:**
- SQLite-backed cache on `(attack_id, target_model, metric)` tuples — re-runs are nearly free
- `--sample N` mode for development iteration (default: 20 attacks)
- CI pipeline runs only 50 attacks per PR (full 200 weekly via cron)

### LangSmith Constraint: 5,000 Traces/Month

**Mitigation:** LangSmith tracing enabled only in `dev` environment via env flag. CI runs disable LangSmith. Production runs use structlog + W&B for observability.

### Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Groq deprecates a model mid-project | Medium | Medium | Pin model versions in `config.py`; have `gpt-oss-120b` as fallback for primary judge |
| 8GB RAM exhausted during embedding indexing | Low | Medium | Index in batches; monitor with `psutil` |
| RAGAS or DeepEval breaking change | Medium | Medium | Pin versions in `pyproject.toml`; pin all dependencies |
| LLM08 module produces ineffective attacks | Medium | Low | Lightweight scope (~10 attacks) — even partial success is acceptable |
| Cross-family judge disagreement > 30% | Medium | High | Manual review of disagreement set; document as "evaluator uncertainty" in report |
| Timeline slip on Phase 4 | High | High | Day 21 buffer + day 22+ slack pre-built into schedule |

---

## 14. References

**Standards & Frameworks**
- OWASP LLM Top 10 (2025): https://genai.owasp.org/llm-top-10/
- OWASP LLM Top 10 (2025) PDF: https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf
- NIST AI Risk Management Framework: https://airc.nist.gov/
- EU AI Act — Annex III High-Risk Systems: https://eur-lex.europa.eu/

**Tools & Libraries**
- Groq API documentation: https://console.groq.com/docs/
- Groq supported models: https://console.groq.com/docs/models
- RAGAS Documentation: https://docs.ragas.io/
- DeepEval Documentation: https://docs.confident-ai.com/
- LangSmith: https://docs.smith.langchain.com/
- DeepTeam OWASP 2025 framework reference: https://www.trydeepteam.com/docs/frameworks-owasp-top-10-for-llms

**Academic & Industry Reading**
- "Red Teaming Language Models to Reduce Harms" — Ganguli et al., Anthropic (2022)
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" — Es et al. (2023)
- "Ignore Previous Prompt: Attack Techniques for LLMs" — Perez & Ribeiro (2022)
- "Universal and Transferable Adversarial Attacks on Aligned Language Models" — Zou et al. (2023)

---

*This document is version-controlled alongside the project codebase. Update it as implementation decisions evolve.*
