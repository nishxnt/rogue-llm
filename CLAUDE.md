# RogueLLM — Claude Code Context

## Project
See @PROJECT_SPEC.md for the full v2.0 specification.
Currently in: Phase 0 — Project Setup.

## Stack
- Python 3.11, uv for packaging, Typer for CLI
- LangChain 0.3+, FAISS, sentence-transformers
- Groq API only (no local LLMs — 8GB M4 constraint)
- pytest, black, ruff, mypy via pre-commit

## Hard Rules
- Type hints on all public functions
- Pydantic v2 for all config and structured data
- No hardcoded secrets — read from .env via pydantic-settings
- No print() — use structlog
- Conventional commits: feat|fix|docs|test|refactor|ci|chore(scope): message
- New code in src/evaluation/ requires unit tests (target ≥80% coverage)
- Never commit: .env, cache/, reports/run_*, data/knowledge_base/

## Workflow
- One feature branch per phase: feat/phase-N-<name>
- Run `make lint && make test` before any commit
- Open issues for each phase before starting (acts as sprint ticket)

## Model IDs (pinned — do not "upgrade" without asking)
- Target: llama-3.1-8b-instant
- Mutator: llama-3.3-70b-versatile
- Judge: qwen/qwen3-32b
- Cross-validator: openai/gpt-oss-120b
- Safety: openai/gpt-oss-safeguard-20b

## Future tasks
- Upgrade target: LangChain 1.x + `langchain-classic` once the v1 stack stabilizes (currently held on 0.3 line for tutorial compatibility; this also caps `groq` at 0.x via `langchain-groq` 0.3.x).

## Phase 1 outcomes (recorded for future phases)

- Baseline faithfulness (cross-family, gpt-oss-120b judge): 0.6333 (n=40/40)
- Same-family bias quantified: +5.9pp inflation when judge and target share
  family (0.6923 same-family vs 0.6333 cross-family)
- Checkpointing pattern in src/evaluation/baseline.py is project-wide
  infrastructure to promote to src/pipeline/cache.py in Phase 3
- OWASP Web anomaly (faithfulness 0.4314 vs 0.66/0.74) flagged for Phase 4
  investigation — see IMPLEMENTATION_NOTES.md

## Phase 4 reminders (do not lose)

- Primary judge: openai/gpt-oss-120b (Mode.JSON, verified working)
- Cross-validator: keep llama-3.1-8b unless Groq adds qwen structured-output
  support
- Investigate OWASP Web faithfulness anomaly — likely retrieval/chunking
  issue with bullet-list-style markdown
