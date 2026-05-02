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