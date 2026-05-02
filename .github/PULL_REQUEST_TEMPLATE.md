## What

<!-- 1–2 sentences: what this PR changes -->

## Why

<!-- Link to phase issue, describe motivation -->

## Phase

- [ ] Phase 0 — Setup
- [ ] Phase 1 — RAG Target
- [ ] Phase 2 — Attack Generator
- [ ] Phase 3 — Attack Runner
- [ ] Phase 4 — Evaluation Engine
- [ ] Phase 5 — Guardrails
- [ ] Phase 6 — Reporting & CI

## OWASP 2025 Coverage (if applicable)

<!-- LLM01 / LLM02 / ... categories addressed -->

## Checklist

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] New code in `src/evaluation/` has unit tests (≥80% coverage)
- [ ] No secrets committed (`.env` gitignored, used `.env.example`)
- [ ] Conventional commit messages (`feat|fix|docs|test|refactor|ci|chore(scope): ...`)
- [ ] PROJECT_SPEC.md updated if scope changed

## Test Plan

<!-- How was this tested locally? -->
