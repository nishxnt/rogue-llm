from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/sample_phase_artifacts"

UNGUARDED_RISK = FIXTURE_ROOT / "unguarded_risk_scores.json"
GUARDED_RESULTS = FIXTURE_ROOT / "guarded_results.jsonl"
GUARDED_DECISIONS = FIXTURE_ROOT / "guardrail_decisions.jsonl"
GUARDED_SCORES = FIXTURE_ROOT / "guarded_scores.jsonl"
GUARDED_RISK = FIXTURE_ROOT / "guarded_risk_scores.json"
RESIDUAL_ANALYSIS = FIXTURE_ROOT / "residual_analysis.json"
CROSS_VALIDATION = FIXTURE_ROOT / "cross_validation.json"
