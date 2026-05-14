from __future__ import annotations

import pytest

from src.evaluation.engine import AttackEvaluationInput
from src.evaluation.metrics.hallucination import HallucinationMetric


class FakeDeepEvalScorer:
    def __init__(self) -> None:
        self.score: float | None = 0.6
        self.reason: str | None = "unsupported claim"
        self.verdicts: object = [{"claim": "redacted fixture", "verdict": "hallucinated"}]
        self.measured: list[object] = []

    def measure(self, test_case: object) -> None:
        self.measured.append(test_case)


@pytest.mark.asyncio
async def test_hallucination_scores_contextual_response() -> None:
    scorer = FakeDeepEvalScorer()
    metric = HallucinationMetric(scorer=scorer)
    attack = AttackEvaluationInput(
        attack_id="LLM09-0001",
        owasp_category="LLM09:2025",
        attack_prompt="Summarize",
        target_response="Unsupported response",
        retrieved_chunks=["source context"],
    )

    result = await metric.score(attack)

    assert result.score == 0.6
    assert result.evidence["mode"] == "contextual"
    assert result.evidence["retrieved_context_count"] == 1
    assert len(scorer.measured) == 1


@pytest.mark.asyncio
async def test_hallucination_scores_empty_context_reference_free() -> None:
    scorer = FakeDeepEvalScorer()
    metric = HallucinationMetric(scorer=scorer)
    attack = AttackEvaluationInput(
        attack_id="LLM09-0002",
        owasp_category="LLM09:2025",
        attack_prompt="Invent package advice",
        target_response="The fictitious package fixes everything.",
        retrieved_chunks=[],
    )

    result = await metric.score(attack)

    assert result.score == 0.6
    assert result.skipped is False
    assert result.evidence["mode"] == "reference_free"
    assert result.evidence["retrieved_context_count"] == 0
