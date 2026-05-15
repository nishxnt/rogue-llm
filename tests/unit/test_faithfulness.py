from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.evaluation.engine import AttackEvaluationInput
from src.evaluation.metrics import faithfulness
from src.evaluation.metrics.faithfulness import (
    FaithfulnessMetric,
    FallbackFaithfulnessJudgment,
)


class FakeRagasScorer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def ascore(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            }
        )
        return SimpleNamespace(value=0.8, reason="grounded enough")


class FailingRagasScorer:
    async def ascore(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> SimpleNamespace:
        raise ValueError("json_validate_failed")


@pytest.mark.asyncio
async def test_faithfulness_scores_with_ragas_inputs() -> None:
    scorer = FakeRagasScorer()
    metric = FaithfulnessMetric(scorer=scorer)
    attack = AttackEvaluationInput(
        attack_id="LLM09-0001",
        owasp_category="LLM09:2025",
        attack_prompt="Explain the finding",
        target_response="The finding is supported.",
        retrieved_chunks=["supporting context"],
    )

    result = await metric.score(attack)

    assert result.score == 0.8
    assert result.skipped is False
    assert result.reason == "grounded enough"
    assert scorer.calls == [
        {
            "user_input": "Explain the finding",
            "response": "The finding is supported.",
            "retrieved_contexts": ["supporting context"],
        }
    ]


@pytest.mark.asyncio
async def test_faithfulness_skips_without_retrieval_context() -> None:
    metric = FaithfulnessMetric(scorer=FakeRagasScorer())
    attack = AttackEvaluationInput(
        attack_id="LLM10-0001",
        owasp_category="LLM10:2025",
        attack_prompt="Generate a long answer",
        target_response="Long response",
        retrieved_chunks=[],
    )

    result = await metric.score(attack)

    assert result.score is None
    assert result.skipped is True
    assert result.reason == "no_retrieval_context"


@pytest.mark.asyncio
async def test_faithfulness_skips_empty_target_response() -> None:
    metric = FaithfulnessMetric(scorer=FakeRagasScorer())
    attack = AttackEvaluationInput(
        attack_id="LLM01-0014",
        owasp_category="LLM01:2025",
        attack_prompt="prompt",
        target_response="",
        retrieved_chunks=["context"],
        status="infrastructure_failure",
    )

    result = await metric.score(attack)

    assert result.score is None
    assert result.skipped is True
    assert result.reason == "empty_target_response"


@pytest.mark.asyncio
async def test_faithfulness_uses_direct_judge_fallback_on_ragas_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fallback(
        *,
        attack: AttackEvaluationInput,
        judge_model: str,
    ) -> FallbackFaithfulnessJudgment:
        return FallbackFaithfulnessJudgment(score=0.25, reason="mostly unsupported")

    monkeypatch.setattr(faithfulness, "score_faithfulness_with_direct_judge", fake_fallback)
    metric = FaithfulnessMetric(scorer=FailingRagasScorer())
    attack = AttackEvaluationInput(
        attack_id="LLM09-0001",
        owasp_category="LLM09:2025",
        attack_prompt="Explain the finding",
        target_response="The finding is unsupported.",
        retrieved_chunks=["supporting context"],
    )

    result = await metric.score(attack)

    assert result.score == 0.25
    assert result.reason == "mostly unsupported"
    assert result.evidence["scoring_mode"] == "direct_groq_fallback"
    assert result.evidence["ragas_error_type"] == "ValueError"
