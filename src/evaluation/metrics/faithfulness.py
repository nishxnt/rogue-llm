"""RAGAS faithfulness metric for Phase 4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from src.config import get_settings
from src.evaluation.config import PRIMARY_JUDGE_MODEL
from src.evaluation.engine import AttackEvaluationInput, MetricResult

if TYPE_CHECKING:
    from collections.abc import Callable


class RagasFaithfulnessScorer(Protocol):
    """Subset of the RAGAS collections Faithfulness API used here."""

    async def ascore(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> Any:
        """Score one response against retrieved context."""


class FaithfulnessMetric:
    """Score whether the target response stays grounded in retrieved context.

    RAGAS faithfulness requires retrieval context. Attacks with no retrieved
    chunks are skipped with ``reason="no_retrieval_context"``; hallucination
    scoring handles reference-free response judging separately.
    """

    name = "faithfulness"
    judge_version = "ragas-collections-v1"

    def __init__(
        self,
        *,
        scorer: RagasFaithfulnessScorer | None = None,
        scorer_factory: Callable[[], RagasFaithfulnessScorer] | None = None,
        judge_model: str = PRIMARY_JUDGE_MODEL,
    ) -> None:
        self._scorer = scorer
        self._scorer_factory = scorer_factory
        self.judge_model = judge_model

    async def score(self, attack: AttackEvaluationInput) -> MetricResult:
        """Score one attack result with RAGAS Faithfulness."""
        if not attack.retrieved_chunks:
            return MetricResult(
                attack_id=attack.attack_id,
                metric_name=self.name,
                score=None,
                skipped=True,
                reason="no_retrieval_context",
                judge_model=self.judge_model,
                judge_version=self.judge_version,
            )

        result = await self._get_scorer().ascore(
            user_input=attack.attack_prompt,
            response=attack.target_response,
            retrieved_contexts=attack.retrieved_chunks,
        )
        return MetricResult(
            attack_id=attack.attack_id,
            metric_name=self.name,
            score=float(result.value),
            reason=str(getattr(result, "reason", "") or "") or None,
            evidence={"retrieved_context_count": len(attack.retrieved_chunks)},
            judge_model=self.judge_model,
            judge_version=self.judge_version,
        )

    def _get_scorer(self) -> RagasFaithfulnessScorer:
        if self._scorer is None:
            if self._scorer_factory is not None:
                self._scorer = self._scorer_factory()
            else:
                self._scorer = build_ragas_faithfulness_scorer(self.judge_model)
        return self._scorer


def build_ragas_faithfulness_scorer(
    judge_model: str = PRIMARY_JUDGE_MODEL,
) -> RagasFaithfulnessScorer:
    """Build the RAGAS Faithfulness scorer using Groq + Instructor.

    This follows the Phase 1 baseline pattern and the RAGAS collections API:
    ``Faithfulness(...).ascore(user_input=..., response=..., retrieved_contexts=...)``.
    """
    import instructor
    from groq import AsyncGroq as AsyncGroqSDK
    from ragas.llms.base import InstructorLLM, InstructorModelArgs
    from ragas.metrics.collections import Faithfulness

    settings = get_settings()
    groq_client = instructor.from_groq(
        AsyncGroqSDK(api_key=settings.groq_api_key.get_secret_value()),
        mode=instructor.Mode.JSON,
    )
    judge_llm = InstructorLLM(
        client=groq_client,
        model=judge_model,
        provider="groq",
        model_args=InstructorModelArgs(max_tokens=4096),
    )
    return cast("RagasFaithfulnessScorer", Faithfulness(llm=judge_llm))
