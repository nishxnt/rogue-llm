"""Metric suite construction for Phase 4 evaluation runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.evaluation.config import PRIMARY_JUDGE_MODEL
from src.evaluation.metrics.faithfulness import FaithfulnessMetric
from src.evaluation.metrics.hallucination import HallucinationMetric
from src.evaluation.metrics.injection_detector import GroqInjectionJudge, InjectionDetectorMetric
from src.evaluation.metrics.pii_detector import PIILeakageMetric
from src.evaluation.metrics.refusal_classifier import GroqRefusalJudge, RefusalClassifierMetric
from src.evaluation.metrics.response_patterns import (
    OutputAmplificationMetric,
    agency_compliance_metric,
    memorized_content_disclosure_metric,
    poisoning_indicator_metric,
    resource_compliance_metric,
    unsafe_confidence_metric,
    unsafe_output_metric,
    unsafe_recommendation_metric,
)
from src.evaluation.metrics.system_prompt_leak import SystemPromptLeakMetric

if TYPE_CHECKING:
    from src.evaluation.engine import EvaluationMetric

LLM_GRADED_METRIC_NAMES = ("faithfulness", "hallucination", "injection_success", "refusal")


def build_metric_suite(
    *,
    judge_model: str = PRIMARY_JUDGE_MODEL,
    live_llm_judges: bool = True,
    enable_spacy: bool = False,
    include_response_patterns: bool = True,
) -> list[EvaluationMetric]:
    """Build the Phase 4 metric suite.

    The original six evaluation metrics are always included. ``live_llm_judges``
    controls whether hybrid injection/refusal metrics call the configured judge
    for ambiguous responses; deterministic refusal/marker paths still run either
    way. Response-pattern detectors are supplemental deterministic metrics used
    by the category-specific scorer formulas.
    """
    injection_judge = GroqInjectionJudge(model=judge_model) if live_llm_judges else None
    refusal_judge = GroqRefusalJudge(model=judge_model) if live_llm_judges else None
    metrics: list[EvaluationMetric] = [
        FaithfulnessMetric(judge_model=judge_model),
        HallucinationMetric(judge_model=judge_model),
        PIILeakageMetric(enable_spacy=enable_spacy),
        InjectionDetectorMetric(judge=injection_judge, judge_model=judge_model),
        SystemPromptLeakMetric(),
        RefusalClassifierMetric(judge=refusal_judge, judge_model=judge_model),
    ]
    if include_response_patterns:
        metrics.extend(
            [
                memorized_content_disclosure_metric(),
                unsafe_recommendation_metric(),
                poisoning_indicator_metric(),
                unsafe_output_metric(),
                agency_compliance_metric(),
                unsafe_confidence_metric(),
                OutputAmplificationMetric(),
                resource_compliance_metric(),
            ]
        )
    return metrics
