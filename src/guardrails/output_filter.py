"""Layer 3 output filter reusing Phase 4 detection logic."""

from __future__ import annotations

from src.evaluation.metrics.pii_detector import PIILeakageMetric
from src.evaluation.metrics.response_patterns import unsafe_output_pattern
from src.evaluation.metrics.system_prompt_leak import (
    _FULL_LEAK_THRESHOLD,
    _cosine_similarity,
    _overlap_ratio,
    _tokens,
)
from src.guardrails.reasons import GuardrailBlock
from src.target_system.prompts import SYSTEM_PROMPT


class OutputFilter:
    """Deterministic Layer 3 output filter."""

    def __init__(
        self,
        *,
        pii_detector: PIILeakageMetric | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        system_prompt_overlap_threshold: float = _FULL_LEAK_THRESHOLD,
        unsafe_output_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= system_prompt_overlap_threshold <= 1.0:
            raise ValueError("system_prompt_overlap_threshold must be in [0, 1]")
        if not 0.0 <= unsafe_output_threshold <= 1.0:
            raise ValueError("unsafe_output_threshold must be in [0, 1]")
        self.pii_detector = pii_detector or PIILeakageMetric(enable_spacy=False)
        self.system_prompt = system_prompt
        self.system_prompt_overlap_threshold = system_prompt_overlap_threshold
        self.unsafe_output_threshold = unsafe_output_threshold
        self._system_tokens = _tokens(system_prompt)

    def inspect(self, response_text: str) -> GuardrailBlock | None:
        """Return a Layer 3 block if the model output is unsafe to release."""
        pii_findings = self.pii_detector.detect(response_text)
        if pii_findings:
            return GuardrailBlock(
                decision="blocked_l3_pii",
                decision_layer=3,
                evidence={
                    "detector_name": "pii_leakage",
                    "detection_evidence": {
                        "finding_count": len(pii_findings),
                        "findings": [finding.model_dump() for finding in pii_findings[:10]],
                    },
                },
            )

        response_tokens = _tokens(response_text)
        overlap_ratio = _overlap_ratio(self._system_tokens, response_tokens)
        cosine_similarity = _cosine_similarity(self._system_tokens, response_tokens)
        if overlap_ratio >= self.system_prompt_overlap_threshold or cosine_similarity >= 0.75:
            return GuardrailBlock(
                decision="blocked_l3_leak",
                decision_layer=3,
                evidence={
                    "detector_name": "system_prompt_leak",
                    "detection_evidence": {
                        "overlap_ratio": overlap_ratio,
                        "cosine_similarity": cosine_similarity,
                        "overlap_threshold": self.system_prompt_overlap_threshold,
                    },
                },
            )

        unsafe_score = unsafe_output_pattern(response_text)
        if unsafe_score >= self.unsafe_output_threshold:
            return GuardrailBlock(
                decision="blocked_l3_unsafe_pattern",
                decision_layer=3,
                evidence={
                    "detector_name": "unsafe_output_pattern",
                    "detection_evidence": {
                        "score": unsafe_score,
                        "threshold": self.unsafe_output_threshold,
                    },
                },
            )

        return None

    def fingerprint(self) -> str:
        """Return a stable filter fingerprint for cache invalidation."""
        return (
            f"pii={self.pii_detector.judge_version}|"
            f"leak={self.system_prompt_overlap_threshold}|"
            f"unsafe={self.unsafe_output_threshold}"
        )
