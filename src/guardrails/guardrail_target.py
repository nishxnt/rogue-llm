"""Guardrail-wrapped target system for Phase 5."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Protocol

from src.guardrails.input_sanitizer import InputSanitizer
from src.guardrails.output_filter import OutputFilter
from src.guardrails.reasons import GuardrailBlock, GuardrailDecisionRecord, refusal_message_for
from src.guardrails.safety_classifier import SafetyClassifier
from src.target_system.models import Response


class BaseTarget(Protocol):
    async def aquery(self, prompt: str) -> Response:
        """Run one async query against the underlying target."""


class GuardrailTarget:
    """Defense-in-depth wrapper around the unguarded Phase 1 RAG chatbot."""

    def __init__(
        self,
        *,
        base_rag_chatbot: BaseTarget,
        policy_path: Path | str,
        input_sanitizer: InputSanitizer | None = None,
        safety_classifier: SafetyClassifier | None = None,
        output_filter: OutputFilter | None = None,
    ) -> None:
        self.base_rag_chatbot = base_rag_chatbot
        self.policy_path = Path(policy_path)
        self.input_sanitizer = input_sanitizer or InputSanitizer()
        self.safety_classifier = safety_classifier or SafetyClassifier(policy_path=self.policy_path)
        self.output_filter = output_filter or OutputFilter()
        self._settings = getattr(base_rag_chatbot, "_settings", None)
        self._prompt = getattr(base_rag_chatbot, "_prompt", "")
        self._attack_context_id: str | None = None
        self.guardrail_fingerprint = self._build_fingerprint()

    def set_attack_context(self, attack_id: str) -> None:
        """Set the current attack id so decision logs can be attack-addressable."""
        self._attack_context_id = attack_id

    def query(self, prompt: str) -> Response:
        return asyncio.run(self.aquery(prompt))

    async def aquery(self, prompt: str) -> Response:
        started_at = time.perf_counter()

        l1_block = self.input_sanitizer.inspect(prompt)
        if l1_block is not None:
            return self._blocked_response(l1_block, started_at, base_target_called=False)

        l2_block = await self.safety_classifier.inspect(prompt)
        if l2_block is not None:
            return self._blocked_response(l2_block, started_at, base_target_called=False)

        base_response = await self.base_rag_chatbot.aquery(prompt)

        l3_block = self.output_filter.inspect(base_response.answer)
        if l3_block is not None:
            return self._blocked_response(
                l3_block,
                started_at,
                base_target_called=True,
                template=base_response,
            )

        return Response(
            answer=base_response.answer,
            retrieved_chunks=base_response.retrieved_chunks,
            latency_ms=base_response.latency_ms,
            tokens_used=base_response.tokens_used,
            conversation_id=base_response.conversation_id,
            guardrail_decision="allowed",
            guardrail_decision_layer=None,
            guardrail_evidence={},
            base_target_called=True,
            guardrail_timestamp=GuardrailDecisionRecord(
                attack_id=self._attack_context_id,
                decision="allowed",
                decision_layer=None,
                evidence={},
                base_target_called=True,
            ).timestamp,
        )

    async def aclose(self) -> None:
        await self.safety_classifier.aclose()

    def _blocked_response(
        self,
        block: GuardrailBlock,
        started_at: float,
        *,
        base_target_called: bool,
        template: Response | None = None,
    ) -> Response:
        record = GuardrailDecisionRecord(
            attack_id=self._attack_context_id,
            decision=block.decision,
            decision_layer=block.decision_layer,
            evidence=block.evidence,
            base_target_called=base_target_called,
        )
        return Response(
            answer=refusal_message_for(block.decision),
            retrieved_chunks=template.retrieved_chunks if template is not None else [],
            latency_ms=(
                template.latency_ms
                if template is not None
                else (time.perf_counter() - started_at) * 1000
            ),
            tokens_used=template.tokens_used if template is not None else 0,
            conversation_id=(
                template.conversation_id if template is not None else record.attack_id or ""
            ),
            guardrail_decision=record.decision,
            guardrail_decision_layer=record.decision_layer,
            guardrail_evidence=record.evidence,
            base_target_called=record.base_target_called,
            guardrail_timestamp=record.timestamp,
        )

    def _build_fingerprint(self) -> str:
        policy_hash = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        material = "|".join(
            [
                f"policy={policy_hash}",
                f"l1={self.input_sanitizer.fingerprint()}",
                f"l2={self.safety_classifier.fingerprint()}",
                f"l3={self.output_filter.fingerprint()}",
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()
