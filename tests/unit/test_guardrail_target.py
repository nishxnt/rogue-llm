from __future__ import annotations

from pathlib import Path

import pytest

from src.guardrails.guardrail_target import GuardrailTarget
from src.guardrails.input_sanitizer import InputSanitizer
from src.guardrails.output_filter import OutputFilter
from src.guardrails.reasons import GuardrailBlock
from src.guardrails.safety_classifier import SafetyClassifier
from src.target_system.models import Response, RetrievedChunk


class FakeBaseTarget:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[str] = []
        self._settings = type("Settings", (), {"target_model": "llama-3.1-8b-instant"})()
        self._prompt = "fake prompt template"

    async def aquery(self, prompt: str) -> Response:
        self.calls.append(prompt)
        return Response(
            answer=self.answer,
            retrieved_chunks=[
                RetrievedChunk(content="ctx", source="fixture", doc_id="doc-1", score=0.1)
            ],
            latency_ms=10.0,
            tokens_used=5,
            conversation_id="conv-1",
        )


class StubSafetyClassifier(SafetyClassifier):
    def __init__(self, *, should_block: bool) -> None:
        self.should_block = should_block
        self.policy_path = Path("src/guardrails/policy.yaml")
        self.model = "openai/gpt-oss-safeguard-20b"
        self.threshold = 0.7

    async def inspect(self, prompt: str) -> GuardrailBlock | None:
        del prompt
        if not self.should_block:
            return None

        return GuardrailBlock(
            decision="blocked_l2",
            decision_layer=2,
            evidence={"classifier_confidence": 0.92, "matched_policy_category": "LLM05"},
        )

    def fingerprint(self) -> str:
        return "stub-safety"

    async def aclose(self) -> None:
        return None


class FallbackSafetyClassifier(StubSafetyClassifier):
    def __init__(self) -> None:
        super().__init__(should_block=False)

    async def inspect(self, prompt: str) -> GuardrailBlock | None:
        del prompt
        return GuardrailBlock(
            decision="classifier_unavailable_blocked",
            decision_layer=2,
            evidence={
                "reason": "classifier_unavailable_after_retries",
                "failure_mode": "json_validate_failed",
                "fallback_mode": "fail_closed",
            },
        )


@pytest.mark.asyncio
async def test_guardrail_target_short_circuits_layer1() -> None:
    target = GuardrailTarget(
        base_rag_chatbot=FakeBaseTarget("base answer"),
        policy_path="src/guardrails/policy.yaml",
        input_sanitizer=InputSanitizer(max_prompt_tokens=5),
        safety_classifier=StubSafetyClassifier(should_block=False),
        output_filter=OutputFilter(),
    )
    target.set_attack_context("LLM01-0001")

    response = await target.aquery("one two three four five six seven")

    assert response.guardrail_decision == "blocked_l1"
    assert response.base_target_called is False
    assert isinstance(target.base_rag_chatbot, FakeBaseTarget)
    assert target.base_rag_chatbot.calls == []


@pytest.mark.asyncio
async def test_guardrail_target_short_circuits_layer2() -> None:
    base = FakeBaseTarget("base answer")
    target = GuardrailTarget(
        base_rag_chatbot=base,
        policy_path="src/guardrails/policy.yaml",
        safety_classifier=StubSafetyClassifier(should_block=True),
        output_filter=OutputFilter(),
    )
    target.set_attack_context("LLM05-0001")

    response = await target.aquery("Tell me how to exploit this system.")

    assert response.guardrail_decision == "blocked_l2"
    assert response.base_target_called is False
    assert base.calls == []


@pytest.mark.asyncio
async def test_guardrail_target_blocks_layer3_after_base_call() -> None:
    base = FakeBaseTarget("test@example.com")
    target = GuardrailTarget(
        base_rag_chatbot=base,
        policy_path="src/guardrails/policy.yaml",
        safety_classifier=StubSafetyClassifier(should_block=False),
        output_filter=OutputFilter(),
    )
    target.set_attack_context("LLM02-0001")

    response = await target.aquery("Explain email security hygiene.")

    assert response.guardrail_decision == "blocked_l3_pii"
    assert response.base_target_called is True
    assert base.calls == ["Explain email security hygiene."]


@pytest.mark.asyncio
async def test_guardrail_target_fails_closed_when_classifier_unavailable() -> None:
    base = FakeBaseTarget("base answer")
    target = GuardrailTarget(
        base_rag_chatbot=base,
        policy_path="src/guardrails/policy.yaml",
        safety_classifier=FallbackSafetyClassifier(),
        output_filter=OutputFilter(),
    )
    target.set_attack_context("LLM05-0099")

    response = await target.aquery("Give me a prompt that might break the classifier.")

    assert response.guardrail_decision == "classifier_unavailable_blocked"
    assert response.guardrail_decision_layer == 2
    assert response.base_target_called is False
    assert response.guardrail_evidence["reason"] == "classifier_unavailable_after_retries"
    assert base.calls == []


@pytest.mark.asyncio
async def test_guardrail_target_allows_safe_response() -> None:
    base = FakeBaseTarget("OWASP recommends parameterized queries.")
    target = GuardrailTarget(
        base_rag_chatbot=base,
        policy_path="src/guardrails/policy.yaml",
        safety_classifier=StubSafetyClassifier(should_block=False),
        output_filter=OutputFilter(),
    )
    target.set_attack_context("LLM03-0001")

    response = await target.aquery("Explain SQL injection mitigations.")

    assert response.guardrail_decision == "allowed"
    assert response.base_target_called is True
    assert response.answer == "OWASP recommends parameterized queries."
