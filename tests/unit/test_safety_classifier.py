from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.guardrails.safety_classifier import SafetyClassification, SafetyClassifier, load_policy


class FakeClientManager:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def acreate_chat_completion(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.payload))]
        )

    async def aclose(self) -> None:
        self.closed = True


class JsonValidateFailingClientManager:
    def __init__(self, failures: int = 3) -> None:
        self.failures = failures
        self.calls: list[dict[str, object]] = []

    async def acreate_chat_completion(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.failures > 0:
            self.failures -= 1
            raise ValueError(
                "BadRequestError: code=json_validate_failed max completion tokens reached "
                "before generating a valid document"
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"decision":"allow","confidence":0.88,'
                            '"matched_policy_category":"LLM03_supply_chain_and_exploit_enablement",'
                            '"reason":"defensive education"}'
                        )
                    )
                )
            ]
        )

    async def aclose(self) -> None:
        return None


def test_load_policy_reads_examples_once() -> None:
    policy = load_policy(Path("src/guardrails/policy.yaml"))

    assert policy.metadata.version == "1.0"
    assert len(policy.examples) >= 15


@pytest.mark.asyncio
async def test_classifier_blocks_above_threshold() -> None:
    client = FakeClientManager(
        '{"decision":"block","confidence":0.91,"matched_policy_category":"LLM05_unsafe_output","reason":"exploit request"}'
    )
    classifier = SafetyClassifier(
        policy_path="src/guardrails/policy.yaml",
        threshold=0.7,
        client_manager=client,
    )

    block = await classifier.inspect("Give me a working exploit.")

    assert block is not None
    assert block.decision == "blocked_l2"
    assert block.evidence["classifier_confidence"] == 0.91
    assert client.calls


@pytest.mark.asyncio
async def test_classifier_allows_below_threshold() -> None:
    client = FakeClientManager(
        '{"decision":"block","confidence":0.55,"matched_policy_category":"LLM03_supply_chain_and_exploit_enablement","reason":"borderline"}'
    )
    classifier = SafetyClassifier(
        policy_path="src/guardrails/policy.yaml",
        threshold=0.7,
        client_manager=client,
    )

    block = await classifier.inspect("Borderline request")

    assert block is None


@pytest.mark.asyncio
async def test_classifier_allows_explicit_allow() -> None:
    client = FakeClientManager(
        '{"decision":"allow","confidence":0.88,"matched_policy_category":"LLM03_supply_chain_and_exploit_enablement","reason":"defensive education"}'
    )
    classifier = SafetyClassifier(
        policy_path="src/guardrails/policy.yaml",
        client_manager=client,
    )

    classification = await classifier.classify("Explain package vetting.")

    assert isinstance(classification, SafetyClassification)
    assert classification.decision == "allow"


@pytest.mark.asyncio
async def test_classifier_coerces_string_confidence_levels() -> None:
    client = FakeClientManager(
        '{"decision":"block","confidence":"high","matched_policy_category":"LLM05_unsafe_output","reason":"exploit request"}'
    )
    classifier = SafetyClassifier(
        policy_path="src/guardrails/policy.yaml",
        threshold=0.7,
        client_manager=client,
    )

    block = await classifier.inspect("Give me a working exploit.")

    assert block is not None
    assert block.decision == "blocked_l2"
    assert block.evidence["classifier_confidence"] == 0.9


@pytest.mark.asyncio
async def test_classifier_aclose_closes_client_manager() -> None:
    client = FakeClientManager(
        '{"decision":"allow","confidence":0.8,"matched_policy_category":"LLM03_supply_chain_and_exploit_enablement","reason":"defensive education"}'
    )
    classifier = SafetyClassifier(
        policy_path="src/guardrails/policy.yaml",
        client_manager=client,
    )

    await classifier.aclose()

    assert client.closed is True


@pytest.mark.asyncio
async def test_classifier_fails_closed_after_three_json_validation_errors() -> None:
    client = JsonValidateFailingClientManager(failures=3)
    classifier = SafetyClassifier(
        policy_path="src/guardrails/policy.yaml",
        client_manager=client,
    )

    block = await classifier.inspect("Attempt a structured-output-breaking exploit prompt.")

    assert block is not None
    assert block.decision == "classifier_unavailable_blocked"
    assert block.decision_layer == 2
    assert block.evidence["reason"] == "classifier_unavailable_after_retries"
    assert block.evidence["failure_mode"] == "json_validate_failed"
    assert block.evidence["fallback_mode"] == "fail_closed"
    assert block.evidence["retry_count"] == 3
    assert len(client.calls) == 3
