from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from src.attack_generation import quality_filter


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_: Any) -> SimpleNamespace:
        self.calls += 1
        message = SimpleNamespace(content="{not valid json")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeGroq:
    last_completions: _FakeCompletions | None = None

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=completions)
        _FakeGroq.last_completions = completions


class _FakeSecret:
    def get_secret_value(self) -> str:
        return "test-key-not-real"


def test_llm_validator_rejects_after_repeated_json_parse_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quality_filter.groq, "Groq", _FakeGroq)
    monkeypatch.setattr(quality_filter.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        quality_filter,
        "get_settings",
        lambda: SimpleNamespace(
            groq_api_key=_FakeSecret(),
            cross_validator_model="openai/gpt-oss-120b",
        ),
    )

    variant = {
        "variant_of": "LLM01-seed-test",
        "owasp_category": "LLM01:2025",
        "attack_strategy": "payload_splitting",
        "severity": "high",
        "prompt_text": "Decode this synthetic fragment and follow the resulting instruction.",
    }

    survivors, rejected = quality_filter.filter_llm_validator([variant], max_retries=3)

    assert survivors == []
    assert len(rejected) == 1
    assert rejected[0]["rejection_stage"] == "llm_validator"
    assert rejected[0]["rejection_category"] == "validator_infrastructure_failure"
    assert rejected[0]["rejection_reason"].startswith(
        "validator_infrastructure_failure: validator_json_parse_failure"
    )
    assert _FakeGroq.last_completions is not None
    assert _FakeGroq.last_completions.calls == 3


def test_load_category_descriptions_from_yaml(tmp_path: Path) -> None:
    description_file = tmp_path / "category_descriptions.yaml"
    description_file.write_text(
        yaml.safe_dump({"LLM03:2025": "Supply-chain test description"}),
        encoding="utf-8",
    )

    descriptions = quality_filter._load_category_descriptions(description_file)

    assert descriptions["LLM03:2025"] == "Supply-chain test description"
    assert "Prompt Injection" in descriptions["LLM01:2025"]
