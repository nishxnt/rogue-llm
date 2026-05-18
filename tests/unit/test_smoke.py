"""Smoke test: settings load and pinned model IDs are present."""

import pytest

from src.config import Settings, get_settings


def test_settings_load_with_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    settings = get_settings()
    assert settings.groq_api_key.get_secret_value() == "test-key-not-real"


def test_pinned_model_ids() -> None:
    s = Settings(groq_api_key="x")
    assert s.target_model == "llama-3.1-8b-instant"
    assert s.mutator_model == "llama-3.3-70b-versatile"
    assert s.judge_model == "qwen/qwen3-32b"
    assert s.cross_validator_model == "openai/gpt-oss-120b"
    assert s.safety_model == "openai/gpt-oss-safeguard-20b"


def test_observability_keys_optional() -> None:
    s = Settings(groq_api_key="x")
    # Either None or an empty-string placeholder (from .env with blank values) is acceptable.
    assert s.langsmith_api_key is None or s.langsmith_api_key.get_secret_value() == ""
    assert s.wandb_api_key is None or s.wandb_api_key.get_secret_value() == ""
