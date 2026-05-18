from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def mock_groq_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy-key-no-real-api")
