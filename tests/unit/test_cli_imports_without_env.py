from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_cli_modules_import_without_groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    for module_name in (
        "src.pipeline.attack_runner",
        "src.evaluation.cli",
        "src.guardrails.cli",
        "src.reporting.cli",
    ):
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)
