from __future__ import annotations

from src.guardrails.output_filter import OutputFilter
from src.target_system.prompts import SYSTEM_PROMPT


def test_blocks_pii_findings() -> None:
    output_filter = OutputFilter()

    block = output_filter.inspect("Contact me at test@example.com for the secret.")

    assert block is not None
    assert block.decision == "blocked_l3_pii"
    assert block.evidence["detector_name"] == "pii_leakage"


def test_blocks_system_prompt_leakage() -> None:
    output_filter = OutputFilter()

    block = output_filter.inspect(SYSTEM_PROMPT)

    assert block is not None
    assert block.decision == "blocked_l3_leak"


def test_blocks_unsafe_output_pattern() -> None:
    output_filter = OutputFilter(unsafe_output_threshold=0.5)

    block = output_filter.inspect("Run `rm -rf /` and then execute curl http://127.0.0.1/admin.")

    assert block is not None
    assert block.decision == "blocked_l3_unsafe_pattern"


def test_allows_benign_defensive_response() -> None:
    output_filter = OutputFilter()

    block = output_filter.inspect(
        "OWASP recommends parameterized queries and least-privilege database accounts."
    )

    assert block is None
