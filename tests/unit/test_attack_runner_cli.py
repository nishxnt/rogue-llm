import json
from pathlib import Path

from typer.testing import CliRunner

from src.pipeline.attack_runner import app, select_attacks


def _write_dataset(path: Path) -> None:
    rows = [
        {
            "id": "LLM01-0001",
            "owasp_category": "LLM01:2025",
            "attack_strategy": "direct_override",
            "prompt_text": "prompt 1",
        },
        {
            "id": "LLM02-0001",
            "owasp_category": "LLM02:2025",
            "attack_strategy": "pii_extraction",
            "prompt_text": "prompt 2",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_select_attacks_filters_short_category_name(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]

    selected = select_attacks(rows, category="LLM01")

    assert [attack["id"] for attack in selected] == ["LLM01-0001"]


def test_cli_dry_run_prints_plan_without_target_initialization(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    cache = tmp_path / "cache.sqlite"
    _write_dataset(dataset)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--dataset",
            str(dataset),
            "--cache",
            str(cache),
            "--category",
            "LLM02",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Dry run: 1 attack(s) would execute" in result.output
    assert "LLM02-0001\tLLM02:2025\tpii_extraction" in result.output
    assert not cache.exists()
