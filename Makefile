.PHONY: install test lint format run report clean help

help:
	@echo "Targets: install | test | lint | format | run | report | clean"

install:
	uv sync --all-groups
	uv run pre-commit install

test:
	uv run pytest -q --cov=src --cov-report=term-missing

lint:
	uv run ruff check src tests
	uv run black --check src tests
	uv run mypy src

format:
	uv run black src tests
	uv run ruff check --fix src tests

run:
	@echo "Pipeline run lands in Phase 6 (src/cli.py). Phase 0 is scaffold-only."

report:
	@echo "Report generation lands in Phase 6 (src/reporting/)."

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf cache/* reports/run_* build dist *.egg-info
	@touch cache/.gitkeep
