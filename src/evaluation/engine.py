"""Phase 4 evaluation engine scaffold."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import structlog
from pydantic import BaseModel, Field

from src.evaluation.config import DEFAULT_CONCURRENCY
from src.pipeline.cache import ResultCache, hash_json

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = structlog.get_logger()


class AttackEvaluationInput(BaseModel):
    """One Phase 3 attack result prepared for metric scoring."""

    attack_id: str
    owasp_category: str
    attack_prompt: str
    target_response: str
    retrieved_chunks: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    status: str = "success"
    metadata: dict[str, object] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """Normalized metric output used by the engine and scorer."""

    attack_id: str
    metric_name: str
    score: float | None
    skipped: bool = False
    reason: str | None = None
    evidence: dict[str, object] = Field(default_factory=dict)
    judge_model: str = "deterministic"
    judge_version: str = "v1"


class EvaluationMetric(Protocol):
    """Async metric interface shared by deterministic and LLM-graded metrics."""

    name: str
    judge_model: str
    judge_version: str

    async def score(self, attack: AttackEvaluationInput) -> MetricResult:
        """Score one attack result."""


class EvaluationEngine:
    """Orchestrates Phase 4 metric scoring over Phase 3 results."""

    def __init__(
        self,
        *,
        results_path: Path | str,
        cache_path: Path | str,
        metrics: Sequence[EvaluationMetric],
        output_root: Path | str = "results",
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        self.results_path = Path(results_path)
        self.cache = ResultCache(cache_path)
        self.metrics = list(metrics)
        self.output_root = Path(output_root)
        self.concurrency = concurrency

    async def run(self) -> list[MetricResult]:
        """Score all loaded attack results and write scores.jsonl."""
        attacks = self.load_attack_results()
        semaphore = asyncio.Semaphore(self.concurrency)

        async def score_one(
            metric: EvaluationMetric, attack: AttackEvaluationInput
        ) -> MetricResult:
            async with semaphore:
                return await self._score_with_cache(metric, attack)

        tasks = [score_one(metric, attack) for attack in attacks for metric in self.metrics]
        scores = list(await asyncio.gather(*tasks))
        self.write_scores(scores)
        return scores

    def close(self) -> None:
        """Close held resources."""
        self.cache.close()

    def load_attack_results(self) -> list[AttackEvaluationInput]:
        """Load Phase 3 results.jsonl records."""
        if not self.results_path.exists():
            raise FileNotFoundError(f"results not found: {self.results_path}")
        attacks: list[AttackEvaluationInput] = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            known = {
                "attack_id",
                "owasp_category",
                "attack_prompt",
                "target_response",
                "retrieved_chunks",
                "retrieved_doc_ids",
                "status",
            }
            metadata = {key: value for key, value in row.items() if key not in known}
            attacks.append(
                AttackEvaluationInput(
                    attack_id=str(row["attack_id"]),
                    owasp_category=str(row["owasp_category"]),
                    attack_prompt=str(row["attack_prompt"]),
                    target_response=str(row.get("target_response", "")),
                    retrieved_chunks=list(row.get("retrieved_chunks", [])),
                    retrieved_doc_ids=list(row.get("retrieved_doc_ids", [])),
                    status=str(row.get("status", "success")),
                    metadata=metadata,
                )
            )
        log.info("Evaluation inputs loaded", path=str(self.results_path), count=len(attacks))
        return attacks

    def write_scores(self, scores: Sequence[MetricResult]) -> Path:
        """Write per-metric scores to the Phase 4 run directory."""
        run_dir = self.output_root / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "scores.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for score in scores:
                fh.write(score.model_dump_json() + "\n")
        log.info("Evaluation scores written", path=str(path), count=len(scores))
        return path

    async def _score_with_cache(
        self,
        metric: EvaluationMetric,
        attack: AttackEvaluationInput,
    ) -> MetricResult:
        input_hash = metric_input_hash(metric.name, attack)
        cached = self.cache.get_metric_score(
            attack_id=attack.attack_id,
            metric_name=metric.name,
            judge_model=metric.judge_model,
            judge_version=metric.judge_version,
            input_hash=input_hash,
        )
        if cached is not None:
            return MetricResult.model_validate(cached)

        result = await metric.score(attack)
        self.cache.set_metric_score(
            attack_id=attack.attack_id,
            metric_name=metric.name,
            judge_model=metric.judge_model,
            judge_version=metric.judge_version,
            input_hash=input_hash,
            score=result.model_dump(),
        )
        return result


def metric_input_hash(metric_name: str, attack: AttackEvaluationInput) -> str:
    """Hash metric-relevant inputs for score cache invalidation."""
    payload: Mapping[str, object] = {
        "metric_name": metric_name,
        "attack_id": attack.attack_id,
        "owasp_category": attack.owasp_category,
        "attack_prompt": attack.attack_prompt,
        "target_response": attack.target_response,
        "retrieved_chunks": attack.retrieved_chunks,
        "retrieved_doc_ids": attack.retrieved_doc_ids,
        "status": attack.status,
        "metadata": attack.metadata,
    }
    return hash_json(payload)
