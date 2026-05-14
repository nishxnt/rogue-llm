"""Async attack execution pipeline."""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field

from src.pipeline.cache import ResultCache, build_target_version, hash_text
from src.pipeline.rate_limiter import TokenBucketRateLimiter
from src.pipeline.retry import RetryExhaustedError, retry_transient
from src.target_system.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from src.target_system.models import Response

log = structlog.get_logger()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "results"
_DEFAULT_TARGET_MODEL = "llama-3.1-8b-instant"
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class AsyncTargetSystem(Protocol):
    async def aquery(self, prompt: str) -> Response:
        """Run an async single-turn query against the target system."""


class AttackResult(BaseModel):
    """Structured result emitted by Phase 3 attack execution."""

    model_config = ConfigDict(extra="allow")

    attack_id: str
    owasp_category: str
    attack_prompt: str
    target_response: str
    retrieved_chunks: list[str]
    latency_ms: float
    tokens_used: int
    cache_hit: bool
    timestamp: str
    status: str = "success"
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class AttackRunner:
    """Async executor for the versioned OWASP attack dataset."""

    def __init__(
        self,
        *,
        target_system: AsyncTargetSystem,
        dataset_path: Path | str,
        cache_path: Path | str,
        concurrency: int = 5,
        dry_run_sample_n: int | None = None,
        results_root: Path | str = _DEFAULT_RESULTS_ROOT,
        rate_limiter: TokenBucketRateLimiter | None = None,
        retry_sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if dry_run_sample_n is not None and dry_run_sample_n <= 0:
            raise ValueError("dry_run_sample_n must be positive when provided")

        self.target_system = target_system
        self.dataset_path = Path(dataset_path)
        self.cache = ResultCache(cache_path)
        self.concurrency = concurrency
        self.dry_run_sample_n = dry_run_sample_n
        self.results_root = Path(results_root)
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter(rate_per_minute=30, burst=2)
        self.retry_sleeper = retry_sleeper
        self.target_model = _target_model(target_system)
        self.target_version = _target_version(target_system, self.target_model)

    async def run(self) -> list[AttackResult]:
        """Run attacks, persist JSONL results, and return structured records."""
        attacks = self._load_dataset()
        if self.dry_run_sample_n is not None:
            attacks = _stratified_sample(attacks, self.dry_run_sample_n)

        results = await self._run_attacks(attacks)
        self._write_results(results)
        return results

    async def run_with_sample(self, n: int) -> list[AttackResult]:
        """Run a stratified random sample for development iteration."""
        if n <= 0:
            raise ValueError("sample size must be positive")
        attacks = _stratified_sample(self._load_dataset(), n)
        results = await self._run_attacks(attacks)
        self._write_results(results)
        return results

    def close(self) -> None:
        """Close held resources."""
        self.cache.close()

    async def _run_attacks(self, attacks: Sequence[dict[str, Any]]) -> list[AttackResult]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(attack: dict[str, Any]) -> AttackResult:
            async with semaphore:
                return await self._run_one(attack)

        return list(await asyncio.gather(*(run_one(attack) for attack in attacks)))

    async def _run_one(self, attack: dict[str, Any]) -> AttackResult:
        attack_id = str(attack["id"])
        prompt = str(attack["prompt_text"])
        prompt_hash = hash_text(prompt)
        cached = self.cache.get(
            attack_id=attack_id,
            target_model=self.target_model,
            target_version=self.target_version,
            prompt_hash=prompt_hash,
        )
        if cached is not None:
            return AttackResult.model_validate(cached)

        await self.rate_limiter.acquire()
        try:
            response = await retry_transient(
                lambda: self.target_system.aquery(prompt),
                sleeper=self.retry_sleeper,
            )
            result = AttackResult(
                attack_id=attack_id,
                owasp_category=str(attack["owasp_category"]),
                attack_prompt=prompt,
                target_response=response.answer,
                retrieved_chunks=[chunk.content for chunk in response.retrieved_chunks],
                latency_ms=float(response.latency_ms),
                tokens_used=int(response.tokens_used),
                cache_hit=False,
                timestamp=_utc_now(),
                status="success",
                retrieved_doc_ids=[chunk.doc_id for chunk in response.retrieved_chunks],
            )
        except RetryExhaustedError as exc:
            result = AttackResult(
                attack_id=attack_id,
                owasp_category=str(attack["owasp_category"]),
                attack_prompt=prompt,
                target_response="",
                retrieved_chunks=[],
                latency_ms=0.0,
                tokens_used=0,
                cache_hit=False,
                timestamp=_utc_now(),
                status="infrastructure_failure",
                retrieved_doc_ids=[],
                error_type=type(exc.last_error).__name__,
                error_message=str(exc.last_error),
            )

        self.cache.set(
            attack_id=attack_id,
            target_model=self.target_model,
            target_version=self.target_version,
            prompt_hash=prompt_hash,
            result=result.model_dump(),
        )
        return result

    def _load_dataset(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"dataset not found: {self.dataset_path}")
        records = [
            json.loads(line)
            for line in self.dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        log.info("Attack dataset loaded", path=str(self.dataset_path), count=len(records))
        return records

    def _write_results(self, results: Sequence[AttackResult]) -> Path:
        run_dir = self.results_root / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "results.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for result in results:
                fh.write(result.model_dump_json() + "\n")
        log.info("Attack results written", path=str(path), count=len(results))
        return path


def _stratified_sample(attacks: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n >= len(attacks):
        return list(attacks)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attack in attacks:
        groups[str(attack["owasp_category"])].append(attack)

    rng = random.Random(0)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    categories = sorted(groups)
    while len(selected) < n and categories:
        next_categories: list[str] = []
        for category in categories:
            if groups[category] and len(selected) < n:
                selected.append(groups[category].pop())
            if groups[category]:
                next_categories.append(category)
        categories = next_categories
    return selected


def _target_model(target_system: AsyncTargetSystem) -> str:
    settings = getattr(target_system, "_settings", None)
    return str(getattr(settings, "target_model", _DEFAULT_TARGET_MODEL))


def _target_version(target_system: AsyncTargetSystem, target_model: str) -> str:
    settings = getattr(target_system, "_settings", None)
    embedding_model = str(getattr(settings, "embedding_model", _DEFAULT_EMBEDDING_MODEL))
    prompt_template = str(getattr(target_system, "_prompt", ""))
    retrieval_config = {
        "top_k": 4,
        "embedding_model": embedding_model,
        "search_type": "similarity",
    }
    return build_target_version(
        target_model=target_model,
        system_prompt=SYSTEM_PROMPT,
        prompt_template=prompt_template,
        retrieval_config=retrieval_config,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
