"""SQLite-backed result cache for attack execution.

The cache persists each completed attack result immediately. Keys include
both prompt identity and target-system version so reruns are cheap while
configuration changes invalidate stale responses.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = 1


class CacheVersionError(RuntimeError):
    """Raised when an existing cache uses an unsupported schema version."""


def hash_text(value: str) -> str:
    """Return a stable SHA-256 hash for text cache keys."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for structured cache key material."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hash_text(canonical)


def build_target_version(
    *,
    target_model: str,
    system_prompt: str,
    prompt_template: str,
    retrieval_config: Mapping[str, Any],
) -> str:
    """Hash target configuration that should invalidate cached responses."""
    return hash_json(
        {
            "target_model": target_model,
            "system_prompt_hash": hash_text(system_prompt),
            "prompt_template_hash": hash_text(prompt_template),
            "retrieval_config": retrieval_config,
        }
    )


class ResultCache:
    """SQLite-backed cache keyed by attack and target version."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> ResultCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(
        self,
        *,
        attack_id: str,
        target_model: str,
        target_version: str,
        prompt_hash: str,
    ) -> dict[str, Any] | None:
        """Return a cached result, or ``None`` on cache miss."""
        row = self._conn.execute(
            """
            SELECT result_json
            FROM attack_results
            WHERE attack_id = ?
              AND target_model = ?
              AND target_version = ?
              AND prompt_hash = ?
            """,
            (attack_id, target_model, target_version, prompt_hash),
        ).fetchone()
        if row is None:
            return None

        result = cast("dict[str, Any]", json.loads(str(row["result_json"])))
        result["cache_hit"] = True
        return result

    def set(
        self,
        *,
        attack_id: str,
        target_model: str,
        target_version: str,
        prompt_hash: str,
        result: Mapping[str, Any],
    ) -> None:
        """Persist one attack result immediately."""
        now = _utc_now()
        payload = dict(result)
        payload["cache_hit"] = False
        result_json = json.dumps(payload, sort_keys=True, default=str)

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO attack_results (
                    attack_id,
                    target_model,
                    target_version,
                    prompt_hash,
                    result_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attack_id, target_model, target_version, prompt_hash)
                DO UPDATE SET
                    result_json = excluded.result_json,
                    updated_at = excluded.updated_at
                """,
                (
                    attack_id,
                    target_model,
                    target_version,
                    prompt_hash,
                    result_json,
                    now,
                    now,
                ),
            )

    def get_metric_score(
        self,
        *,
        attack_id: str,
        metric_name: str,
        judge_model: str,
        judge_version: str,
        input_hash: str,
    ) -> dict[str, Any] | None:
        """Return a cached metric score, or ``None`` on cache miss."""
        row = self._conn.execute(
            """
            SELECT score_json
            FROM metric_scores
            WHERE attack_id = ?
              AND metric_name = ?
              AND judge_model = ?
              AND judge_version = ?
              AND input_hash = ?
            """,
            (attack_id, metric_name, judge_model, judge_version, input_hash),
        ).fetchone()
        if row is None:
            return None
        return cast("dict[str, Any]", json.loads(str(row["score_json"])))

    def set_metric_score(
        self,
        *,
        attack_id: str,
        metric_name: str,
        judge_model: str,
        judge_version: str,
        input_hash: str,
        score: Mapping[str, Any],
    ) -> None:
        """Persist one metric score immediately."""
        now = _utc_now()
        score_json = json.dumps(score, sort_keys=True, default=str)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO metric_scores (
                    attack_id,
                    metric_name,
                    judge_model,
                    judge_version,
                    input_hash,
                    score_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attack_id, metric_name, judge_model, judge_version, input_hash)
                DO UPDATE SET
                    score_json = excluded.score_json,
                    updated_at = excluded.updated_at
                """,
                (
                    attack_id,
                    metric_name,
                    judge_model,
                    judge_version,
                    input_hash,
                    score_json,
                    now,
                    now,
                ),
            )

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_schema_version (
                    version INTEGER NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_scores (
                    attack_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    judge_model TEXT NOT NULL,
                    judge_version TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    score_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        attack_id,
                        metric_name,
                        judge_model,
                        judge_version,
                        input_hash
                    )
                )
                """)
            row = self._conn.execute(
                "SELECT version FROM cache_schema_version ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO cache_schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, _utc_now()),
                )
            elif int(row["version"]) != SCHEMA_VERSION:
                raise CacheVersionError(
                    f"unsupported cache schema version {row['version']}; "
                    f"expected {SCHEMA_VERSION}"
                )

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS attack_results (
                    attack_id TEXT NOT NULL,
                    target_model TEXT NOT NULL,
                    target_version TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        attack_id,
                        target_model,
                        target_version,
                        prompt_hash
                    )
                )
                """)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
