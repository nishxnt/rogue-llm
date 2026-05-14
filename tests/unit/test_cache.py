import sqlite3
from pathlib import Path

import pytest

from src.pipeline.cache import (
    SCHEMA_VERSION,
    CacheVersionError,
    ResultCache,
    build_target_version,
    hash_text,
)


def test_cache_creates_schema_and_version_table(tmp_path: Path) -> None:
    cache_path = tmp_path / "results_cache.sqlite"

    with ResultCache(cache_path):
        pass

    conn = sqlite3.connect(cache_path)
    try:
        version = conn.execute("SELECT version FROM cache_schema_version").fetchone()[0]
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        conn.close()

    assert version == SCHEMA_VERSION
    assert "attack_results" in tables
    assert "cache_schema_version" in tables


def test_cache_miss_then_hit_sets_cache_hit_flag(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "results_cache.sqlite")
    key = {
        "attack_id": "LLM01-0001",
        "target_model": "llama-3.1-8b-instant",
        "target_version": "version-a",
        "prompt_hash": hash_text("attack prompt"),
    }
    result = {
        "attack_id": "LLM01-0001",
        "target_response": "synthetic response",
        "cache_hit": False,
    }

    try:
        assert cache.get(**key) is None

        cache.set(**key, result=result)
        cached = cache.get(**key)
    finally:
        cache.close()

    assert cached is not None
    assert cached["attack_id"] == "LLM01-0001"
    assert cached["target_response"] == "synthetic response"
    assert cached["cache_hit"] is True


def test_target_version_change_invalidates_cache(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "results_cache.sqlite")
    base_key = {
        "attack_id": "LLM01-0001",
        "target_model": "llama-3.1-8b-instant",
        "prompt_hash": hash_text("attack prompt"),
    }

    try:
        cache.set(
            **base_key,
            target_version="version-a",
            result={"attack_id": "LLM01-0001", "target_response": "old"},
        )

        assert cache.get(**base_key, target_version="version-a") is not None
        assert cache.get(**base_key, target_version="version-b") is None
    finally:
        cache.close()


def test_prompt_hash_change_invalidates_cache(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "results_cache.sqlite")
    base_key = {
        "attack_id": "LLM01-0001",
        "target_model": "llama-3.1-8b-instant",
        "target_version": "version-a",
    }

    try:
        cache.set(
            **base_key,
            prompt_hash=hash_text("attack prompt"),
            result={"attack_id": "LLM01-0001", "target_response": "old"},
        )

        assert cache.get(**base_key, prompt_hash=hash_text("attack prompt")) is not None
        assert cache.get(**base_key, prompt_hash=hash_text("changed prompt")) is None
    finally:
        cache.close()


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    cache_path = tmp_path / "results_cache.sqlite"
    conn = sqlite3.connect(cache_path)
    try:
        conn.execute(
            "CREATE TABLE cache_schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO cache_schema_version (version, applied_at) VALUES (?, ?)",
            (999, "2026-05-14T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(CacheVersionError):
        ResultCache(cache_path)


def test_build_target_version_includes_retrieval_config() -> None:
    base = build_target_version(
        target_model="llama-3.1-8b-instant",
        system_prompt="system",
        prompt_template="template",
        retrieval_config={"top_k": 4, "embedding_model": "all-MiniLM-L6-v2"},
    )
    changed_top_k = build_target_version(
        target_model="llama-3.1-8b-instant",
        system_prompt="system",
        prompt_template="template",
        retrieval_config={"top_k": 5, "embedding_model": "all-MiniLM-L6-v2"},
    )

    assert base != changed_top_k
