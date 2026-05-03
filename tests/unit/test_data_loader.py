"""Unit tests for src.target_system.data_loader.

No network calls. Uses tmp_path and fixture files from tests/unit/fixtures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.target_system.data_loader import (
    _extract_english_description,
    _load_nvd_documents,
    _load_owasp_documents,
)

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# _extract_english_description
# ---------------------------------------------------------------------------


def test_extract_english_description_returns_english() -> None:
    descriptions = [
        {"lang": "es", "value": "Una vulnerabilidad"},
        {"lang": "en", "value": "A SQL injection vulnerability"},
    ]
    assert _extract_english_description(descriptions) == "A SQL injection vulnerability"


def test_extract_english_description_returns_none_when_missing() -> None:
    descriptions = [{"lang": "de", "value": "Eine Schwachstelle"}]
    assert _extract_english_description(descriptions) is None


def test_extract_english_description_returns_none_for_empty_list() -> None:
    assert _extract_english_description([]) is None


# ---------------------------------------------------------------------------
# _load_nvd_documents — uses tmp_path to simulate KB dir
# ---------------------------------------------------------------------------


def test_load_nvd_documents_parses_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.target_system.data_loader._KB_DIR",
        _FIXTURES,
    )
    # Fixture JSONL has 5 records, all descriptions ≥ 30 chars.
    docs = _load_nvd_documents()
    assert len(docs) == 5


def test_load_nvd_documents_preserves_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.target_system.data_loader._KB_DIR", _FIXTURES)
    docs = _load_nvd_documents()
    ids = {d.metadata["doc_id"] for d in docs}
    assert "CVE-2024-00001" in ids
    assert "CVE-2024-00003" in ids


def test_load_nvd_documents_sets_source_nvd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.target_system.data_loader._KB_DIR", _FIXTURES)
    docs = _load_nvd_documents()
    assert all(d.metadata["source"] == "nvd" for d in docs)


def test_load_nvd_documents_raises_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.target_system.data_loader._KB_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        _load_nvd_documents()


# ---------------------------------------------------------------------------
# Chunking — split a known document and verify overlap + metadata survive
# ---------------------------------------------------------------------------


def test_chunking_preserves_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from src.target_system.data_loader import _load_nvd_documents

    monkeypatch.setattr("src.target_system.data_loader._KB_DIR", _FIXTURES)
    docs = _load_nvd_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    # Every chunk must carry source + doc_id from the parent document.
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "doc_id" in chunk.metadata


def test_chunking_produces_at_least_one_chunk_per_doc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    monkeypatch.setattr("src.target_system.data_loader._KB_DIR", _FIXTURES)
    docs = _load_nvd_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= len(docs)


# ---------------------------------------------------------------------------
# Build index idempotency (filesystem only — no embedding calls)
# ---------------------------------------------------------------------------


def test_build_index_skips_when_index_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.target_system.data_loader import build_index

    index_path = tmp_path / "faiss_index"
    index_path.mkdir()
    monkeypatch.setattr("src.target_system.data_loader._INDEX_DIR", index_path)

    called: list[bool] = []

    def _fake_load_nvd() -> list:  # type: ignore[return]
        called.append(True)

    monkeypatch.setattr("src.target_system.data_loader._load_nvd_documents", _fake_load_nvd)

    build_index(force=False)
    assert not called, "build_index should skip when index already exists"


def test_build_index_force_flag_ignores_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=True triggers rebuild even when index dir exists.

    We only check that the load functions are called; we don't run the full
    embedding pipeline in unit tests.
    """
    from src.target_system.data_loader import build_index

    index_path = tmp_path / "faiss_index"
    index_path.mkdir()
    monkeypatch.setattr("src.target_system.data_loader._INDEX_DIR", index_path)
    monkeypatch.setattr("src.target_system.data_loader._KB_DIR", _FIXTURES)

    called: list[bool] = []

    original_load_nvd = _load_nvd_documents
    original_load_owasp = _load_owasp_documents

    def _spy_load_nvd() -> list:
        called.append(True)
        return original_load_nvd()

    def _spy_load_owasp() -> list:
        return original_load_owasp()

    import unittest.mock as mock

    with (
        mock.patch("src.target_system.data_loader._load_nvd_documents", _spy_load_nvd),
        mock.patch("src.target_system.data_loader._load_owasp_documents", _spy_load_owasp),
        mock.patch("src.target_system.data_loader.HuggingFaceEmbeddings"),
        mock.patch("src.target_system.data_loader.FAISS") as mock_faiss,
    ):
        mock_faiss.from_documents.return_value.save_local = lambda *a, **kw: None
        build_index(force=True)

    assert called, "build_index(force=True) should call _load_nvd_documents"


# ---------------------------------------------------------------------------
# download_metadata.json content
# ---------------------------------------------------------------------------


def test_save_download_metadata_writes_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.target_system.data_loader import _save_download_metadata

    monkeypatch.setattr("src.target_system.data_loader._KB_DIR", tmp_path)
    _save_download_metadata(nvd_record_count=750)

    metadata_path = tmp_path / "download_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["nvd"]["record_count"] == 750
    assert metadata["nvd"]["query_params"]["pubStartDate"] == "2024-01-01T00:00:00.000"
    assert "HIGH" in metadata["nvd"]["query_params"]["severities"]
    assert "CRITICAL" in metadata["nvd"]["query_params"]["severities"]
