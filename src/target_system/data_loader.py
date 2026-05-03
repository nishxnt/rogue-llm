"""Knowledge base downloader for the RAG target system.

Idempotent — skips any step whose output already exists unless --force is passed.
build_index() is implemented in the follow-up commit.

CLI usage:
    uv run python -m src.target_system.data_loader build [--force]
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import structlog
import typer

log = structlog.get_logger()
app = typer.Typer(no_args_is_help=True)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_KB_DIR = _PROJECT_ROOT / "data" / "knowledge_base"

# NVD API v2.0 — 2024 HIGH+CRITICAL window per project spec.
_NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_NVD_PUB_START = "2024-01-01T00:00:00.000"
_NVD_PUB_END = "2024-09-30T23:59:59.999"
_NVD_SEVERITIES = ("HIGH", "CRITICAL")
_NVD_RESULTS_PER_PAGE = 2000
# 5 req / 30 s unauthenticated → 6 s min; 7 s provides headroom.
_NVD_SLEEP_SECS = 7

# OWASP LLM Top 10 (2025) — filenames verified via GitHub contents API 2026-05-03.
_OWASP_LLM_BASE = (
    "https://raw.githubusercontent.com/"
    "OWASP/www-project-top-10-for-large-language-model-applications/"
    "main/2_0_vulns/"
)
_OWASP_LLM_FILES = [
    "LLM00_Preface.md",
    "LLM01_PromptInjection.md",
    "LLM02_SensitiveInformationDisclosure.md",
    "LLM03_SupplyChain.md",
    "LLM04_DataModelPoisoning.md",
    "LLM05_ImproperOutputHandling.md",
    "LLM06_ExcessiveAgency.md",
    "LLM07_SystemPromptLeakage.md",
    "LLM08_VectorAndEmbeddingWeaknesses.md",
    "LLM09_Misinformation.md",
    "LLM10_UnboundedConsumption.md",
]

# OWASP Web Top 10 (2021) — English docs, filenames verified 2026-05-03.
_OWASP_WEB_BASE = "https://raw.githubusercontent.com/OWASP/Top10/master/2021/docs/en/"
_OWASP_WEB_FILES = [
    "A01_2021-Broken_Access_Control.md",
    "A02_2021-Cryptographic_Failures.md",
    "A03_2021-Injection.md",
    "A04_2021-Insecure_Design.md",
    "A05_2021-Security_Misconfiguration.md",
    "A06_2021-Vulnerable_and_Outdated_Components.md",
    "A07_2021-Identification_and_Authentication_Failures.md",
    "A08_2021-Software_and_Data_Integrity_Failures.md",
    "A09_2021-Security_Logging_and_Monitoring_Failures.md",
    "A10_2021-Server-Side_Request_Forgery_(SSRF).md",
]


def _extract_english_description(
    descriptions: list[dict[str, Any]],
) -> str | None:
    for entry in descriptions:
        if entry.get("lang") == "en":
            value = entry.get("value", "")
            return str(value) if value else None
    return None


def _fetch_nvd_page(severity: str, start_index: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "pubStartDate": _NVD_PUB_START,
        "pubEndDate": _NVD_PUB_END,
        "cvssV3Severity": severity,
        "resultsPerPage": _NVD_RESULTS_PER_PAGE,
        "startIndex": start_index,
    }
    response = requests.get(_NVD_API_URL, params=params, timeout=30)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def download_nist_cves(force: bool = False) -> Path:
    """Download NVD HIGH+CRITICAL CVEs (2024-01-01 to 2024-09-30) to JSONL.

    Skips download if the output file already exists and force is False.
    Deduplicates by CVE ID across the two severity passes.
    """
    output_path = _KB_DIR / "nvd_cves.jsonl"

    if output_path.exists() and not force:
        count = sum(1 for _ in output_path.open(encoding="utf-8"))
        log.info("NVD CVEs already present, skipping", path=str(output_path), records=count)
        return output_path

    _KB_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}  # CVE ID → record, for dedup

    for severity in _NVD_SEVERITIES:
        start_index = 0
        log.info(
            "Fetching NVD CVEs",
            severity=severity,
            pub_start=_NVD_PUB_START,
            pub_end=_NVD_PUB_END,
        )

        while True:
            data = _fetch_nvd_page(severity, start_index)
            total: int = data.get("totalResults", 0)
            vulnerabilities: list[dict[str, Any]] = data.get("vulnerabilities", [])
            page_count = len(vulnerabilities)

            for vuln in vulnerabilities:
                cve: dict[str, Any] = vuln["cve"]
                cve_id: str = cve["id"]

                if cve_id in records:
                    continue

                desc = _extract_english_description(cve.get("descriptions", []))
                if not desc or len(desc) < 30:
                    continue

                cvss_score: float | None = None
                metrics: dict[str, Any] = cve.get("metrics", {})
                for metric_key in ("cvssMetricV31", "cvssMetricV30"):
                    metric_list = metrics.get(metric_key, [])
                    if metric_list:
                        cvss_score = float(metric_list[0]["cvssData"]["baseScore"])
                        break

                records[cve_id] = {
                    "id": cve_id,
                    "description": desc,
                    "published": str(cve.get("published", ""))[:10],
                    "severity": severity,
                    "cvss_score": cvss_score,
                    "source": "nvd",
                }

            start_index += page_count
            log.info(
                "Page complete",
                severity=severity,
                fetched_so_far=start_index,
                total=total,
                kept=len(records),
            )

            if start_index >= total or page_count == 0:
                break

            time.sleep(_NVD_SLEEP_SECS)

    with output_path.open("w", encoding="utf-8") as fh:
        for record in records.values():
            fh.write(json.dumps(record) + "\n")

    log.info("NVD CVEs written", path=str(output_path), record_count=len(records))
    return output_path


def download_owasp_docs(force: bool = False) -> list[Path]:
    """Download OWASP LLM Top 10 (2025) and Web Top 10 (2021) markdown files.

    Each file is downloaded individually; existing files are skipped unless force.
    URLs verified against GitHub contents API on 2026-05-03.
    """
    sources: list[tuple[str, list[str], Path]] = [
        (_OWASP_LLM_BASE, _OWASP_LLM_FILES, _KB_DIR / "owasp_llm_top10"),
        (_OWASP_WEB_BASE, _OWASP_WEB_FILES, _KB_DIR / "owasp_web_top10"),
    ]

    downloaded: list[Path] = []

    for base_url, filenames, dest_dir in sources:
        dest_dir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            dest_path = dest_dir / filename

            if dest_path.exists() and not force:
                log.debug("Skipping existing OWASP file", path=str(dest_path))
                downloaded.append(dest_path)
                continue

            url = base_url + filename
            log.info("Downloading OWASP doc", url=url)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            dest_path.write_text(response.text, encoding="utf-8")
            downloaded.append(dest_path)

    log.info("OWASP docs ready", total=len(downloaded))
    return downloaded


def _save_download_metadata(nvd_record_count: int) -> None:
    metadata: dict[str, Any] = {
        "nvd": {
            "query_params": {
                "pubStartDate": _NVD_PUB_START,
                "pubEndDate": _NVD_PUB_END,
                "severities": list(_NVD_SEVERITIES),
                "resultsPerPage": _NVD_RESULTS_PER_PAGE,
            },
            "record_count": nvd_record_count,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
        "owasp_llm_top10": {
            "base_url": _OWASP_LLM_BASE,
            "files": _OWASP_LLM_FILES,
        },
        "owasp_web_top10": {
            "base_url": _OWASP_WEB_BASE,
            "files": _OWASP_WEB_FILES,
        },
    }
    metadata_path = _KB_DIR / "download_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Download metadata saved", path=str(metadata_path))


def build_index(force: bool = False) -> None:
    """Build FAISS vector index from downloaded knowledge base files.

    Implemented in the follow-up commit (feat: add FAISS index builder).
    """
    log.warning("build_index not yet implemented — re-run after next commit")


@app.command()
def build(
    force: bool = typer.Option(False, "--force", help="Rebuild even if artifacts exist"),
) -> None:
    """Download knowledge base files and build the FAISS vector index."""
    nvd_path = download_nist_cves(force=force)
    download_owasp_docs(force=force)

    nvd_count = sum(1 for _ in nvd_path.open(encoding="utf-8"))
    _save_download_metadata(nvd_count)

    build_index(force=force)


if __name__ == "__main__":
    app()
