"""Baseline RAGAS evaluation pipeline.

Cross-family judge: openai/gpt-oss-120b grades llama-3.1-8b-instant responses.
This matches PROJECT_SPEC.md §5 — judge and target must be different model families
to prevent correlated failures inflating faithfulness scores.

Two-step workflow:
    Step 1 — generate 30 semi-synthetic QA candidates and save for review:
        uv run python -m src.evaluation.baseline generate

    Step 2 — after manually reviewing / editing data/eval/baseline_qa.jsonl,
    run RAGAS Faithfulness scoring:
        uv run python -m src.evaluation.baseline score

Resume safety: a checkpoint JSONL is written after every scored entry. If the run
is interrupted, re-running `score` will skip already-scored entries and continue
from where it left off. The checkpoint is deleted on successful completion.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import typer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import get_settings

log = structlog.get_logger()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_KB_DIR = _PROJECT_ROOT / "data" / "knowledge_base"
_INDEX_DIR = _PROJECT_ROOT / "data" / "index" / "faiss_index"
_EVAL_DIR = _PROJECT_ROOT / "data" / "eval"
_RESULTS_DIR = _PROJECT_ROOT / "results"
_QA_PATH = _EVAL_DIR / "baseline_qa.jsonl"
_SCORES_PATH = _RESULTS_DIR / "baseline_ragas.json"
_CHECKPOINT_PATH = _RESULTS_DIR / "baseline_ragas_checkpoint.jsonl"

_SEMI_SYNTHETIC_COUNT = 30

_GENERATE_SYSTEM = """\
You are a cybersecurity question generator. Given a security document excerpt, \
write exactly ONE question a security professional might ask, then provide a \
concise factual answer based ONLY on the excerpt — no external knowledge.

Output valid JSON on a single line with this exact structure:
{"question": "<question text>", "ground_truth": "<answer text>"}

Rules:
- The answer must be fully supported by the excerpt.
- Do not add information not present in the excerpt.
- Questions must be specific and answerable (not "tell me everything about X").
- Minimum answer length: 20 words."""

_GENERATE_HUMAN = """\
Document excerpt:
---
{chunk}
---

Source: {source} | ID: {doc_id}

Generate the JSON now:"""


def _load_chunks() -> list[dict[str, Any]]:
    """Load all KB chunks for sampling — NVD + OWASP docs."""
    chunks: list[dict[str, Any]] = []

    nvd_path = _KB_DIR / "nvd_cves.jsonl"
    if nvd_path.exists():
        with nvd_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                desc = record.get("description", "")
                if len(desc) >= 80:  # skip very short NVD entries
                    chunks.append(
                        {
                            "content": desc,
                            "source": "nvd",
                            "doc_id": record.get("id", ""),
                        }
                    )

    for owasp_dir in ("owasp_llm_top10", "owasp_web_top10"):
        owasp_path = _KB_DIR / owasp_dir
        if owasp_path.exists():
            for md_file in sorted(owasp_path.glob("*.md")):
                text = md_file.read_text(encoding="utf-8")
                # Split on double-newlines to get paragraph-level chunks
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 80]
                source = "owasp_llm" if "llm" in owasp_dir else "owasp_web"
                for para in paragraphs:
                    chunks.append(
                        {
                            "content": para[:1200],  # cap at ~1200 chars
                            "source": source,
                            "doc_id": md_file.stem,
                        }
                    )

    log.info("Loaded KB chunks for QA sampling", total=len(chunks))
    return chunks


def _stratified_sample(chunks: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Sample n chunks with rough source stratification."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(c)

    sources = list(by_source.keys())
    per_source = max(1, n // len(sources))
    sampled: list[dict[str, Any]] = []
    for src in sources:
        pool = by_source[src]
        random.shuffle(pool)
        sampled.extend(pool[:per_source])

    # Top up to exactly n if needed
    remaining = [c for c in chunks if c not in sampled]
    random.shuffle(remaining)
    sampled.extend(remaining[: n - len(sampled)])
    return sampled[:n]


def _generate_qa(
    chunk: dict[str, Any],
    llm: ChatGroq,
    idx: int,
) -> dict[str, Any] | None:
    """Call mutator LLM to generate one QA pair from a chunk. Returns None on failure."""
    prompt = _GENERATE_HUMAN.format(
        chunk=chunk["content"],
        source=chunk["source"],
        doc_id=chunk["doc_id"],
    )
    try:
        raw = llm.invoke(
            [
                SystemMessage(content=_GENERATE_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        text = str(raw.content).strip()
        # Extract the JSON — it may be wrapped in ```json ... ``` fences
        if "```" in text:
            text = text.split("```")[-2].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        qa = json.loads(text)
        if not isinstance(qa.get("question"), str) or not isinstance(qa.get("ground_truth"), str):
            raise ValueError("Missing question or ground_truth keys")
        return {
            "id": f"semi-{idx:03d}",
            "question": qa["question"].strip(),
            "ground_truth": qa["ground_truth"].strip(),
            "reference_doc_id": chunk["doc_id"],
            "source": chunk["source"],
            "construction": "semi-synthetic",
        }
    except Exception as exc:
        log.warning("QA generation failed for chunk", doc_id=chunk["doc_id"], error=str(exc))
        return None


def _run_generate() -> None:
    settings = get_settings()

    if not _KB_DIR.exists() or not any(_KB_DIR.glob("**/*.*")):
        log.error(
            "Knowledge base not found — run the data loader first",
            cmd="uv run python -m src.target_system.data_loader build",
        )
        raise typer.Exit(code=1)

    chunks = _load_chunks()
    if not chunks:
        log.error("No usable chunks found in knowledge base")
        raise typer.Exit(code=1)

    sampled = _stratified_sample(chunks, _SEMI_SYNTHETIC_COUNT)
    log.info("Sampled chunks for QA generation", n=len(sampled))

    llm = ChatGroq(
        model_name=settings.mutator_model,
        temperature=0.4,
        api_key=settings.groq_api_key.get_secret_value(),
    )

    results: list[dict[str, Any]] = []
    for i, chunk in enumerate(sampled):
        log.info("Generating QA", idx=i + 1, total=len(sampled), doc_id=chunk["doc_id"])
        qa = _generate_qa(chunk, llm, i)
        if qa:
            results.append(qa)
        # Stay within Groq's 30 RPM limit for mutator (1000 RPD / 30s window)
        time.sleep(2.5)

    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with _QA_PATH.open("w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    log.info(
        "QA candidates saved — review before scoring",
        path=str(_QA_PATH),
        generated=len(results),
        requested=_SEMI_SYNTHETIC_COUNT,
    )
    print(f"\n{'=' * 60}")
    print(f"Generated {len(results)} semi-synthetic QA candidates.")
    print(f"File: {_QA_PATH}")
    print("\nNext steps:")
    print("  1. Review and edit data/eval/baseline_qa.jsonl")
    print("  2. Add 20 hand-written entries (construction: 'hand-written')")
    print("  3. Run: uv run python -m src.evaluation.baseline score")
    print(f"{'=' * 60}\n")


def _load_checkpoint() -> dict[str, dict[str, Any]]:
    """Load previously scored entries from checkpoint. Returns id → record map."""
    pre_scored: dict[str, dict[str, Any]] = {}
    if not _CHECKPOINT_PATH.exists():
        return pre_scored
    with _CHECKPOINT_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("faithfulness_score") is not None:
                    pre_scored[record["id"]] = record
            except json.JSONDecodeError:
                pass
    return pre_scored


async def _score_all(
    qa_records: list[dict[str, Any]],
    chatbot: Any,
    scorer: Any,
    pre_scored: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score QA records with RAGAS Faithfulness. Skips pre-scored entries.

    Writes a checkpoint after every successful score so progress survives interruption.
    """
    # Preserve ordering: pre-scored entries interleaved at their original positions
    result_map: dict[str, dict[str, Any]] = dict(pre_scored)

    to_score = [r for r in qa_records if r["id"] not in pre_scored]
    log.info(
        "Scoring plan",
        total=len(qa_records),
        pre_scored=len(pre_scored),
        to_score=len(to_score),
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(to_score):
        log.info(
            "Scoring",
            idx=i + 1,
            total=len(to_score),
            question=record["question"][:60],
        )
        try:
            response = chatbot.query(record["question"])
            result = await scorer.ascore(
                user_input=record["question"],
                response=response.answer,
                retrieved_contexts=[c.content for c in response.retrieved_chunks],
            )
            entry: dict[str, Any] = {
                **record,
                "target_answer": response.answer,
                "faithfulness_score": float(result.value),
                "retrieved_doc_ids": [c.doc_id for c in response.retrieved_chunks],
            }
            # Persist immediately — a subsequent interruption cannot lose this score.
            with _CHECKPOINT_PATH.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
            result_map[record["id"]] = entry
            # gpt-oss-120b: separate TPM pool from the target model; 1000 RPD / ~30 RPM.
            # RAGAS makes 4-6 internal calls per question (statement extraction + NLI per
            # statement). At 8s sleep + ~5s latency = ~8 iter/min × 5 calls = 40 calls/min.
            # Below the 30 RPM ceiling with headroom.
            await asyncio.sleep(8.0)
        except Exception as exc:
            log.warning("Scoring failed", question=record["question"][:60], error=str(exc))
            result_map[record["id"]] = {**record, "faithfulness_score": None, "error": str(exc)}

    # Return in original QA order
    return [
        result_map.get(r["id"], {**r, "faithfulness_score": None, "error": "not scored"})
        for r in qa_records
    ]


def _run_score() -> None:
    # Import here to avoid loading heavy deps when just generating
    import instructor
    from groq import AsyncGroq as AsyncGroqSDK
    from ragas.llms.base import InstructorLLM, InstructorModelArgs
    from ragas.metrics.collections import Faithfulness

    from src.target_system.rag_chatbot import RAGChatbot

    if not _QA_PATH.exists():
        log.error(
            "QA file not found — run generate first",
            cmd="uv run python -m src.evaluation.baseline generate",
        )
        raise typer.Exit(code=1)

    if not _INDEX_DIR.exists():
        log.error(
            "FAISS index not found — run the data loader first",
            cmd="uv run python -m src.target_system.data_loader build",
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    qa_records: list[dict[str, Any]] = []
    with _QA_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                qa_records.append(json.loads(line))

    log.info("Loaded QA records for scoring", count=len(qa_records))

    # Resume: load checkpoint so we skip already-scored entries.
    pre_scored = _load_checkpoint()
    if pre_scored:
        log.info(
            "Resuming from checkpoint",
            already_scored=len(pre_scored),
            remaining=len(qa_records) - len(pre_scored),
            checkpoint=str(_CHECKPOINT_PATH),
        )

    # Judge model selection (confirmed 2026-05-04, cross-family design per PROJECT_SPEC §5):
    # - qwen/qwen3-32b: cannot produce structured outputs on Groq (tool_use_failed /
    #   json_validate_failed in both TOOLS and JSON instructor modes).
    # - openai/gpt-oss-120b: confirmed working in both Mode.JSON and Mode.TOOLS. Different
    #   model family from target (OpenAI vs Meta/Llama) — satisfies cross-family requirement.
    #   Separate TPM pool; 1000 RPD free tier is sufficient for 40-question baseline scoring.
    # - llama-3.1-8b-instant (previously attempted): shares TPM pool with the RAG target,
    #   causing combined token exhaustion; also same-family, violating PROJECT_SPEC §5.
    effective_judge = settings.judge_model
    instructor_mode = instructor.Mode.JSON
    if "qwen" in settings.judge_model.lower():
        effective_judge = settings.cross_validator_model  # openai/gpt-oss-120b
        instructor_mode = instructor.Mode.TOOLS
        log.warning(
            "qwen3-32b structured output unsupported on Groq — switching to cross-family judge",
            fallback=effective_judge,
            mode="TOOLS",
            note="Cross-family judge restored per PROJECT_SPEC.md §5 (OpenAI family vs Llama target)",
        )

    # AsyncGroq required: RAGAS ascore() calls agenerate() which needs an async client.
    groq_client = instructor.from_groq(
        AsyncGroqSDK(api_key=settings.groq_api_key.get_secret_value()),
        mode=instructor_mode,
    )
    judge_llm = InstructorLLM(
        client=groq_client,
        model=effective_judge,
        provider="groq",
        model_args=InstructorModelArgs(max_tokens=4096),
    )
    scorer = Faithfulness(llm=judge_llm)
    chatbot = RAGChatbot(settings=settings)

    scored = asyncio.run(_score_all(qa_records, chatbot, scorer, pre_scored))

    def _stratify(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[float]] = {}
        for r in records:
            v = r.get(key, "unknown")
            fs = r.get("faithfulness_score")
            if fs is not None:
                groups.setdefault(str(v), []).append(float(fs))
        return {
            grp: {
                "mean": round(sum(vals) / len(vals), 4),
                "n": len(vals),
            }
            for grp, vals in sorted(groups.items())
        }

    valid_scores = [
        s["faithfulness_score"] for s in scored if s.get("faithfulness_score") is not None
    ]
    error_count = sum(1 for s in scored if s.get("error") is not None)
    mean_faithfulness = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    by_source: dict[str, dict[str, Any]] = _stratify(scored, "source")
    by_construction: dict[str, dict[str, Any]] = _stratify(scored, "construction")

    output = {
        "run_at": datetime.now(UTC).isoformat(),
        "model_target": settings.target_model,
        "model_judge_configured": settings.judge_model,
        "model_judge_effective": effective_judge,
        "judge_model": effective_judge,
        "run_metadata": {
            "total_qa": len(qa_records),
            "scored": len(valid_scores),
            "errored": error_count,
            "resumed_from_checkpoint": len(pre_scored),
        },
        "qa_count": len(qa_records),
        "scored_count": len(valid_scores),
        "mean_faithfulness": round(mean_faithfulness, 4),
        "by_source": by_source,
        "by_construction": by_construction,
        "scores": scored,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with _SCORES_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    # Clean up checkpoint on successful write — all scores are now in the JSON.
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        log.info("Checkpoint deleted after successful write", path=str(_CHECKPOINT_PATH))

    log.info(
        "Baseline RAGAS scoring complete",
        judge=effective_judge,
        mean_faithfulness=mean_faithfulness,
        scored=len(valid_scores),
        errored=error_count,
        output=str(_SCORES_PATH),
    )
    print(f"\n{'=' * 60}")
    print(f"RAGAS Faithfulness baseline: {mean_faithfulness:.4f}  (n={len(valid_scores)})")
    print(f"Judge model: {effective_judge}")
    print()
    print("By source:")
    for src, stat in by_source.items():
        print(f"  {src:<20} {stat['mean']:.4f}  (n={stat['n']})")
    print()
    print("By construction:")
    for ctype, stat in by_construction.items():
        print(f"  {ctype:<25} {stat['mean']:.4f}  (n={stat['n']})")
    print()
    print(f"Results: {_SCORES_PATH}")
    print(f"{'=' * 60}\n")


app = typer.Typer(add_completion=False)


@app.command()
def generate() -> None:
    """Generate 30 semi-synthetic QA candidates from the knowledge base."""
    _run_generate()


@app.command()
def score() -> None:
    """Score the reviewed QA set with RAGAS Faithfulness using the RAG chatbot."""
    _run_score()


if __name__ == "__main__":
    app()
