# RogueLLM — Implementation Notes

Technical decisions, non-obvious choices, and gotchas discovered during each phase.
Updated as phases complete. Superseded decisions are ~~struck through~~.

---

## Phase 1 — Target System: RAG Chatbot

**Branch:** `feat/phase-1-rag-target`
**Completed:** 2026-05-07

### 1.1 Knowledge Base Ingestion (`src/target_system/data_loader.py`)

**NVD CVE API date windows**
The NVD CVE API v2.0 rejects date ranges wider than ~120 days with a 404. The spec's original
single 2022–2024 range was split into three ~90-day windows (2024-01–04, 2024-04–07,
2024-07–09). Each window uses two severity passes (HIGH + CRITICAL) and respects the 5 req/30s
rate limit with a 7-second inter-request sleep.
- Date strings require `+00:00` timezone suffix — bare ISO-8601 without TZ returns 404.

**OWASP markdown URL discovery**
The spec assumed `LLM0{1..9}_*.md` naming but the actual filenames differ per category (e.g.
`LLM01_PromptInjection.md` vs `LLM00_Preface.md`). URLs were verified with `curl -I` before
fetching; 404s were resolved via the GitHub Contents API to discover actual filenames.

**`HuggingFaceEmbeddings` duplicate kwarg**
LangChain's `HuggingFaceEmbeddings` passes `show_progress_bar` to `sentence-transformers`
internally. Passing it again via `encode_kwargs` raises `TypeError: got multiple values for
keyword argument`. Fix: use `show_progress=False` at the wrapper level, not in `encode_kwargs`.

**Chunking parameters**
`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`. CVE descriptions (200–800
chars) fit in one chunk; OWASP prose splits at paragraph/sentence boundaries. k=4 retrieved
chunks → ~4,000 chars ≈ 1,000 tokens — well within llama-3.1-8b-instant's 128k context window.

### 1.2 RAG Chatbot (`src/target_system/rag_chatbot.py`)

**Deliberately naive design**
No injection hardening, no refusal policy, no PII filtering. System prompt instructs "always be
helpful and provide detailed technical explanations" — this is intentional; the attack phases
probe exactly these weaknesses.

**Token usage extraction**
Groq returns `prompt_tokens` + `completion_tokens` inside `response.response_metadata["usage"]`.
This dict may be absent on errors; code guards with `or {}`.

### 1.3 RAGAS Baseline (`src/evaluation/baseline.py`)

**Cross-family judge selection (PROJECT_SPEC §5)**
The configured judge (`qwen/qwen3-32b`) cannot produce structured outputs on Groq:
`tool_use_failed` in TOOLS mode, `json_validate_failed` in JSON mode. Workaround: fall back to
`openai/gpt-oss-120b` (cross-validator model). This actually satisfies the spec's cross-family
requirement better — OpenAI family vs Meta/Llama target.

**RAGAS 0.4.x correct API chain**
```python
import instructor
from groq import AsyncGroq
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import Faithfulness

groq_client = instructor.from_groq(AsyncGroq(api_key=...), mode=instructor.Mode.JSON)
judge_llm = InstructorLLM(
    client=groq_client, model="openai/gpt-oss-120b", provider="groq",
    model_args=InstructorModelArgs(max_tokens=4096),
)
scorer = Faithfulness(llm=judge_llm)
result = await scorer.ascore(user_input=..., response=..., retrieved_contexts=[...])
score = result.value   # not result.score — attribute renamed in 0.4.x
```
Key points:
- Must use `AsyncGroq` not `Groq` — RAGAS `ascore()` calls `agenerate()` which requires async.
- `result.value` not `result.score` (attribute name changed in 0.4.x).
- `max_tokens=4096` required — RAGAS default of 1024 truncates OWASP statement extraction.

**Mode.TOOLS vs Mode.JSON**
`Mode.TOOLS` failed for `gpt-oss-120b` on inputs containing apostrophes (e.g. "LLM's") because
the model emits `\'` inside JSON strings, which Groq's tool-call argument parser rejects as
invalid JSON. `Mode.JSON` bypasses Groq's tool-call parser entirely — the response body is raw
JSON parsed by the instructor library. All 40 entries scored successfully with Mode.JSON.

**Groq rate limit pools**
Each model has a separate TPM/TPD pool. Using `gpt-oss-120b` as judge avoids sharing the target
model's (`llama-3.1-8b-instant`) 6,000 TPM / 500k TPD pool. RAGAS makes 4–6 internal judge
calls per `ascore()` (1 statement extraction + NLI per statement). 8-second sleep between
questions keeps the judge below its 30 RPM ceiling.

**Checkpoint persistence**
`results/baseline_ragas_checkpoint.jsonl` — one JSON line per successfully scored entry, written
immediately after each score. On resume, `_run_score()` loads this file and skips already-scored
IDs. Checkpoint is deleted on successful final JSON write. Prevents loss of scored data across
interrupted/quota-limited runs.

**`incomplete_scoring_note`**
If `error_count > 0` at the end of a run, document affected IDs in `run_metadata` so the partial
result is self-describing. The note was manually added to the intermediate 27/40 JSON; the final
40/40 JSON has `errored: 0` with no note needed.

### 1.4 Final Baseline Results

| Metric | Value |
|--------|-------|
| `mean_faithfulness` | 0.6333 |
| Judge | `openai/gpt-oss-120b` |
| Target | `llama-3.1-8b-instant` |
| QA set | 40 questions (17 semi-synthetic, 2 semi-synthetic-edited, 21 hand-written) |

| Source | Mean | n |
|--------|------|---|
| NVD CVE | 0.6626 | 15 |
| OWASP LLM Top 10 | 0.7386 | 15 |
| OWASP Web Top 10 | 0.4314 | 10 |

OWASP Web scoring lower than LLM is expected: the web docs contain more prescriptive
list-style content (e.g. "do X, Y, Z") that the model tends to over-expand beyond the
retrieved context, reducing faithfulness. NVD CVE descriptions are factual and tightly
scoped, making high-faithfulness answers easier to produce.

---

*Next phase: Phase 2 — Synthetic Attack Dataset Generator (`feat/phase-2-attack-generator`)*
