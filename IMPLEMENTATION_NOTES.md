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

## Phase 2 — Synthetic Attack Dataset Generator

**Branch:** `feat/phase-2-attack-generator`
**Status:** In progress

### 2.1 Canary Mutation Findings

**OWASP category enforcement**
The LLM mutator can drift into the wrong OWASP taxonomy or category label even when the prompt
explicitly asks it to preserve `owasp_category`. This is a high-risk dataset integrity bug because
the next nine categories would inherit the same failure mode at scale.

Permanent rule: generated variants must always take `owasp_category` from the seed record in code,
not from model output. The mutator prompt should still instruct the model to copy the field, but
`src/attack_generation/mutator.py::_parse_variants()` is the source of truth and hardcodes the
variant category from `seed["owasp_category"]`. Do not remove or weaken this pattern when changing
the prompt, parser, or full-dataset builder.

**Strategy drift: roleplay jailbreak**
The first LLM01 canary produced roleplay variants that preserved the harmful request but stripped
the fictional framing down to an ordinary request for exploit instructions. The validator correctly
rejected those as not genuine prompt-injection attempts. Fix: add strategy-specific guidance for
`roleplay_jailbreak` so the fictional persona/narrative is preserved as the mechanism that carries
the instruction override.

**Observation: indirect retrieval drift**
One `indirect_via_retrieval` survivor drifted from "instructions embedded in retrieved document" to
"assume the role of a penetration testing instructor." This is not blocking the canary, but the
full run should track whether indirect retrieval also needs strategy-specific mutator guidance.

### Strategy decisions during canary

**LLM01 roleplay de-prioritized**
Drop `roleplay_jailbreak` as a primary LLM01 generation strategy. The rerun showed only 1/3
roleplay variants survived after the strategy guidance fix, which points to a structural mismatch
rather than a prompt-tuning issue: the mutator collapses mechanism-implicit strategies, and the
validator's category-coherence threshold requires an explicit prompt-injection mechanism.

Keep `LLM01-0013`, the surviving Riven variant, because it is a real benchmark attack. Drop the two
rejected roleplay variants permanently and do not regenerate them.

**Strict prompt-injection definition**
For LLM01, use the strict definition: the attack must make the instruction-override mechanism
explicit. The broader interpretation, where fictional framing alone counts as prompt injection, is
not the chosen project definition. Roleplay attacks that elicit harmful content without explicit
instruction override belong in LLM05 (Improper Output Handling) or LLM09 (Misinformation), not
LLM01.

Implication: when LLM05 and LLM09 seeds are written later, mechanism-implicit roleplay framings are
in scope there.

**LLM01 final distribution target**
Target 15 final LLM01 seeds with this strategy distribution:
- `direct_override`: 4 total (1 canary + 3 new)
- `indirect_via_retrieval`: 4 total (1 canary + 3 new)
- `payload_splitting`: 3 total (1 canary + 2 new)
- `multilingual_obfuscated`: 3 total (1 canary + 2 new)
- `roleplay_jailbreak`: 1 total (canary only, no more)

Keep the `_STRATEGY_GUIDANCE` mechanism in `mutator.py`. It may be needed for other strategies and
categories, including mechanism-implicit cases such as possible LLM06 excessive-agency seeds. The
`roleplay_jailbreak` guidance can remain for future use if roleplay seeds are explicitly re-added.

### LLM01 expansion and validator decisions

The 10-seed LLM01 expansion run used the `dataset-expander` skill and mutated only seeds
`LLM01-seed-006` through `LLM01-seed-015`. Do not mutate LLM02+ content until explicitly instructed.

Seed `LLM01-seed-008` (output-prefix priming) had 1/3 survival. Keep the surviving variant and do
not iterate further on this seed; the two rejects collapsed into direct harmful-content requests
without a clear prompt-injection override mechanism.

**Validator fail-open bug**
The LLM validator previously failed open: after three validator errors or JSON parse failures,
`_validate_one()` returned `is_valid=True` with reasoning beginning `validator unavailable`. This is
wrong for a quality filter because infrastructure failures silently admit unvalidated variants.

Fix: the validator now fails closed. After exhausted retries, the variant is rejected with
`rejection_category="validator_infrastructure_failure"` and a reason beginning
`validator_json_parse_failure`. Content-level validator decisions are separately labeled
`validator_content_rejection`, so rejected JSONL output distinguishes bad variants from broken
tooling.

**Validator token budget**
The validator response contract is small: JSON with `is_valid` and `reasoning`. The failing
`LLM01-0031` run hit the deterministic Groq error `max completion tokens reached before generating a
valid document` on all three retries. Retrying unchanged could never fix this. The validator
completion cap was `max_tokens=256`; it has been raised to `512` to give the JSON-mode validator
enough room while keeping responses bounded.

**Fail-open checkpoint audit**
Audit of `attacks/v1/checkpoints/LLM01.jsonl` found one existing fail-open acceptance:
`LLM01-0031` (`LLM01-seed-013`, `payload_splitting`). Manual review decision: keep it. The decoded
payload is a clear system-prompt-disclosure instruction, and the failure was validator
infrastructure, not attack quality.

### LLM02 expansion findings

The LLM02 Sensitive Information Disclosure expansion used 9 seeds and produced 24 accepted
survivors in `attacks/v1/checkpoints/LLM02.jsonl`. Results were accepted as-is.

Seed `LLM02-seed-006` (repetition attack for training-data leakage) had 0/3 survival for a
structural validator reason: the current validator scores prompts, not model responses, while
repetition attacks are only visible as a vulnerability when the response degenerates or regurgitates
memorized content. Do not iterate on this seed during Phase 2. Revisit this in Phase 4 as a known
validator limitation when response-aware evaluation is available.

### LLM03 expansion findings

The LLM03 Supply Chain expansion used 3 seeds and produced 6 accepted survivors in
`attacks/v1/checkpoints/LLM03.jsonl`. Results were accepted as-is.

`model_identity_probing` had 1/3 survival. This may indicate validator over-strictness at the LLM03
category boundary because model identity and supply-chain metadata disclosure can be relevant setup
for supply-chain targeting. Do not act now; revisit if this pattern repeats in LLM04+.

Two near-duplicate mutator outputs reached the validator for `memorized_poisoned_artifact_probing`.
This suggests the MinHash threshold of 0.85 may be too lenient for short, narrow prompts. Do not
change it yet; revisit after 3+ categories show the same pattern.

### LLM04 expansion findings

The LLM04 Data and Model Poisoning expansion used 5 seeds and produced 9 accepted survivors in
`attacks/v1/checkpoints/LLM04.jsonl` across 3 retained strategies:
`backdoor_trigger_probing`, `memorized_poisoned_content`, and
`poisoning_technique_solicitation`.

Taxonomy decision: drop `biased_output_elicitation` from LLM04 and do not re-attempt it. Reframe
`embedding_poisoning_via_query` for LLM08 later, where embedding and retrieval-layer behavior is
the natural category boundary.
