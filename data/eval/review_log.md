# baseline_qa.jsonl — Review Log (v2, post spot-check)

## Summary

| Metric | Count |
|---|---|
| Original synthetic candidates | 30 |
| Accepted unchanged | 17 |
| Edited and accepted | 2 |
| Rejected | 11 |
| Hand-written added | 21 |
| Hand-written corrected after spot-check | 4 |
| **Final total** | **40** |

## Distribution

| Source | Count |
|---|---|
| nvd | 15 |
| owasp_llm | 15 |
| owasp_web | 10 |

| Construction | Count |
|---|---|
| semi-synthetic | 17 |
| semi-synthetic-edited | 2 |
| hand-written | 21 |

## Reject reasons (11 of 30)

**Cluster A — "What is the ID of..." metadata questions (8 of 11):**
The synthetic generator (llama-3.3-70b) repeatedly produced questions of the form
"What is the ID of the security issue related to X" where the ground truth was
just an OWASP identifier. These test whether the model can echo a metadata field,
not whether the RAG retrieves and reasons. Faithfulness scores on these would be
near-trivially high and would not measure pipeline quality.

Affected IDs: semi-014, semi-020, semi-021, semi-023, semi-025, semi-027, semi-028, semi-029.

**Cluster B — Pure version-range metadata (2 of 11):**
semi-005 and semi-008 asked only "What is the version range affected by X" — the
answer is a version string. Extracts from the source but does not test understanding.

**Cluster C — Generic "what does this section contain" (1 of 11):**
semi-015 was so generic any chunk could plausibly answer it.

## Edited entries (2 of 30)

- **semi-007** (Vinchin default credentials): Stripped meta-commentary
  ("as noted in the security vulnerability documented under CVE-2024-22902")
  from ground truth — phrase is not in source CVE, would inflate faithfulness.
- **semi-016** (LLM05 vulnerabilities): Tightened concatenated 3-vulnerability
  ground truth to focus on indirect prompt injection specifically.

## Spot-check fixes (4 of 21 hand-written)

Coverage analysis against actual OWASP source content found 4 issues:

- **hand-010 (LLM04 Data and Model Poisoning)**: `reference_doc_id` referenced
  `LLM04_DataAndModelPoisoning` but the actual filename is `LLM04_DataModelPoisoning`
  (no "And"). Fixed — would have failed retrieval at scoring time.
- **hand-009 (LLM03 Supply Chain)**: ground truth used vocabulary not in source
  ("modifications", "republishing", "impersonations"). Rewrote to use source's
  actual phrasing ("tampering", "fake version", "WizardLM", "backdoors").
  Coverage: 56% → 94%.
- **hand-011 (LLM05 Output Handling)**: ground truth used phrasing absent from
  source ("attacker-influenced", "rendering"). Rewrote with source phrasing
  ("passed downstream", "XSS and CSRF in web browsers", "SSRF, privilege
  escalation, or remote code execution"). Coverage: 58% → 85%.
- **hand-017 (A04 Insecure Design)**: rewrote to use source's exact term
  "implementation defects" instead of "implementation flaws". Coverage: 52% → 89%.

## Verification results

All 40 entries verified against source content. NVD entries: 71%–100% keyword
coverage. OWASP entries: 65%–100% keyword coverage (one entry, hand-019 / A10
SSRF, could not be locally verified — A10 markdown was not available during
review but is expected in the repo's `data/knowledge_base/owasp_web_top10/`).

## Signal worth tracking for Phase 2

The 30% reject rate on OWASP Web entries (7 of 10) is a real signal about how
this synthetic generator behaves on bullet-list-style markdown sources. When
generating attack prompts in Phase 2 with the same generator, the prompt should
explicitly prohibit "What is the ID of..." or "What is the identifier..."
patterns and require questions/inputs that test reasoning over the document body.
