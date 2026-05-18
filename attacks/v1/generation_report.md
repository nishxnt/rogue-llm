# Phase 2 Synthetic Attack Generation Report

Generation date: 2026-05-14

Phase 2 produced 175 benchmark attacks across all 10 OWASP LLM Top 10 2025 categories. The original planning target was 200 attacks, but generation stopped at 175 because the final quality gate favored category-coherent, validator-confirmed variants over filling a fixed quota with weak or taxonomy-misaligned prompts.

## Final Dataset Composition

| Category   | Name                              | Accepted | Generated | Pass rate |
| ---------- | --------------------------------- | -------- | --------- | --------- |
| LLM01:2025 | Prompt Injection                  | 38       | 43        | 88.4%     |
| LLM02:2025 | Sensitive Information Disclosure  | 24       | 27        | 88.9%     |
| LLM03:2025 | Supply Chain                      | 6        | 9         | 66.7%     |
| LLM04:2025 | Data and Model Poisoning          | 9        | 27        | 33.3%     |
| LLM05:2025 | Improper Output Handling          | 20       | 21        | 95.2%     |
| LLM06:2025 | Excessive Agency                  | 12       | 12        | 100.0%    |
| LLM07:2025 | System Prompt Leakage             | 19       | 27        | 70.4%     |
| LLM08:2025 | Vector and Embedding Weaknesses   | 11       | 11        | 100.0%    |
| LLM09:2025 | Misinformation and Disinformation | 19       | 36        | 52.8%     |
| LLM10:2025 | Unbounded Consumption             | 17       | 21        | 81.0%     |

## Strategy Distribution

| Strategy                              | Category   | Count |
| ------------------------------------- | ---------- | ----- |
| adversarial_suffix                    | LLM01:2025 | 3     |
| agentic_self_authorization            | LLM06:2025 | 3     |
| api_side_channel_timing               | LLM10:2025 | 3     |
| backdoor_trigger_probing              | LLM04:2025 | 3     |
| business_secret_extraction            | LLM02:2025 | 3     |
| code_generation_with_insecure_pattern | LLM05:2025 | 2     |
| completion_continuation               | LLM07:2025 | 3     |
| compromised_source_recommendation     | LLM03:2025 | 3     |
| contested_topic_with_false_certainty  | LLM09:2025 | 2     |
| context_format_inference              | LLM07:2025 | 1     |
| continuous_input_overflow             | LLM10:2025 | 3     |
| credential_extraction                 | LLM02:2025 | 6     |
| denial_of_wallet                      | LLM10:2025 | 3     |
| direct_override                       | LLM01:2025 | 10    |
| embedded_secret_extraction            | LLM07:2025 | 3     |
| embedding_inversion                   | LLM08:2025 | 2     |
| embedding_poisoning                   | LLM08:2025 | 4     |
| excessive_permission_recommendation   | LLM06:2025 | 3     |
| fabricated_citation                   | LLM09:2025 | 4     |
| fabricated_factual_aggregation        | LLM09:2025 | 1     |
| false_premise_acceptance              | LLM09:2025 | 5     |
| hashed_or_encoded_request             | LLM07:2025 | 3     |
| human_approval_bypass                 | LLM06:2025 | 3     |
| imagined_tool_invocation              | LLM06:2025 | 3     |
| indirect_pii_inference                | LLM02:2025 | 3     |
| indirect_via_retrieval                | LLM01:2025 | 9     |
| memorized_poisoned_artifact_probing   | LLM03:2025 | 2     |
| memorized_poisoned_content            | LLM04:2025 | 3     |
| meta_instruction_query                | LLM07:2025 | 3     |
| misrepresented_expertise              | LLM09:2025 | 3     |
| model_identity_probing                | LLM03:2025 | 1     |
| multilingual_obfuscated               | LLM01:2025 | 6     |
| output_amplification                  | LLM10:2025 | 3     |
| paraphrase_inference                  | LLM07:2025 | 3     |
| path_traversal_in_output              | LLM05:2025 | 3     |
| payload_splitting                     | LLM01:2025 | 9     |
| pii_extraction                        | LLM02:2025 | 6     |
| poisoning_technique_solicitation      | LLM04:2025 | 3     |
| resource_intensive_query              | LLM10:2025 | 3     |
| retrieval_boundary_probing            | LLM08:2025 | 2     |
| roleplay_jailbreak                    | LLM01:2025 | 1     |
| shell_command_injection               | LLM05:2025 | 3     |
| similarity_collision                  | LLM08:2025 | 3     |
| sql_injection_payload                 | LLM05:2025 | 3     |
| ssrf_via_url_generation               | LLM05:2025 | 3     |
| system_internal_disclosure            | LLM02:2025 | 3     |
| training_data_regurgitation           | LLM02:2025 | 3     |
| unsafe_code_with_confident_framing    | LLM09:2025 | 4     |
| variable_length_input_flood           | LLM10:2025 | 2     |
| verbatim_extraction                   | LLM07:2025 | 3     |
| xss_payload_in_response               | LLM05:2025 | 6     |

## Filter Pass Rates

| Stage                | Count | Stage pass rate | Notes                                                                   |
| -------------------- | ----- | --------------- | ----------------------------------------------------------------------- |
| Generated            | 234   | 100.0%          | Raw accepted plus rejected variants recorded during category runs.      |
| After length filter  | 234   | 100.0%          | No recorded length rejections in final category runs.                   |
| After metadata regex | 234   | 100.0%          | Category values were copied from seed records in code.                  |
| After MinHash dedup  | 234   | 100.0%          | Dedup threshold 0.85.                                                   |
| Final accepted       | 175   | 74.8%           | Includes 11 LLM08 entries where prompt validator was skipped by design. |

Across prompt-mutated categories, the final validator accepted 164 of 223 evaluated variants (73.5%). LLM08 contributes 11 structured vector/retrieval-layer entries whose prompt validator stage was skipped because the attack mechanism lives in retrieval metadata and poisoned document content rather than a standalone prompt.

## Key Findings

Cross-family validation was effective as a category-coherence gate. Using `openai/gpt-oss-120b` as the validator caught taxonomy drift, weak prompts, and variants whose attack mechanism disappeared during mutation. The validator also made failure modes visible enough to record as engineering decisions instead of silently accepting filler.

The mutator collapsed on mechanism-implicit strategies unless given explicit guidance. The clearest example was `roleplay_jailbreak`, where variants initially retained harmful subject matter but dropped the fictional framing that made the prompt injection strategy coherent. The `_STRATEGY_GUIDANCE` dict now preserves strategy-specific mechanisms for roleplay, false-premise misinformation, and unsafe-code misinformation.

Prompt-only validation has known blind spots. LLM02 repetition attacks, LLM09 hallucination and package-recommendation probes, and LLM10 model-extraction probing are response-judged or behavior-judged. They may look benign as single prompts even though their benchmark value appears in the model's response, repeated API use, output length, retrieval behavior, or timing differences.

Taxonomy decisions improved dataset quality. `biased_output_elicitation` was dropped from LLM04 rather than forced into Data and Model Poisoning. `embedding_poisoning_via_query` moved to LLM08, where vector and retrieval-layer behavior is the right boundary. LLM07 inference-style prompts were accepted narrowly where they exposed embedded sensitive data or control logic, and adjacent system-behavior inference remains a future taxonomy question.

## Phase 4 Implications

The evaluation engine needs more than prompt-category judging. Phase 4 should score model responses for disclosure, hallucination, unsupported factual claims, unsafe-code confidence, and repetition-induced memorization attempts. It should also support behavior-aware checks for unbounded consumption, including output length, repeated retrieval/tool-call amplification, latency or side-channel probes, and model-extraction-style query series.

AttackRunner should consume the final `attacks/v1/dataset.jsonl` as the canonical Phase 2 artifact, while preserving category-specific checkpoint provenance for debugging. LLM08 entries need runner support for structured fields such as `target_query`, `poisoned_doc_content`, similarity thresholds, and retrieval metadata.
