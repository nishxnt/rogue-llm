# `risk_report.json` Schema v1.0

This file documents the external Phase 6 reporting contract.

Top-level fields:

- `schema_version`: fixed string, currently `"1.0"`
- `run_metadata`
- `system_risk`
- `per_category_risk`
- `decision_distribution`
- `layer_attribution`
- `residual_analysis`
- `l2_fail_closed_analysis`
- `llm06_weakness_analysis`
- `faithfulness_coverage`
- `cross_validator_summary`
- `token_cost_summary`
- `owasp_web_chunking_finding`
- `honest_findings`
- `methodology_notes`
- `limitations`

Notes:

- The report is built from existing Phase 4 and Phase 5 artifacts only. It must not trigger new
  evaluation work.
- `l2_fail_closed_analysis`, `llm06_weakness_analysis`, `faithfulness_coverage`, and
  `owasp_web_chunking_finding` are first-class sections by design; they are not buried inside
  generic metadata because they capture benchmark-critical findings.
- `system_risk` and `per_category_risk` carry both accounting modes:
  - `with_infrastructure_failures`
  - `without_infrastructure_failures`
- `residual_analysis` is a compact summary over the separate Phase 5 residual artifact. It keeps
  bypass counts, attack IDs, and remediation notes in the main report for downstream tooling.
