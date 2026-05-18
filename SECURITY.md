# Security Policy

## Project Scope

RogueLLM is a research and portfolio project for adversarial testing of Retrieval-Augmented Generation (RAG) systems. It implements attacks aligned to the [OWASP LLM Top 10 (2025)](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) taxonomy for educational and defensive research purposes.

This codebase is NOT intended for production use without additional review. The attack dataset is synthetic and generated for benchmarking the project's own RAG target, not for deployment against third-party systems.

## Reporting a Vulnerability

If you discover a security issue in this codebase:

- **Non-sensitive issues:** Open a GitHub issue with the label `security`.
- **Sensitive disclosures:** Contact the maintainer directly via the email address listed on the GitHub profile.

## Responsible Use

The attack dataset (`attacks/v1/dataset.jsonl`) contains 175 adversarial prompts categorized by OWASP LLM Top 10 (2025). These prompts are designed to test defensive measures in RAG systems built by the user. Using them against systems you do not own or have explicit permission to test may violate applicable laws.

## Acknowledgments

This project's methodology aligns with the OWASP LLM Top 10 (2025) standard. The authors of that standard are the upstream authority on LLM security taxonomy.
