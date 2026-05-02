"""Application settings loaded from environment.

LangSmith and W&B are dev-only — they should remain unset in CI/prod runs to
avoid hitting free-tier quotas (LangSmith: 5k traces/month, W&B: public projects
only). Both keys are Optional and downstream callers must no-op when missing.

Model IDs are defaults; override via env if Groq deprecates a pin (see
PROJECT_SPEC.md §13 risk register).
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Required: Groq API ---
    groq_api_key: SecretStr

    # --- Optional: Observability (DEV-ONLY — must be unset in CI) ---
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "rogue-llm"
    wandb_api_key: SecretStr | None = None
    wandb_project: str = "rogue-llm"

    # --- Pinned model IDs (Groq free tier; see CLAUDE.md "Model IDs") ---
    target_model: str = "llama-3.1-8b-instant"
    mutator_model: str = "llama-3.3-70b-versatile"
    judge_model: str = "qwen/qwen3-32b"
    cross_validator_model: str = "openai/gpt-oss-120b"
    safety_model: str = "openai/gpt-oss-safeguard-20b"

    # --- Embeddings (local, MPS-accelerated on M4) ---
    embedding_model: str = "all-MiniLM-L6-v2"


def get_settings() -> Settings:
    """Return a fresh Settings instance. Cheap; pydantic-settings caches env reads."""
    return Settings()
