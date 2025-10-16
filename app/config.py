from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Runtime configuration for the multi-agent system."""

    openai_api_key: str | None = None
    qwen_api_key: str | None = None
    rag_top_k: int = 3


def get_settings() -> Settings:
    """Return default settings; expand as needed for production."""

    return Settings()
