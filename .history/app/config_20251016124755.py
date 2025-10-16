from __future__ import annotations

from dataclasses import dataclass
import os

@dataclass
class Settings:
    """Runtime configuration for the multi-agent system."""
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    qwen_api_key: str | None = os.getenv("OPENAI_API_KEY")
    rag_top_k: int = 3


def get_settings() -> Settings:
    """Return default settings; expand as needed for production."""

    return Settings()
