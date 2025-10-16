from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path

@dataclass
class Settings:
    """Runtime configuration for the multi-agent system."""
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    qwen_api_key: str | None = os.getenv("OPENAI_API_KEY")
    rag_top_k: int = 5
    PAGE_OPEN_RE  = re.compile(r"<page\b[^>]*>", re.IGNORECASE)
    PAGE_CLOSE_RE = re.compile(r"</page\b[^>]*>", re.IGNORECASE)
    HEAD_RE       = re.compile(r"^(#{1,6})\s+(.*)$", re.M)

    # ========== 常量 ==========
    EMBED_MODEL = "text-embedding-v4"
    EMBED_DIM = 1024
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBED_API_KEY = os.getenv("OPENAI_API_KEY")

    INDEX_PATH = Path("./faiss_index.index")
    META_PATH = Path("./faiss_meta.json")


def get_settings() -> Settings:
    """Return default settings; expand as needed for production."""

    return Settings()
