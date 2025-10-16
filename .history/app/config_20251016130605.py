from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path

@dataclass
class Config:
    """Runtime configuration for the multi-agent system."""
    MATH_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    ROOT_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    EMBED_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    
    MATH_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ROOT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBED_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    MATH_MODEL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ROOT_MODEL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBED_MODEL = "text-embedding-v4"
    
    
    RAG_TOP_K = 5
    INDEX_PATH = Path("./textbook_embed/faiss_index.index")
    META_PATH = Path("./textbook_embed/faiss_meta.json")
    PAGE_OPEN_RE  = re.compile(r"<page\b[^>]*>", re.IGNORECASE)
    PAGE_CLOSE_RE = re.compile(r"</page\b[^>]*>", re.IGNORECASE)
    HEAD_RE       = re.compile(r"^(#{1,6})\s+(.*)$", re.M)

    EMBED_MODEL = "text-embedding-v4"
    EMBED_DIM = 1024
    
    
    
