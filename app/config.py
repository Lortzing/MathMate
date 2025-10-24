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
    OCR_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    
    MATH_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ROOT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBED_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    OCR_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    MATH_MODEL = "qwen-math-plus"
    ROOT_MODEL = "qwen-plus"
    EMBED_MODEL = "text-embedding-v4"
    OCR_MODEL = "qwen3-vl-plus"
    
    
    RAG_TOP_K = 5
    INDEX_PATH = Path("./textbook_embed/vector/faiss_index.index")
    META_PATH = Path("./textbook_embed/meta/faiss_meta.json")

    EMBED_DIM = 1024
    
    
    
