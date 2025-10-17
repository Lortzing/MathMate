"""Helper functions for building the default LangGraph runtime."""
from __future__ import annotations

from app.agents import MathAgent, MockVideoRAG, OCRAgent, TextbookRAG

from . import RootGraphDeps, build_root_graph


def create_default_deps() -> RootGraphDeps:
    """Instantiate the default dependencies for the root graph."""
    return RootGraphDeps(
        video_rag=MockVideoRAG(),
        textbook_rag=TextbookRAG(),
        ocr_agent=OCRAgent(),
        math_agent=MathAgent(),
    )


def create_default_graph():
    """Build the compiled LangGraph application used across servers."""
    return build_root_graph(create_default_deps())
