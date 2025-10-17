from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable, Literal
from io import BytesIO


@runtime_checkable
class VideoRAGProtocol(Protocol):
    """Protocol for video retrieval-augmented generation tools."""

    def search(self, query: str) -> Dict[str, Any]:
        """Search for video resources relevant to the query."""
        ...


@runtime_checkable
class TextbookRAGProtocol(Protocol):
    """Protocol for textbook retrieval-augmented generation tools."""

    def search(self, query: str) -> str:
        """Search for textbook resources relevant to the query."""
        ...


@runtime_checkable
class OCRAgentProtocol(Protocol):
    """Protocol for OCR agents used to transcribe images to text."""

    def run(self, images: BytesIO | list[BytesIO]) -> str:
        """Return textual transcription for the provided image bytes."""
        ...

Instruction = Literal[
        "Solve",            # 1. 解题
        "GenerateProblems", # 2. 根据知识点/问题出题
        "MultiMethods",     # 3. 使用多种方式解题
        "ExplainStep",      # 4. 解释该步骤
        "Heuristic",        # 5. 启发式调用
    ]

@runtime_checkable
class MathAgentProtocol(Protocol):
    """Protocol for math reasoning agents."""   

    def explain(
        self,
        question: str,
        instruction: Instruction,
        prompt: str | None
    ) -> str:
        """Return a detailed explanation for the user's question."""
        ...