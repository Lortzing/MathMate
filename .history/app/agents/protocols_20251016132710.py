from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class VideoRAG(Protocol):
    """Protocol for video retrieval-augmented generation tools."""

    def search(self, query: str) -> Dict[str, Any]:
        """Search for video resources relevant to the query."""


@runtime_checkable
class TextbookRAG(Protocol):
    """Protocol for textbook retrieval-augmented generation tools."""

    def search(self, query: str) -> str:
        """Search for textbook resources relevant to the query."""


@runtime_checkable
class OCRAgent(Protocol):
    """Protocol for OCR agents used to transcribe images to text."""

    def run(self, images: List[bytes]) -> str:
        """Return textual transcription for the provided image bytes."""


@runtime_checkable
class MathAgent(Protocol):
    """Protocol for math reasoning agents."""

    def explain(
        self,
        question: str,
        video_ctx: Dict[str, Any],
        textbook_ctx: Dict[str, Any],
        ocr_text: str,
    ) -> str:
        """Return a detailed explanation for the user's question."""
