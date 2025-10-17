from __future__ import annotations

from dataclasses import dataclass
from typing import List


class SimpleOCR:
    """Placeholder OCR agent.

    Replace ``run`` with a call to your OCR service (e.g. Tesseract,
    PaddleOCR, or a multimodal model) when integrating for production use.
    """

    def run(self, images: List[bytes]) -> str:
        if not images:
            return ""
        total = sum(len(b) for b in images)
        return (
            "OCR_TEXT(len="
            f"{total}"
            ")  # Replace with real OCR text."
        )
