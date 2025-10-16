from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MockTextbookRAG:
    """Mock implementation of the textbook RAG search tool."""

    name: str = "mock-textbook-rag"

    def search(self, query: str) -> Dict[str, Any]:
        """Return demo results. Replace with a real retrieval backend."""
        return {
            "query": query,
            "hits": [
                {
                    "book": "Thomas' Calculus 14e",
                    "section": "7.2 Substitution",
                    "page": 412,
                    "score": 0.82,
                },
                {
                    "book": "Stewart Calculus 8e",
                    "section": "8.1 Integration by Parts",
                    "page": 516,
                    "score": 0.79,
                },
            ],
            "source": self.name,
        }
