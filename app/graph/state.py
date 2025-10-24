from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class RootState(TypedDict, total=False):
    """Global state shared across the LangGraph pipeline."""

    # Inputs
    user_query: str
    images: List[bytes]

    # Intermediate values
    ocr_text: str
    combined_query: str
    video: Dict[str, Any]
    textbook: Dict[str, Any]
    math_explanation: str

    # Outputs
    reply_to_user: str
    finalize: dict

    # Supervisor decision (latest action emitted by the root agent)
    decision: Dict[str, Any]
    _stream: bool