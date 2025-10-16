from __future__ import annotations

import base64
from pathlib import Path
from typing import List


def load_image_bytes(paths: List[str]) -> List[bytes]:
    """Load image files into memory as raw bytes."""

    return [Path(path).read_bytes() for path in paths]


def b64_to_bytes(b64: str) -> bytes:
    """Decode base64-encoded image content."""

    return base64.b64decode(b64)


def bytes_to_b64(data: bytes) -> str:
    """Encode raw bytes as a base64 string."""

    return base64.b64encode(data).decode("utf-8")
