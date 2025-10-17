# tests/test_ocr_agent_smoke.py
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import importlib
import os
import pytest

from app.agents.ocr_agent import OCRAgent

def test_ocr_agent():
    p1 = Path("./assets/image-1.png")
    p2 = Path("./assets/image-2.png")
    if not (p1.exists() and p2.exists()):
        pytest.skip()

    agent = OCRAgent()
    for i in [p1, p2]:
        with i.open("rb") as f:
            out = agent.run(BytesIO(f.read()))

        print(out)
        assert isinstance(out, str)
        assert out.strip(), "模型返回为空字符串"
