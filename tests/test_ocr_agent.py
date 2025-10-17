# tests/test_ocr_agent_smoke.py
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import importlib
import os
import pytest

from app.agents.ocr_agent import OCRAgent

@pytest.mark.timeout(120)  # 防止接口偶发卡住
def test_ocr_agent_on_two_pngs():
    p1 = Path("./assets/simple1.png")
    p2 = Path("./assets/simple2.png")
    if not (p1.exists() and p2.exists()):
        pytest.skip("simple1.png / simple2.png 未找到，跳过集成测试")

    agent = OCRAgent()

    with p1.open("rb") as f1, p2.open("rb") as f2:
        out = agent.run([BytesIO(f1.read()), BytesIO(f2.read())])

    assert isinstance(out, str)
    assert out.strip(), "模型返回为空字符串"
