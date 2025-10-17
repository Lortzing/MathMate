from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, Optional

from openai import OpenAI

from .protocols import OCRAgentProtocol
from app.config import Config

API_KEY = Config.OCR_API_KEY
BASE_URL = Config.OCR_BASE_URL
MODEL = Config.OCR_MODEL

# 可选：从配置拿自定义提示词/extra_body；否则使用默认
DEFAULT_OCR_PROMPT = (
        "You are a vision-language OCR expert. Extract the main body text of the page "
        "as clean Markdown. Preserve headings, lists, tables when obvious; keep math "
        "as inline $...$ or block $$...$$. Do NOT output code fences. "
        "Reply in the original language found on the page."
    )


class OCRAgent(OCRAgentProtocol):
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

    def _guess_mime(self, raw: bytes) -> str:
        """最常见的三种：PNG/JPEG/GIF；否则回退 PNG。"""
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        return "image/png"

    def _bytesio_to_data_url(self, stream: BytesIO, mime: Optional[str] = None) -> str:
        """将 BytesIO 编码为 data: URL（base64）。不会改变外部的 stream 游标位置。"""
        pos = stream.tell()
        try:
            stream.seek(0)
            raw = stream.read()
        finally:
            stream.seek(pos)
        mime = mime or self._guess_mime(raw)
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _call_model(
        self,
        messages: list[dict],
        extra_body: dict = {},
    ) -> str | None:
        completion = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_body=extra_body,  # 例如 {"enable_thinking": True, "thinking_budget": 256}
        )
        try:
            result = completion.choices[0].message.content
            return result
        except Exception as e:
            print(completion)
            raise e

    def run(self, images: BytesIO | list[BytesIO]) -> str:
        """
        兼容两种用法：
        1) 传入单个 BytesIO
        2) 传入可迭代的 BytesIO（例如 [BytesIO, BytesIO, ...]）
        返回模型输出的文本（Markdown 或纯文本）。
        """
        if images is None:
            return ""


        if hasattr(images, "read"):  # 单个 BytesIO
            streams: list[BytesIO] = [images]  # type: ignore[assignment]
        else:
            try:
                streams = list(images)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError("OCRAgent.run(images): expected a BytesIO or an iterable of BytesIO")

        if not streams:
            return ""

        content: list[dict[str, Any]] = [{"type": "text", "text": DEFAULT_OCR_PROMPT}]
        for s in streams:
            data_url = self._bytesio_to_data_url(s)
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = [{"role": "user", "content": content}]

        out = self._call_model(messages, extra_body={})
        return out or ""
