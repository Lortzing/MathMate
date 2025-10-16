from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TemplateMathAgent:
    """Template math agent that can be swapped out for a real LLM."""

    def explain(
        self,
        question: str,
        video_ctx: Dict[str, Any],
        textbook_ctx: Dict[str, Any],
        ocr_text: str,
    ) -> str:
        bullets = []
        if ocr_text:
            bullets.append(f"- 我从图片中读到了（OCR）：{ocr_text[:160]}...")
        if video_ctx.get("hits"):
            topv = video_ctx["hits"][0]
            bullets.append(
                f"- 推荐优先看视频：{topv.get('title')}（相关性较高，得分 {topv.get('score')}）"
            )
        if textbook_ctx.get("hits"):
            topb = textbook_ctx["hits"][0]
            bullets.append(
                "- 教材参考："
                f"{topb.get('book')} · {topb.get('section')} · p.{topb.get('page')}"
            )
        bullets.append(
            "- 解题思路：先检查是否可代换 u=ax+b 简化结构；若存在乘积项，可考虑分部积分。"
        )
        summary = "\n".join(bullets) if bullets else "- 暂无上下文建议，建议先复习相关概念。"
        return f"你的问题：{question}\n\n建议：\n{summary}"
