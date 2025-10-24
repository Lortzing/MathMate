from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

from openai import OpenAI

from app.config import Config
from app.utils import get_logger

from .state import RootState


logger = get_logger("root-supervisor")

NextStep = Literal[
    "ocr",
    "build_query",
    "search_both",
    "math",
    "finalize",
]


@dataclass
class Action:
    """Structured decision produced by the root agent."""

    next: NextStep
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class RootSupervisor:
    """Root agent that decides the next step via an LLM call."""

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=Config.ROOT_API_KEY,
            base_url=Config.ROOT_BASE_URL,
        )

    # ------------------------------- public API -------------------------------
    def decide(self, state: RootState) -> Action:
        """Invoke the LLM to decide the next step for the workflow."""

        try:
            action = self._call_model(state)
            logger.debug("Root supervisor LLM decision: %s", action)
            return action
        except Exception as exc:  # pragma: no cover - safety fallback
            logger.warning(
                "Root supervisor LLM failed (%s). Falling back to heuristics.",
                exc,
            )
            return self._fallback_decision(state)

    # ------------------------------ internal helpers -----------------------------
    def _call_model(self, state: RootState) -> Action:
        sanitized_state = self._summarize_state(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the root supervisor of a math multi-agent system. "
                    "You orchestrate OCR, query building, retrieval, and math reasoning. "
                    "Always respond with a JSON object containing keys 'next', 'params', and 'reason'. "
                    f"'next' must be one of {NextStep}. "
                    "Use 'ocr' to transcribe pending images, 'build_query' to combine user input with OCR, "
                    "'search_both' to call both RAG tools, 'math' to ask the math agent, "
                    "and 'finalize' when ready to produce the final response or nothing to do but generate a simple reply. "
                    "Place any tool-specific options inside 'params'. When no options are needed return an empty object."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Current shared state (JSON):\n" + json.dumps(sanitized_state, ensure_ascii=False)
                ),
            },
        ]

        completion = self.client.chat.completions.create(
            model=Config.ROOT_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        next_step = data.get("next", "finalize")
        params = data.get("params") or {}
        reason = data.get("reason") or ""
        print(data)

        if next_step not in {"ocr", "build_query", "search_both", "math", "finalize"}:
            logger.warning("Invalid next step from LLM (%s). Falling back to finalize.", next_step)
            next_step = "finalize"

        if not isinstance(params, dict):
            logger.warning("LLM params payload is not a dict: %s", params)
            params = {}

        return Action(next=next_step, params=params, reason=str(reason))
    
    def _finalize(self, state: RootState) -> str:
        sanitized_state = self._summarize_state(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the agent of a math multi-agent system at the FINALIZE stage. "
                    "Your task is to produce the final message that will be shown to the user.\n\n"
                    "Use the provided shared state (which may contain OCR text, retrieval results, "
                    "and math reasoning) to compose a clear, accurate answer. "
                    "Do not include any JSON, metadata, or technical notes — only return the natural-language reply.\n\n"
                    "Guidelines:\n"
                    "1. If math reasoning is available, check it carefully and reply to the user with correct steps and results. "
                    "Accuracy is essential, but clarity and structure also matter.\n"
                    "2. If textbook or video retrieval results exist, synthesize them concisely "
                    "(for example: 'According to the textbook...' ).\n"
                    "3. If OCR text contributed to reasoning, integrate it naturally.\n"
                    "4. Keep the tone explanatory and user-facing, suitable for math learning.\n"
                    "5. Respond in the user's language (usually Chinese if the input was Chinese).\n"
                    "6. Output only the final answer text — no JSON, no lists of actions.\n"
                    "7. Your output must be **directly renderable as Markdown**: do not escape newlines, "
                    "and do not use double backslashes (\\\\) unless they are part of LaTeX math syntax. "
                    "Use real line breaks and Markdown section headings (###, **bold**, etc.) when needed.\n"
                    "8. Never include visible escape sequences like '\\n' or '\\t'. Write actual newlines instead.\n"
                    "9. When writing LaTeX, use single backslashes for commands (e.g., '\\int', '\\frac', '\\ln'), "
                    "and wrap expressions properly in $...$ or $$...$$. "
                    "Do not add redundant escapes or JSON-safe quoting — the response should be clean and readable."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Current shared state (JSON):\n" + json.dumps(sanitized_state, ensure_ascii=False)
                ),
            },
        ]

        completion = self.client.chat.completions.create(
            model=Config.ROOT_MODEL,
            messages=messages,
        )
        content = completion.choices[0].message.content
        return content if content else ""

    def _summarize_state(self, state: RootState) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, value in state.items():
            if key == "images":
                summary[key] = f"{len(value)} image(s)" if isinstance(value, list) else "unknown"
            elif key in {"video", "textbook"}:
                if isinstance(value, dict) and value.get("hits"):
                    summary[key] = {
                        "hits": len(value.get("hits", [])),
                        "error": value.get("error"),
                    }
                else:
                    summary[key] = "empty"
            elif key == "decision":
                # Avoid feeding the previous decision back to the LLM verbatim
                continue
            else:
                summary[key] = value
        return summary

    def _fallback_decision(self, state: RootState) -> Action:
        user_q = (state.get("user_query") or "").strip()
        has_images = bool(state.get("images"))
        has_ocr = bool((state.get("ocr_text") or "").strip())
        combined = (state.get("combined_query") or "").strip()
        has_video = bool(state.get("video"))
        has_textbook = bool(state.get("textbook"))
        has_math = bool((state.get("math_explanation") or "").strip())

        if has_images and not has_ocr:
            return Action(next="ocr", reason="OCR needed for provided images")
        if user_q and not combined:
            return Action(next="build_query", reason="Need combined query before retrieval")
        if combined and (not has_video or not has_textbook):
            return Action(next="search_both", reason="Fetch retrieval contexts")
        if combined and not has_math:
            return Action(next="math", reason="Generate math explanation")
        return Action(next="finalize", reason="All steps completed or nothing to do")