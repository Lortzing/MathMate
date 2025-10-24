from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Mapping, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from langgraph.config import get_stream_writer  # 仅当你想在函数内自行获取

from app.agents.protocols import (
    MathAgentProtocol,
    OCRAgentProtocol,
    TextbookRAGProtocol,
    VideoRAGProtocol,
    Instruction,
)
from app.utils import get_logger

from .state import RootState
from .supervisor import RootSupervisor


logger = get_logger("root-graph")


@dataclass
class RootGraphDeps:
    video_rag: VideoRAGProtocol
    textbook_rag: TextbookRAGProtocol
    ocr_agent: OCRAgentProtocol
    math_agent: MathAgentProtocol
    supervisor: RootSupervisor | None = None


_ALLOWED_INSTRUCTIONS: set[str] = {
    "Solve",
    "GenerateProblems",
    "MultiMethods",
    "ExplainStep",
    "Heuristic",
}


def build_root_graph(deps: RootGraphDeps):
    """Construct and compile the root LangGraph workflow driven by the supervisor."""

    supervisor = deps.supervisor or RootSupervisor()
    graph = StateGraph(RootState)

    def _current_params(state: RootState) -> Dict[str, Any]:
        decision = state.get("decision") or {}
        if isinstance(decision, Mapping):
            params = decision.get("params", {})
        else:
            params = {}
        return params if isinstance(params, dict) else {}

    def supervisor_node(state: RootState) -> RootState:
        action = supervisor.decide(state)
        logger.debug("Supervisor decided: %s", action)
        return {
            "decision": {
                "next": action.next,
                "params": action.params,
                "reason": action.reason,
            }
        }

    def ocr(state: RootState) -> RootState:
        params = _current_params(state)
        images = state.get("images", [])
        if not images:
            return {}

        max_images = params.get("max_images")
        try:
            if max_images is not None:
                max_images = max(0, int(max_images))
        except (TypeError, ValueError):
            max_images = None

        selected_images = images[:max_images] if max_images else images
        texts: list[str] = []
        for idx, img in enumerate(selected_images, start=1):
            try:
                text = deps.ocr_agent.run(BytesIO(img))
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("OCR agent failed on image %s: %s", idx, exc)
                text = ""
            if text:
                texts.append(f"Image {idx}:\n{text}")
        if not texts:
            return {}
        return {"ocr_text": "\n\n".join(texts)}

    def build_query(state: RootState) -> RootState:
        params = _current_params(state)
        user_q = (state.get("user_query") or "").strip()
        ocr_t = (state.get("ocr_text") or "").strip()
        include_ocr = params.get("include_ocr", True)
        prefix = params.get("prefix") or ""
        postfix = params.get("postfix") or ""

        sections: list[str] = []
        if prefix:
            sections.append(str(prefix))
        if user_q:
            sections.append(user_q)
        if include_ocr and ocr_t:
            sections.append("[OCR]\n" + ocr_t)
        if postfix:
            sections.append(str(postfix))

        combined = "\n\n".join(section for section in sections if section)
        return {"combined_query": combined or user_q}

    def search_both(state: RootState) -> RootState:
        params = _current_params(state)
        query = (state.get("combined_query") or state.get("user_query") or "").strip()
        if not query:
            return {}

        video_top_k = params.get("video_top_k")
        textbook_top_k = params.get("textbook_top_k")

        try:
            video_result = deps.video_rag.search(query)
        except Exception as exc:  # pragma: no cover - retrieval failure safeguard
            logger.warning("Video RAG search failed: %s", exc)
            video_result = {"query": query, "hits": [], "error": str(exc)}
        else:
            if not isinstance(video_result, dict):
                video_result = {"query": query, "hits": video_result}
        if isinstance(video_result, dict) and video_top_k is not None:
            try:
                k = max(0, int(video_top_k))
                if isinstance(video_result.get("hits"), list):
                    video_result["hits"] = video_result["hits"][:k]
            except (TypeError, ValueError):
                pass

        try:
            textbook_result = deps.textbook_rag.search(query)
        except Exception as exc:  # pragma: no cover - retrieval failure safeguard
            logger.warning("Textbook RAG search failed: %s", exc)
            textbook_result = {"query": query, "content": "", "error": str(exc)}
        else:
            if isinstance(textbook_result, dict):
                pass
            else:
                textbook_result = {"query": query, "content": textbook_result}
        if isinstance(textbook_result, dict) and textbook_top_k is not None:
            try:
                k = max(0, int(textbook_top_k))
                content = textbook_result.get("content")
                if isinstance(content, list):
                    textbook_result["content"] = content[:k]
            except (TypeError, ValueError):
                pass

        return {"video": video_result, "textbook": textbook_result}

    def math_reasoning(state: RootState) -> RootState:
        params = _current_params(state)
        instruction = params.get("instruction", "Solve")
        if instruction not in _ALLOWED_INSTRUCTIONS:
            logger.warning("Invalid instruction '%s', defaulting to Solve", instruction)
            instruction = "Solve"
        prompt_override = params.get("prompt")

        context_chunks: list[str] = []
        if prompt_override:
            context_chunks.append(str(prompt_override))
        video_ctx = state.get("video")
        if video_ctx:
            context_chunks.append("[Video Retrieval]\n" + json.dumps(video_ctx, ensure_ascii=False))
        textbook_ctx = state.get("textbook")
        if textbook_ctx:
            context_chunks.append("[Textbook Retrieval]\n" + json.dumps(textbook_ctx, ensure_ascii=False))
        ocr_text = (state.get("ocr_text") or "").strip()
        if ocr_text:
            context_chunks.append("[OCR]\n" + ocr_text)

        prompt = "\n\n".join(context_chunks) or None
        instruction_value: Instruction = cast(Instruction, instruction)
        try:
            explanation = deps.math_agent.explain(
                question=state.get("user_query", "") or "",
                instruction=instruction_value,
                prompt=prompt,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("Math agent failed: %s", exc)
            explanation = f"（数学讲解生成失败：{exc}）"
        return {"math_explanation": explanation}

    def finalize(state: RootState, writer: StreamWriter | None = None):
        stream_mode = bool(state.get("_stream", False))
        video = state.get("video", {})
        textbook = state.get("textbook", {})

        if not stream_mode:
            params = _current_params(state)
            reply = params.get("override_reply")
            reply = supervisor._finalize(state) or reply or ""
            return {"video": video, "textbook": textbook, "reply_to_user": reply}

        # —— 流式：发“自定义事件” + 同步写回顶层 reply_to_user —— #
        acc = []
        if writer is None:
            writer = get_stream_writer()

        writer({"type": "final_start"})

        for delta in supervisor._finalize_stream(state):
            if not delta:
                continue
            acc.append(delta)
            writer({"type": "final_delta", "delta": delta})
            yield {"reply_to_user": "".join(acc)}

        final_reply = "".join(acc) if acc else ""
        writer({"type": "final_end"})

        return {"video": video, "textbook": textbook, "reply_to_user": final_reply}

    def route(state: RootState) -> str:
        decision = state.get("decision", {})
        if isinstance(decision, Mapping):
            nxt = decision.get("next")
            if isinstance(nxt, str):
                return nxt
        return "finalize"

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("ocr", ocr)
    graph.add_node("build_query", build_query)
    graph.add_node("search_both", search_both)
    graph.add_node("math", math_reasoning)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "ocr": "ocr",
            "build_query": "build_query",
            "search_both": "search_both",
            "math": "math",
            "finalize": "finalize",
        },
    )
    graph.add_edge("ocr", "supervisor")
    graph.add_edge("build_query", "supervisor")
    graph.add_edge("search_both", "supervisor")
    graph.add_edge("math", "supervisor")
    graph.add_edge("finalize", END)

    return graph.compile()
