from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from io import BytesIO

from langgraph.graph import END, START, StateGraph

from app.agents import MathAgent, OCRAgent, TextbookRAG, MockVideoRAG
from .state import RootState


@dataclass
class RootGraphDeps:
    video_rag: MockVideoRAG
    textbook_rag: TextbookRAG
    ocr_agent: OCRAgent
    math_agent: MathAgent


def build_root_graph(deps: RootGraphDeps):
    """Construct and compile the root LangGraph workflow."""

    graph = StateGraph(RootState)

    def ocr(state: RootState) -> RootState:
        images = state.get("images", [])
        if images:
            texts = []
            for img in images:
                try:
                    text = deps.ocr_agent.run(BytesIO(img))
                except Exception:
                    text = ""
                texts.append(text)
            return {"ocr_text": "\n".join([f"No. {i} pictures, content: {txt}" for i, txt in enumerate(texts) if txt])}
        return {}

    def build_query(state: RootState) -> RootState:
        user_q = (state.get("user_query") or "").strip()
        ocr_t = (state.get("ocr_text") or "").strip()
        if ocr_t:
            combined = f"{user_q}\n\n[From OCR]\n{ocr_t}"
        else:
            combined = user_q
        return {"combined_query": combined}

    def search_video(state: RootState) -> RootState:
        query = state["combined_query"]
        try:
            result: Dict[str, Any] = deps.video_rag.search(query)
        except Exception as exc:
            result = {"query": query, "hits": [], "error": str(exc)}
        return {"video": result}

    def search_textbook(state: RootState) -> RootState:
        query = state["combined_query"]
        try:
            result: Dict[str, Any] = deps.textbook_rag.search(query)
        except Exception as exc:
            result = {"query": query, "hits": [], "error": str(exc)}
        return {"textbook": result}

    def join_barrier(state: RootState) -> RootState:
        return {}

    def math_reasoning(state: RootState) -> RootState:
        try:
            explanation = deps.math_agent.explain(
                question=state.get("user_query", "") or "",
                video_ctx=state.get("video", {}) or {},
                textbook_ctx=state.get("textbook", {}) or {},
                ocr_text=state.get("ocr_text", "") or "",
            )
        except Exception as exc:
            explanation = f"（数学讲解生成失败：{exc}）"
        return {"math_explanation": explanation}

    def format_output(state: RootState) -> RootState:
        reply = state.get("math_explanation") or "（暂无可用讲解）"
        state["reply_to_user"] = reply
        return {
            "video": state.get("video", {}),
            "textbook": state.get("textbook", {}),
            "reply_to_user": state.get("reply_to_user", ""),
        }

    graph.add_node("ocr", ocr)
    graph.add_node("build_query", build_query)
    graph.add_node("search_video", search_video)
    graph.add_node("search_textbook", search_textbook)
    graph.add_node("join", join_barrier)
    graph.add_node("math_agent", math_reasoning)
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "ocr")
    graph.add_edge("ocr", "build_query")
    graph.add_edge("build_query", "search_video")
    graph.add_edge("build_query", "search_textbook")
    graph.add_edge("search_video", "join")
    graph.add_edge("search_textbook", "join")
    graph.add_edge("join", "math_agent")
    graph.add_edge("math_agent", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()
