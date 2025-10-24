from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Mapping, Tuple, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from langgraph.config import get_stream_writer

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

    def _compose_query(
        user_q: str,
        ocr_t: str,
        params: Mapping[str, Any] | None = None,
    ) -> str:
        params = params or {}
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
        return combined or user_q

    def _process_video_result(
        raw_result: Any,
        *,
        query: str,
        video_top_k: Any = None,
    ) -> Dict[str, Any]:
        if isinstance(raw_result, Exception):
            logger.warning("Video RAG search failed: %s", raw_result)
            return {"query": query, "hits": [], "error": str(raw_result)}

        video_result: Dict[str, Any]
        if isinstance(raw_result, dict):
            video_result = {"query": query, **raw_result}
            if "query" not in raw_result:
                video_result["query"] = query
        else:
            video_result = {"query": query, "hits": raw_result}

        if isinstance(video_result.get("hits"), list) and video_top_k is not None:
            try:
                k = max(0, int(video_top_k))
            except (TypeError, ValueError):
                k = None
            if k is not None:
                video_result["hits"] = video_result["hits"][:k]

        return video_result

    def _process_textbook_result(
        raw_result: Any,
        *,
        query: str,
        textbook_top_k: Any = None,
    ) -> Dict[str, Any]:
        if isinstance(raw_result, Exception):
            logger.warning("Textbook RAG search failed: %s", raw_result)
            return {"query": query, "content": "", "error": str(raw_result)}

        if isinstance(raw_result, dict):
            textbook_result = {"query": query, **raw_result}
            if "query" not in raw_result:
                textbook_result["query"] = query
        else:
            textbook_result = {"query": query, "content": raw_result}

        if textbook_top_k is not None:
            try:
                k = max(0, int(textbook_top_k))
            except (TypeError, ValueError):
                k = None
            if k is not None:
                content = textbook_result.get("content")
                if isinstance(content, list):
                    textbook_result["content"] = content[:k]

        return textbook_result

    def _auto_search_sync(query: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """同步并发：在线程池里并行跑 video/textbook 检索。"""
        def _run_video():
            try:
                return deps.video_rag.search(query)
            except Exception as exc:
                return exc

        def _run_textbook():
            try:
                return deps.textbook_rag.search(query)
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(_run_video)
            f2 = ex.submit(_run_textbook)
            video_raw = f1.result()
            textbook_raw = f2.result()

        video_result = _process_video_result(video_raw, query=query)
        textbook_result = _process_textbook_result(textbook_raw, query=query)
        return video_result, textbook_result

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

    def _perform_ocr(images: list[bytes], max_images: Any = None) -> Dict[str, Any]:
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

    def bootstrap(state: RootState) -> RootState:
        updates: Dict[str, Any] = {}
        images = state.get("images", [])
        if images:
            ocr_update = _perform_ocr(images)
            if ocr_update:
                updates.update(ocr_update)
                updates["_auto_ocr"] = True

        user_q = (state.get("user_query") or "").strip()
        ocr_t = (updates.get("ocr_text") or state.get("ocr_text") or "").strip()
        combined_query = _compose_query(user_q, ocr_t)
        if combined_query:
            updates["combined_query"] = combined_query
            try:
                video_result, textbook_result = _auto_search_sync(combined_query)
            except Exception as exc:
                logger.warning("Automatic search failed: %s", exc)
            else:
                updates["video"] = video_result
                updates["textbook"] = textbook_result
                updates["_auto_search_results"] = {
                    "video": video_result,
                    "textbook": textbook_result,
                    "query": combined_query,
                }
                updates["_auto_search_seen"] = False
        return updates

    def ocr(state: RootState) -> RootState:
        params = _current_params(state)
        images = state.get("images", [])
        if not images:
            return {}

        result = _perform_ocr(images, params.get("max_images"))
        return result

    def build_query(state: RootState) -> RootState:
        params = _current_params(state)
        user_q = (state.get("user_query") or "").strip()
        ocr_t = (state.get("ocr_text") or "").strip()
        combined = _compose_query(user_q, ocr_t, params)
        return {"combined_query": combined or user_q}

    def search_both(state: RootState) -> RootState:
        params = _current_params(state)
        query = (state.get("combined_query") or state.get("user_query") or "").strip()
        if not query:
            return {}

        auto_results = state.get("_auto_search_results")
        auto_seen = state.get("_auto_search_seen", True)
        if isinstance(auto_results, dict) and not auto_seen:
            video_result = auto_results.get("video") or {}
            textbook_result = auto_results.get("textbook") or {}
            return {
                "video": video_result,
                "textbook": textbook_result,
                "_auto_search_seen": True,
            }

        video_result = deps.video_rag.search(query)
        video_result = _process_video_result(
            video_result,
            query=query,
            video_top_k=params.get("video_top_k"),
        )

        textbook_result = deps.textbook_rag.search(query)
        textbook_result = _process_textbook_result(
            textbook_result,
            query=query,
            textbook_top_k=params.get("textbook_top_k"),
        )

        updates: Dict[str, Any] = {"video": video_result, "textbook": textbook_result}
        if isinstance(auto_results, dict):
            updates.setdefault("_auto_search_results", auto_results)
            updates["_auto_search_seen"] = True
        return updates

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

        params = _current_params(state)

        if not stream_mode:
            override = params.get("override_reply")
            if override is not None:
                reply = override or ""
                relevance = supervisor.assess_retrieval(state)
            else:
                reply, relevance = supervisor.finalize_with_relevance(state)
                reply = reply or ""

            best_video = relevance.get("video") if isinstance(relevance, dict) else {}
            best_textbook = relevance.get("textbook") if isinstance(relevance, dict) else {}
            
            return {
                "video": best_video,
                "textbook": best_textbook,
                "reply_to_user": reply,
            }

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
        writer({"type": "final_end"})
        
        final_reply = "".join(acc) if acc else ""
        # print(final_reply)
        
        
        relevance = supervisor.assess_retrieval(state)
        best_video = relevance.get("video", "") if isinstance(relevance, dict) else {}
        best_textbook = relevance.get("textbook", "") if isinstance(relevance, dict) else {}

        return {
            "video": best_video or {},
            "textbook": best_textbook or {},
            "reply_to_user": final_reply,
        }

    def route(state: RootState) -> str:
        decision = state.get("decision", {})
        if isinstance(decision, Mapping):
            nxt = decision.get("next")
            if isinstance(nxt, str):
                return nxt
        return "finalize"

    graph.add_node("bootstrap", bootstrap)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("ocr", ocr)
    graph.add_node("build_query", build_query)
    graph.add_node("search_both", search_both)
    graph.add_node("math", math_reasoning)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "supervisor")
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
