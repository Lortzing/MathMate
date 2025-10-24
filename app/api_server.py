from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents import (
    TextbookRAG,
    MockVideoRAG,
    OCRAgent,
    MathAgent,
)
from app.graph import RootGraphDeps, build_root_graph
from .graph.state import RootState

app = FastAPI(title="Multi-Agent Root API")

deps = RootGraphDeps(
    video_rag=MockVideoRAG(),
    textbook_rag=TextbookRAG(),
    ocr_agent=OCRAgent(),
    math_agent=MathAgent(),
)
root_graph = build_root_graph(deps)


class AskPayload(BaseModel):
    query: str
    images_b64: List[str] = []


def _build_state_from_payload(payload: AskPayload, _stream=False) -> RootState:
    state = RootState(user_query=payload.query, images=[], ocr_text="", combined_query="", _stream=_stream)
    print(payload)
    if payload.images_b64:
        import base64

        images = [base64.b64decode(item) for item in payload.images_b64 if item]
        if images:
            state["images"] = images
    return state


@app.post("/ask")
async def ask(payload: AskPayload) -> Dict[str, Any]:
    state = _build_state_from_payload(payload)
    output = root_graph.invoke(state)
    return {
        "video": output.get("video", {}),
        "textbook": output.get("textbook", {}),
        "reply_to_user": output.get("reply_to_user", ""),
    }


@app.post("/ask-stream")
async def ask_stream(payload: AskPayload) -> StreamingResponse:
    state = _build_state_from_payload(payload, _stream=True)

    def sse_event(d: dict) -> str:
        return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

    def event_stream():
        last_reply = ""
        final_video, final_textbook = {}, {}

        try:
            for mode, chunk in root_graph.stream(
                state,
                stream_mode=["custom", "updates"],
            ):
                if mode == "custom":
                    t = chunk.get("type")
                    if t == "final_delta":
                        yield sse_event({"type": "final_delta", "data": {"delta": chunk["delta"]}})
                    elif t == "final_start":
                        yield sse_event({"type": "action", "data": {"next": "finalize"}})
                    elif t == "final_end":
                        pass

                elif mode == "updates":
                    upd = chunk
                    sup = upd.get("supervisor")
                    if isinstance(sup, dict):
                        decision = sup.get("decision")
                        if isinstance(decision, dict):
                            yield sse_event({
                                "type": "action",
                                "data": {
                                    "next": decision.get("next"),
                                    "params": decision.get("params", {}),
                                    "reason": decision.get("reason", ""),
                                },
                            })

                    if "reply_to_user" in upd and isinstance(upd["reply_to_user"], str):
                        full = upd["reply_to_user"]
                        if len(full) > len(last_reply):
                            delta = full[len(last_reply):]
                            last_reply = full
                            yield sse_event({"type": "final_delta", "data": {"delta": delta}})

                    if "video" in upd:
                        final_video = upd.get("video", {}) or final_video
                    if "textbook" in upd:
                        final_textbook = upd.get("textbook", {}) or final_textbook

            yield sse_event({
                "type": "final",
                "data": {
                    "video": final_video,
                    "textbook": final_textbook,
                    "reply_to_user": last_reply,
                },
            })

        finally:
            yield sse_event({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



@app.post("/ask-multipart")
async def ask_multipart(
    query: str = Form(...),
    files: List[UploadFile] | None = File(default=None),
) -> Dict[str, Any]:
    images: List[bytes] = []
    if files:
        for file in files:
            images.append(await file.read())
    state = _build_state_from_payload(AskPayload(query=query, images_b64=[]))
    if images:
        state["images"] = images
    output = root_graph.invoke(state)
    return {
        "video": output.get("video", {}),
        "textbook": output.get("textbook", {}),
        "reply_to_user": output.get("reply_to_user", ""),
    }

@app.get("/health")
async def root():
    return {"message": "Hello World", "docs_url": app.docs_url}
