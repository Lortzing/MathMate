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
    images_b64: List[str] | None = None


def _build_state_from_payload(payload: AskPayload) -> Dict[str, Any]:
    state: Dict[str, Any] = {"user_query": payload.query}
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
    state = _build_state_from_payload(payload)

    def event_stream():
        try:
            for update in root_graph.stream(state, stream_mode="updates"):
                if not isinstance(update, dict):
                    continue

                supervisor_update = update.get("supervisor")
                if supervisor_update and isinstance(supervisor_update, dict):
                    decision = supervisor_update.get("decision")
                    if isinstance(decision, dict):
                        payload = {
                            "type": "action",
                            "data": {
                                "next": decision.get("next"),
                                "params": decision.get("params", {}),
                                "reason": decision.get("reason", ""),
                            },
                        }
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                finalize_update = update.get("finalize")
                if finalize_update and isinstance(finalize_update, dict):
                    result_payload = {
                        "type": "final",
                        "data": {
                            "video": finalize_update.get("video", {}),
                            "textbook": finalize_update.get("textbook", {}),
                            "reply_to_user": finalize_update.get("reply_to_user", ""),
                        },
                    }
                    yield f"data: {json.dumps(result_payload, ensure_ascii=False)}\n\n"
                    break
        finally:
            done_message = {"type": "done"}
            yield f"data: {json.dumps(done_message, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ask-multipart")
async def ask_multipart(
    query: str = Form(...),
    files: List[UploadFile] | None = File(default=None),
) -> Dict[str, Any]:
    images: List[bytes] = []
    if files:
        for file in files:
            images.append(await file.read())
    state: Dict[str, Any] = {"user_query": query}
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
